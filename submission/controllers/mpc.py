import os
import json
import numpy as np
from pathlib import Path
from . import BaseController


def _p(name, default):
  """Read a float parameter from the environment (so the tuner can sweep
  values across the multiprocess `run_rollout` workers), else use default."""
  v = os.environ.get(f"MPC_{name}")
  return float(v) if v is not None else default


# ---------------------------------------------------------------------------
# Linear ARX plant model, identified OPEN-LOOP from the tinyphysics simulator
# driven with smooth random steer excitation (unbiased by any controller):
#
#   la[t] = sum_i A[i]*la[t-1-i] + sum_j B[j]*u[t-1-j] + C*roll[t] + BIAS
#
# The simulator has a ~2-step input dead-time (a steer change barely moves
# lataccel for two frames, then ramps in), captured by the leading B lags.
# Steady gain u->la ~= 1.6, roll->la ~= 0.95.  Coeffs are loaded from
# model_coeffs.json if present (written by the sysid step), else these
# defaults are used.
# ---------------------------------------------------------------------------
_DEFAULT = dict(
  a=[0.8239692454431299, -0.09173469776305683, 0.12847573391186487],
  b=[0.12486617485028804, 0.18812621157343132, 0.25994751666409605,
     0.03204340070925914, -0.11536246348045474, -0.27202491197976847],
  c=0.13179299245411363,
  bias=4.6618826181784234e-05,
)


def _load_model():
  p = Path(__file__).resolve().parent.parent / "model_coeffs.json"
  if p.exists():
    try:
      d = json.load(open(p))
      return np.array(d["a"]), np.array(d["b"]), float(d["c"]), float(d["bias"])
    except Exception:
      pass
  return (np.array(_DEFAULT["a"]), np.array(_DEFAULT["b"]),
          _DEFAULT["c"], _DEFAULT["bias"])


A_COEF, B_COEF, C_COEF, BIAS = _load_model()
NA = len(A_COEF)
NB = len(B_COEF)


class Controller(BaseController):
  """
  Model Predictive Controller — per-step direct quadratic optimization.

  Over the 50-step `future_plan` preview the future lateral-accel trajectory
  is an affine function of the chosen steer vector, la = G u + f, because the
  identified plant model is linear (ARX with input dead-time). The challenge
  cost is quadratic,

      J = w_track * sum (la[k]-target[k])^2 + w_jerk * sum (la[k]-la[k-1])^2
          + w_du * sum (u[k]-u[k-1])^2,

  so the optimum is the solution of a single linear system

      (w_track G'G + w_jerk (EG)'(EG) + w_du D'D + w_u I) u* = -(linear term).

  G and the Hessian inverse depend only on the model + weights and are built
  once; each step only the affine term f is rebuilt from the measured state
  and the preview, then u*[0] is applied.

  A slow offset-free disturbance estimate d (EWMA of the one-step prediction
  residual) rejects steady model/roll bias. It is intentionally slow: the
  plant's input dead-time makes fast output feedback oscillate.
  """

  def __init__(self):
    # Defaults are the tuned winner: random search over 50 segments, then the
    # top-5 re-validated on 100/500/1000 segments and the most robust config
    # chosen on the 1000-segment score (5000-seg total 62.1 vs PID 111.3).
    self.H = int(_p("H", 38))
    self.w_track = _p("W_TRACK", 1.0)
    self.w_jerk = _p("W_JERK", 7.45933)
    self.w_du = _p("W_DU", 3.75182)      # steer-rate penalty (smoothness)
    self.w_u = _p("W_U", 0.01409)        # tiny ridge for conditioning
    self.d_gain = _p("D_GAIN", 0.09704)  # SLOW disturbance EWMA (dead-time safe)
    H = self.H
    a, b = A_COEF, B_COEF

    # G[mm, j]: dependence of y_{mm+1}=la[t+mm] on decision U[j]=u[t+j].
    # Input delay => U[0] (action now) first appears in y2 via b[0] (lag 1).
    G = np.zeros((H, H))
    Grows = []  # G rows for y_1..y_H; y_k for k<=0 are constants (zero rows)
    for mm in range(H):                 # y_{mm+1}, array index mm
      row = np.zeros(H)
      for i in range(NA):               # AR: a[i] * y_{mm+1-(i+1)} = y_{mm-i}
        k = mm - i                       # array index of y_{mm-i}; >=0 -> a horizon var
        if k - 1 >= 0:
          row += a[i] * G[k - 1]
      for j in range(NB):               # control: b[j] * u[t + mm - 1 - j]
        dec = mm - 1 - j                 # decision index
        if dec >= 0:
          row[dec] += b[j]
      G[mm] = row
    self.G = G

    # E: first difference, diff[k]=la[k]-la[k-1] (la[-1]=current).
    E = np.eye(H)
    for k in range(1, H):
      E[k, k - 1] = -1.0
    self.E = E
    # D: control rate, du[k]=u[k]-u[k-1] (u[-1]=last applied action).
    D = np.eye(H)
    for k in range(1, H):
      D[k, k - 1] = -1.0
    self.D = D

    Hess = (self.w_track * G.T @ G
            + self.w_jerk * (E @ G).T @ (E @ G)
            + self.w_du * D.T @ D
            + self.w_u * np.eye(H))
    self.H_inv = np.linalg.inv(Hess)
    self.A_t = self.w_track * G.T
    self.A_j = self.w_jerk * (E @ G).T
    self.A_d = self.w_du * D.T

    # History (most-recent-first): la_hist[0]=la[t-1]=current; u_hist[0]=u[t-1].
    self.la_hist = None
    self.u_hist = [0.0] * (NB + 1)
    self.d = 0.0
    self.pred = None

  def _pad(self, seq, n, fallback):
    out = list(seq[:n])
    if not out:
      out = [fallback]
    while len(out) < n:
      out.append(out[-1])
    return np.array(out, dtype=float)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    H = self.H
    roll_now = state.roll_lataccel
    cur = current_lataccel

    if self.la_hist is None:
      self.la_hist = [cur] * (NA + 1)
      self.pred = cur

    # slow offset-free disturbance update
    self.d += self.d_gain * (cur - self.pred)
    # prepend the fresh measurement: la_hist[0]=la[t-1]=cur, la_hist[1]=la[t-2]..
    self.la_hist = [cur] + self.la_hist[:NA]

    tgt = np.empty(H); roll = np.empty(H)
    tgt[0] = target_lataccel; roll[0] = roll_now
    tgt[1:] = self._pad(future_plan.lataccel, H - 1, target_lataccel)
    roll[1:] = self._pad(future_plan.roll_lataccel, H - 1, roll_now)

    # affine offset f: la = G u + f (control terms in f are the known past u's)
    f = np.zeros(H)
    for mm in range(H):
      val = C_COEF * roll[mm] + BIAS + self.d
      for i in range(NA):               # AR over known/affine past outputs
        k = mm - i                       # y_{mm-i}
        if k - 1 >= 0:
          val += A_COEF[i] * f[k - 1]
        else:                            # y_{<=0}: known measurement
          val += A_COEF[i] * self.la_hist[-k]  # k<=0 -> la_hist[0..NA-1]
      for j in range(NB):               # known past actions u[t+mm-1-j]
        dec = mm - 1 - j
        if dec < 0:
          val += B_COEF[j] * self.u_hist[-dec - 1]   # dec<0 -> u_hist[0..]
      f[mm] = val

    g = np.zeros(H)
    g[0] = -cur
    linear = (self.A_t @ (f - tgt)
              + self.A_j @ (self.E @ f + g)
              + self.A_d @ self._du_const())
    u = -self.H_inv @ linear
    action = float(np.clip(u[0], -2.0, 2.0))

    self.pred = float(f[0] + self.G[0] @ u)  # G[0] is zero row (delay) -> f[0]

    self.u_hist = [action] + self.u_hist[:NB]
    return action

  def _du_const(self):
    # constant part of the steer-rate term: du[0] = u[0]-u[-1] -> -u_hist[0]
    g = np.zeros(self.H)
    g[0] = -self.u_hist[0]
    return g
