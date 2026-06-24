import os
from . import BaseController


def _p(name, default):
  """Read a float parameter from the environment (so the tuner can sweep
  values across the multiprocess `run_rollout` workers), else use default."""
  v = os.environ.get(f"PIDFF_{name}")
  return float(v) if v is not None else default


class Controller(BaseController):
  """
  Feedforward + PID controller with future-plan preview.

  Three ideas, layered:

  1. Feedforward. The tinyphysics model behaves close to a static map
     `lataccel ~= steer / ff_gain + roll_lataccel` (R^2 ~= 0.85 fit on the
     logged human steer commands). Inverting it gives the steer needed to
     hold a target *before* any error builds up:
         steer_ff = ff_gain * (target - roll_lataccel)
     This removes most of the tracking error that a pure PID has to chase,
     which is what drives the (heavily weighted) lataccel cost.

  2. Preview / lag compensation. The plant responds to steer with a short
     lag, so we feedforward against a target a couple of frames into the
     future (from `future_plan`) instead of the instantaneous one. This
     lines the response up with the desired trajectory and cuts both
     tracking error and jerk.

  3. PID feedback on the residual. A light PID cleans up the ~15% the
     feedforward map doesn't capture (model nonlinearity, roll, noise).
  """

  def __init__(self):
    # Feedforward
    self.ff_gain = _p("FF_GAIN", 0.535)
    self.ff_lookahead = int(_p("FF_LOOKAHEAD", 2))   # frames ahead for FF target
    # PID feedback (on residual error)
    self.kp = _p("KP", 0.115)
    self.ki = _p("KI", 0.060)
    self.kd = _p("KD", -0.020)
    # Integral anti-windup clamp
    self.i_clip = _p("I_CLIP", 2.5)

    self.error_integral = 0.0
    self.prev_error = 0.0

  def _ff_target(self, target_lataccel, roll_lataccel, future_plan):
    """Pick the (target, roll) pair `ff_lookahead` frames ahead, if available."""
    la = future_plan.lataccel
    rl = future_plan.roll_lataccel
    k = self.ff_lookahead
    if k > 0 and len(la) >= k:
      return la[k - 1], rl[k - 1]
    return target_lataccel, roll_lataccel

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    roll_lataccel = state.roll_lataccel

    # 1 + 2: feedforward against a slightly anticipated target
    ff_target, ff_roll = self._ff_target(target_lataccel, roll_lataccel, future_plan)
    steer_ff = self.ff_gain * (ff_target - ff_roll)

    # 3: PID on the instantaneous tracking error
    error = target_lataccel - current_lataccel
    self.error_integral += error
    self.error_integral = max(-self.i_clip, min(self.i_clip, self.error_integral))
    error_diff = error - self.prev_error
    self.prev_error = error
    steer_fb = self.kp * error + self.ki * self.error_integral + self.kd * error_diff

    return steer_ff + steer_fb
