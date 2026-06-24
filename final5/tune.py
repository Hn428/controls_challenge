"""
Batch evaluator + parameter tuner for the controls challenge.

Usage:
  # evaluate a controller over N segments (no plots, just aggregate costs)
  python tune.py eval --controller preview_pid_ff --num_segs 100

  # compare two controllers on the same segments
  python tune.py compare --test preview_pid_ff --baseline pid --num_segs 100

  # grid/random search over preview_pid_ff parameters (sets PIDFF_* env vars)
  python tune.py search --num_segs 80 --iters 60

Parameters are passed to controllers via PIDFF_* environment variables, which
are inherited by the process_map worker processes.
"""
import argparse
import os
import numpy as np
import pandas as pd
from functools import partial
from pathlib import Path
from tqdm.contrib.concurrent import process_map

from tinyphysics import run_rollout

MODEL_PATH = "./models/tinyphysics.onnx"
DATA_PATH = "./data"


def _cost_only(data_path, controller_type, model_path):
  cost, _, _ = run_rollout(data_path, controller_type, model_path, debug=False)
  return cost


def eval_controller(controller, files, workers=16):
  fn = partial(_cost_only, controller_type=controller, model_path=MODEL_PATH)
  results = process_map(fn, files, max_workers=workers, chunksize=4, disable=True)
  df = pd.DataFrame(results)
  return {
    'lataccel_cost': float(df['lataccel_cost'].mean()),
    'jerk_cost': float(df['jerk_cost'].mean()),
    'total_cost': float(df['total_cost'].mean()),
    'total_std': float(df['total_cost'].std()),
    'n': len(df),
  }


def get_files(num_segs, start=0):
  return sorted(Path(DATA_PATH).iterdir())[start:start + num_segs]


def set_params(params):
  for k, v in params.items():
    os.environ[f"PIDFF_{k}"] = str(v)


def fmt(r):
  return (f"lataccel={r['lataccel_cost']:.4f}  jerk={r['jerk_cost']:.3f}  "
          f"total={r['total_cost']:.3f} (std {r['total_std']:.1f}, n={r['n']})")


def cmd_eval(args):
  files = get_files(args.num_segs)
  r = eval_controller(args.controller, files)
  print(f"{args.controller}: {fmt(r)}")


def cmd_compare(args):
  for n in (args.num_segs if isinstance(args.num_segs, list) else [args.num_segs]):
    files = get_files(n, start=args.start)
    rt = eval_controller(args.test, files)
    rb = eval_controller(args.baseline, files)
    print(f"--- {n} segments (start={args.start}) ---")
    print(f"  baseline {args.baseline:18s}: {fmt(rb)}")
    print(f"  test     {args.test:18s}: {fmt(rt)}")
    delta = rt['total_cost'] - rb['total_cost']
    pct = 100 * delta / rb['total_cost']
    verdict = "BEATS" if delta < 0 else "WORSE THAN"
    print(f"  => test {verdict} baseline by {abs(delta):.2f} ({abs(pct):.1f}%)")


def cmd_validate(args):
  """Evaluate the same controller on a train range and a disjoint holdout
  range to detect overfitting of tuned parameters."""
  train = get_files(args.num_segs, start=0)
  holdout = get_files(args.num_segs, start=args.holdout_start)
  rt = eval_controller(args.controller, train)
  rh = eval_controller(args.controller, holdout)
  print(f"  train   [0:{args.num_segs}]                : {fmt(rt)}")
  print(f"  holdout [{args.holdout_start}:{args.holdout_start+args.num_segs}]    : {fmt(rh)}")
  gap = rh['total_cost'] - rt['total_cost']
  print(f"  => generalization gap: {gap:+.2f} "
        f"({'OK' if abs(gap) < 0.05*rt['total_cost'] else 'CHECK'})")


# Search spaces per controller. (prefix, {param: (lo, hi)}, int_params)
SEARCH_SPACES = {
  'preview_pid_ff': ('PIDFF_', {
    'FF_GAIN':      (0.45, 0.62),
    'FF_LOOKAHEAD': (0, 4),
    'KP':           (0.02, 0.25),
    'KI':           (0.0, 0.15),
    'KD':           (-0.10, 0.02),
    'I_CLIP':       (1.0, 4.0),
  }, {'FF_LOOKAHEAD'}),
  'mpc': ('MPC_', {
    'H':       (15, 40),
    'W_JERK':  (3.0, 20.0),
    'W_DU':    (0.0, 5.0),
    'W_U':     (1e-4, 0.05),
    'D_GAIN':  (0.0, 0.20),
  }, {'H'}),
}
# Backwards-compat default (preview_pid_ff)
SEARCH_SPACE = SEARCH_SPACES['preview_pid_ff'][1]


def sample_params(rng, space, int_params):
  p = {}
  for k, (lo, hi) in space.items():
    if k in int_params:
      p[k] = int(rng.integers(lo, hi + 1))
    else:
      p[k] = round(float(rng.uniform(lo, hi)), 5)
  return p


def cmd_search(args):
  rng = np.random.default_rng(args.seed)
  files = get_files(args.num_segs)
  prefix, space, int_params = SEARCH_SPACES[args.controller]
  best = None
  history = []
  # seed with the current defaults first
  candidates = [None] + [sample_params(rng, space, int_params) for _ in range(args.iters)]
  for i, params in enumerate(candidates):
    if params is not None:
      for k, v in params.items():
        os.environ[f"{prefix}{k}"] = str(v)
    else:
      for k in space:                       # clear so controller defaults apply
        os.environ.pop(f"{prefix}{k}", None)
    r = eval_controller(args.controller, files, workers=args.workers)
    tag = "defaults" if params is None else str(params)
    history.append((r['total_cost'], params, r))
    marker = ""
    if best is None or r['total_cost'] < best[0]:
      best = (r['total_cost'], params, r)
      marker = "  <-- new best"
    print(f"[{i:2d}/{len(candidates)-1}] total={r['total_cost']:.3f} "
          f"(lat={r['lataccel_cost']:.3f} jerk={r['jerk_cost']:.2f}) {tag}{marker}")
  print("\n=== BEST ===")
  print(f"total_cost={best[0]:.3f}  {fmt(best[2])}")
  print(f"params={best[1]}")
  # write best params to a file for reference
  outfile = f"best_params_{args.controller}.txt"
  with open(outfile, "w") as f:
    f.write(f"# best total_cost={best[0]:.4f} on {args.num_segs} segs\n")
    f.write(repr(best[1]) + "\n")
    f.write("\n# full history (sorted):\n")
    for tc, p, r in sorted(history, key=lambda x: x[0]):
      f.write(f"{tc:.3f}  {p}\n")
  print(f"Wrote {outfile}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="cmd", required=True)

  pe = sub.add_parser("eval")
  pe.add_argument("--controller", default="preview_pid_ff")
  pe.add_argument("--num_segs", type=int, default=100)
  pe.set_defaults(func=cmd_eval)

  pc = sub.add_parser("compare")
  pc.add_argument("--test", default="preview_pid_ff")
  pc.add_argument("--baseline", default="pid")
  pc.add_argument("--num_segs", type=int, nargs="+", default=[100])
  pc.add_argument("--start", type=int, default=0)
  pc.set_defaults(func=cmd_compare)

  pv = sub.add_parser("validate")
  pv.add_argument("--controller", default="preview_pid_ff")
  pv.add_argument("--num_segs", type=int, default=200)
  pv.add_argument("--holdout_start", type=int, default=1000)
  pv.set_defaults(func=cmd_validate)

  ps = sub.add_parser("search")
  ps.add_argument("--controller", default="preview_pid_ff", choices=list(SEARCH_SPACES))
  ps.add_argument("--num_segs", type=int, default=80)
  ps.add_argument("--iters", type=int, default=60)
  ps.add_argument("--workers", type=int, default=16)
  ps.add_argument("--seed", type=int, default=0)
  ps.set_defaults(func=cmd_search)

  args = parser.parse_args()
  args.func(args)
