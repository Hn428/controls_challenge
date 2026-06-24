"""
Post-search validation pipeline.

After `tune.py search` writes best_params.txt, this script:
  1. Takes the top-K parameter sets from best_params.txt.
  2. Evaluates each on 100 / 500 / 1000 segments.
  3. Compares each against PID on the same segment counts.
  4. Picks the config with the best *validation* score (largest segment count
     = most reliable), not necessarily the best 50-segment search score.
  5. Runs the chosen config on 5000 segments (vs PID).
  6. Writes config, scores, and comparison summary to controller_results.md.

Usage:
  python validate_top.py --topk 5 --val_counts 100 500 1000 --final 5000
"""
import argparse
import ast
import os

from tune import eval_controller, get_files, SEARCH_SPACES

RESULTS_FILE = "controller_results.md"

# set per-run in main()
CONTROLLER = "mpc"
PREFIX = "MPC_"
SPACE_KEYS = []


def parse_top_params(path, topk):
  """Read the sorted history section of best_params.txt and return up to
  `topk` (cost, params_dict) entries. `None` params -> controller defaults."""
  rows = []
  in_history = False
  with open(path) as f:
    for line in f:
      line = line.rstrip("\n")
      if line.startswith("# full history"):
        in_history = True
        continue
      if not in_history or not line.strip() or line.startswith("#"):
        continue
      cost_str, rest = line.split(None, 1)
      params = ast.literal_eval(rest)
      if params is None:
        params = {}  # controller defaults
      rows.append((float(cost_str), params))
  # de-duplicate identical param sets, keep best-cost order (already sorted)
  seen, uniq = set(), []
  for cost, p in rows:
    key = tuple(sorted(p.items()))
    if key in seen:
      continue
    seen.add(key)
    uniq.append((cost, p))
  return uniq[:topk]


def clear_params():
  for k in SPACE_KEYS:
    os.environ.pop(f"{PREFIX}{k}", None)


def eval_with_params(params, files):
  clear_params()
  if params:
    for k, v in params.items():
      os.environ[f"{PREFIX}{k}"] = str(v)
  return eval_controller(CONTROLLER, files)


def main():
  global CONTROLLER, PREFIX, SPACE_KEYS
  ap = argparse.ArgumentParser()
  ap.add_argument("--controller", default="mpc", choices=list(SEARCH_SPACES))
  ap.add_argument("--best_file", default=None)
  ap.add_argument("--topk", type=int, default=5)
  ap.add_argument("--val_counts", type=int, nargs="+", default=[100, 500, 1000])
  ap.add_argument("--final", type=int, default=5000)
  ap.add_argument("--select_on", type=int, default=None,
                  help="segment count to select the winner on (default: max val_count)")
  args = ap.parse_args()

  CONTROLLER = args.controller
  PREFIX, space, _ = SEARCH_SPACES[args.controller]
  SPACE_KEYS = list(space)
  best_file = args.best_file or f"best_params_{args.controller}.txt"

  select_on = args.select_on or max(args.val_counts)

  top = parse_top_params(best_file, args.topk)
  print(f"Loaded top-{len(top)} configs from {best_file}")
  for i, (c, p) in enumerate(top):
    print(f"  [{i}] search_cost(50segs)={c:.3f}  {p}")

  # Precompute PID baselines once per segment count (reused across configs).
  print("\n=== PID baselines ===")
  pid_baseline = {}
  for n in args.val_counts:
    clear_params()
    r = eval_controller("pid", get_files(n))
    pid_baseline[n] = r
    print(f"  PID @ {n:5d}: total={r['total_cost']:.3f} "
          f"(lat={r['lataccel_cost']:.4f} jerk={r['jerk_cost']:.3f})")

  # Evaluate each top config across all validation segment counts.
  print("\n=== Validating top configs ===")
  records = []  # list of dicts: {idx, params, search_cost, scores:{n:result}}
  for i, (search_cost, params) in enumerate(top):
    scores = {}
    for n in args.val_counts:
      r = eval_with_params(params, get_files(n))
      scores[n] = r
      delta = r['total_cost'] - pid_baseline[n]['total_cost']
      print(f"  cfg[{i}] @ {n:5d}: total={r['total_cost']:.3f} "
            f"(vs PID {pid_baseline[n]['total_cost']:.3f}, {delta:+.2f})")
    records.append({'idx': i, 'params': params, 'search_cost': search_cost, 'scores': scores})

  # Pick winner by the selection segment count.
  winner = min(records, key=lambda rec: rec['scores'][select_on]['total_cost'])
  print(f"\n=== WINNER: cfg[{winner['idx']}] selected on {select_on} segments ===")
  print(f"  params = {winner['params']}")
  print(f"  {select_on}-seg total = {winner['scores'][select_on]['total_cost']:.3f}")

  # Final large-scale run for winner + PID.
  print(f"\n=== Final run @ {args.final} segments ===")
  final_files = get_files(args.final)
  clear_params()
  pid_final = eval_controller("pid", final_files)
  print(f"  PID  @ {args.final}: total={pid_final['total_cost']:.3f}")
  winner_final = eval_with_params(winner['params'], final_files)
  print(f"  test @ {args.final}: total={winner_final['total_cost']:.3f}")

  write_report(args, top, pid_baseline, records, winner, pid_final, winner_final, select_on)
  print(f"\nWrote {RESULTS_FILE}")


def write_report(args, top, pid_baseline, records, winner, pid_final, winner_final, select_on):
  name = CONTROLLER
  L = []
  L.append(f"# Controller Results — {name}\n")
  if name == "mpc":
    L.append("Model Predictive Controller (per-step direct quadratic optimization) "
             "for the comma controls challenge.\n")
  else:
    L.append("Feedforward + future-plan preview + PID controller for the comma controls challenge.\n")
  L.append(f"Cost = `lataccel_cost*50 + jerk_cost` (lower is better). "
           f"Winner selected on the **{select_on}-segment** validation score.\n")

  L.append("## Final chosen configuration\n")
  L.append("```python")
  for k in SPACE_KEYS:
    if winner['params'] and k in winner['params']:
      L.append(f"{k} = {winner['params'][k]}")
    else:
      L.append(f"{k} = <controller default>")
  L.append("```")
  if not winner['params']:
    L.append("\n(Winner was the controller's default parameters.)")
  L.append("")

  L.append(f"## Final {args.final}-segment result (submission scale)\n")
  L.append("| Controller | lataccel_cost | jerk_cost | total_cost |")
  L.append("|---|---|---|---|")
  L.append(f"| PID (baseline) | {pid_final['lataccel_cost']:.4f} | {pid_final['jerk_cost']:.3f} | {pid_final['total_cost']:.3f} |")
  L.append(f"| **{name}** | {winner_final['lataccel_cost']:.4f} | {winner_final['jerk_cost']:.3f} | **{winner_final['total_cost']:.3f}** |")
  d = winner_final['total_cost'] - pid_final['total_cost']
  pct = 100 * d / pid_final['total_cost']
  L.append("")
  L.append(f"**Net: {d:+.2f} total_cost ({pct:+.1f}%) vs PID** over {args.final} segments.\n")

  L.append("## Validation across segment counts (winner vs PID)\n")
  L.append(f"| Segments | PID total | {name} total | delta | % |")
  L.append("|---|---|---|---|---|")
  for n in args.val_counts:
    pidc = pid_baseline[n]['total_cost']
    testc = winner['scores'][n]['total_cost']
    dd = testc - pidc
    L.append(f"| {n} | {pidc:.3f} | {testc:.3f} | {dd:+.2f} | {100*dd/pidc:+.1f}% |")
  L.append(f"| {args.final} | {pid_final['total_cost']:.3f} | {winner_final['total_cost']:.3f} | "
           f"{d:+.2f} | {pct:+.1f}% |")
  L.append("")

  L.append("## All top configs evaluated (selection table)\n")
  cols = "| cfg | params | " + " | ".join(f"{n}-seg total" for n in args.val_counts) + " |"
  L.append(cols)
  L.append("|" + "---|" * (2 + len(args.val_counts)))
  for rec in records:
    mark = " ⭐" if rec['idx'] == winner['idx'] else ""
    pstr = rec['params'] if rec['params'] else "defaults"
    cells = " | ".join(f"{rec['scores'][n]['total_cost']:.3f}" for n in args.val_counts)
    L.append(f"| {rec['idx']}{mark} | `{pstr}` | {cells} |")
  L.append("")

  L.append("## Method\n")
  if name == "mpc":
    L.append("1. Identified the plant open-loop from the tinyphysics simulator with smooth "
             "random steer excitation: an ARX model with a ~2-step input dead-time, steady "
             "gain u→la≈1.6, roll passthrough≈0.95 (`model_coeffs.json`).\n")
    L.append("2. Built an MPC that, each step, expresses the previewed lataccel trajectory as "
             "`la = G u + f` (affine in the steer vector) and solves the quadratic challenge "
             "cost in closed form; a slow offset-free disturbance estimate rejects model bias.\n")
    L.append(f"3. Random-searched {len(SPACE_KEYS)} weights ({', '.join(SPACE_KEYS)}) over 50 "
             "segments (`tune.py search --controller mpc`).\n")
  else:
    L.append("1. Derived feedforward law `steer ≈ 0.55·(target − roll_lataccel)` from the data.\n")
    L.append("2. Random-searched the PID+FF parameters over 50 segments.\n")
  L.append(f"4. Re-validated the top {args.topk} configs on {', '.join(map(str, args.val_counts))} "
           f"segments and selected the winner on the {select_on}-segment score to avoid "
           "overfitting the small search set.\n")
  L.append(f"5. Confirmed the winner on {args.final} segments (challenge submission scale).\n")

  with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(L))


if __name__ == "__main__":
  main()
