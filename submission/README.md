# Comma Controls Challenge — MPC Submission

A **Model Predictive Controller (MPC)** for the comma controls challenge that drives the
simulated car to track a target lateral acceleration. At submission scale (5000 segments)
it scores **`total_cost` ≈ 62.1 vs the PID baseline's ≈ 111.3 — a 44% improvement**, well
inside leaderboard-competitive territory (qualifying threshold is `total_cost < 100`).

| Controller | lataccel_cost | jerk_cost | **total_cost** (5000 segs) |
|---|---|---|---|
| PID (baseline) | 1.71 | 25.6 | 111.3 |
| **MPC (this submission)** | 0.86 | 19.3 | **62.1** |

## Contents

| File | Description |
|---|---|
| `controllers/mpc.py` | The controller. Per-step direct quadratic optimization over the future-plan preview. Self-contained — tuned weights and the identified plant model are baked in as defaults. |
| `controllers/__init__.py` | `BaseController` (required import). |
| `model_coeffs.json` | The linear plant model identified from the simulator (open-loop system ID). |
| `tune.py` | Batch eval / compare / validate / random-search tooling. |
| `controller_results.md` | Full results tables (100/500/1000/5000 segments) and method writeup. |
| `report.html` | Official challenge report (`eval.py`, test=mpc vs baseline=pid, 5000 segments). |

## How it works

The controller is a per-step quadratic program (MPC):

1. **Plant model.** The tinyphysics simulator was identified **open-loop** by driving it with
   smooth random steer excitation (closed-loop PID logs are biased and hide the real dynamics).
   The plant is a linear ARX model with a **~2-step steer dead-time** — a steer change barely
   moves lateral accel for two frames, then ramps in. Steady gain steer→lataccel ≈ 1.6, road-roll
   passthrough ≈ 0.95.

2. **Quadratic optimization.** Over the 50-step `future_plan` preview, the future lataccel
   trajectory is an affine function of the steer vector, `la = G·u + f`. The challenge cost is
   quadratic (`50·tracking² + jerk²`, with jerk weighted ~2× tracking per step), so the optimum
   is the solution of a single linear system. `G` and the Hessian inverse depend only on the
   model and weights, so they're precomputed once; each step only rebuilds the affine term `f`
   from the measured state and the preview, then applies the first optimal steer.

3. **Stability.** Early versions oscillated badly. The fixes: model the dead-time correctly,
   penalize steer-rate for smoothness, and use a **slow** offset-free disturbance estimate
   (fast output feedback is unstable under dead-time). Heavy jerk weighting keeps the MPC from
   fighting the simulator's stochastic sampling noise.

## Reproduce

```bash
# from the repo root, with the challenge environment installed
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller mpc

# official report (submission scale)
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 \
  --test_controller mpc --baseline_controller pid
```

Tuning was done with `tune.py search --controller mpc` (random search over the MPC weights),
followed by re-validation of the top configs on 100/500/1000 segments and selecting the most
robust config on the 1000-segment score (`validate_top.py`) to avoid overfitting the search set.
