<div align="center">
<h1>comma Controls Challenge v2</h1>


<h3>
  <a href="https://comma.ai/leaderboard">Leaderboard</a>
  <span> · </span>
  <a href="https://comma.ai/jobs">comma.ai/jobs</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
  <span> · </span>
  <a href="https://x.com/comma_ai">X</a>
</h3>

</div>

## 🏎️ My Solution — MPC Controller (`total_cost ≈ 62 vs PID's 111`)

This fork adds a **Model Predictive Controller** ([`controllers/mpc.py`](controllers/mpc.py)) that
beats the PID baseline by **~44%** at submission scale (5000 segments):

| Controller | lataccel_cost | jerk_cost | **total_cost** |
|---|---|---|---|
| PID (baseline) | 1.71 | 25.6 | 111.3 |
| **MPC (mine)** | 0.86 | 19.3 | **62.1** |

It identifies the simulator's plant dynamics (a linear ARX model with a ~2-step steer dead-time,
found via open-loop system ID), then each step solves a small quadratic program over the
`future_plan` preview that directly minimizes the challenge cost. See
**[`submission/README.md`](submission/README.md)** for the full write-up and
**[`controller_results.md`](controller_results.md)** for results across 100/500/1000/5000 segments.
A simpler feedforward+preview+PID controller is also included
([`controllers/preview_pid_ff.py`](controllers/preview_pid_ff.py)), along with the tuning
(`tune.py`) and validation (`validate_top.py`) tooling.

### Key files in this fork

| File | Description |
|---|---|
| [`controllers/mpc.py`](controllers/mpc.py) | The MPC controller (self-contained — plant model + tuned weights baked in). |
| [`controllers/preview_pid_ff.py`](controllers/preview_pid_ff.py) | Feedforward + future-plan preview + PID controller (fallback). |
| [`model_coeffs.json`](model_coeffs.json) | Plant model from open-loop system ID (optional; `mpc.py` also hardcodes it). |
| [`tune.py`](tune.py) | Batch eval / compare / validate / random-search tooling. |
| [`validate_top.py`](validate_top.py) | Top-K → 100/500/1000-segment validation → winner selection → 5000-seg final. |
| [`controller_results.md`](controller_results.md) | Results tables + method write-up. |
| `report.html` | Official `eval.py` report (mpc vs pid, 5000 segments). |

**Reproduce the result:**

```bash
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 \
  --test_controller mpc --baseline_controller pid
```

The five files used for the challenge submission are collected in
[`final5/`](final5/): `report.html`, `mpc.py`, `controller_results.md`,
`README.md`, and `tune.py`. The controller (`mpc.py`) runs standalone; the rest
document the method and results.

---

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.

## Getting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual car and road states from [openpilot](https://github.com/commaai/openpilot) users.

```
# install required packages
# recommended python==3.11
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid
```

There are some other scripts to help you get aggregate metrics:
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. Its inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`), then it predicts the resultant lateral acceleration of the car.

## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.

## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(\mathrm{actual{\textunderscore}lat{\textunderscore}accel} - \mathrm{target{\textunderscore}lat{\textunderscore}accel})^2}{\text{steps}} * 100$
- `jerk_cost`: $\dfrac{(\Sigma( \mathrm{actual{\textunderscore}lat{\textunderscore}accel_t} - \mathrm{actual{\textunderscore}lat{\textunderscore}accel_{t-1}}) / \Delta \mathrm{t} )^{2}}{\text{steps} - 1} * 100$

It is important to minimize both costs. `total_cost`: $(\mathrm{lat{\textunderscore}accel{\textunderscore}cost} * 50) + \mathrm{jerk{\textunderscore}cost}$

## Submission
Run the following command, then submit `report.html` and your code to [this form](https://forms.gle/US88Hg7UR6bBuW3BA).

Competitive scores (`total_cost<100`) will be added to the leaderboard

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller pid
```

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma

Like this sort of stuff? You might want to work at comma!
[comma.ai/jobs](https://comma.ai/jobs)
