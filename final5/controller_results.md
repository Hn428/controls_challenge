# Controller Results — mpc

Model Predictive Controller (per-step direct quadratic optimization) for the comma controls challenge.

Cost = `lataccel_cost*50 + jerk_cost` (lower is better). Winner selected on the **1000-segment** validation score.

## Final chosen configuration

```python
H = 38
W_JERK = 7.45933
W_DU = 3.75182
W_U = 0.01409
D_GAIN = 0.09704
```

## Final 5000-segment result (submission scale)

| Controller | lataccel_cost | jerk_cost | total_cost |
|---|---|---|---|
| PID (baseline) | 1.7148 | 25.606 | 111.345 |
| **mpc** | 0.8557 | 19.274 | **62.057** |

**Net: -49.29 total_cost (-44.3%) vs PID** over 5000 segments.

## Validation across segment counts (winner vs PID)

| Segments | PID total | mpc total | delta | % |
|---|---|---|---|---|
| 100 | 80.725 | 49.709 | -31.02 | -38.4% |
| 500 | 103.739 | 58.768 | -44.97 | -43.3% |
| 1000 | 106.630 | 60.115 | -46.51 | -43.6% |
| 5000 | 111.345 | 62.057 | -49.29 | -44.3% |

## All top configs evaluated (selection table)

| cfg | params | 100-seg total | 500-seg total | 1000-seg total |
|---|---|---|---|---|
| 0 | `{'H': 30, 'W_JERK': 3.45024, 'W_DU': 2.23406, 'W_U': 0.01866, 'D_GAIN': 0.09541}` | 48.081 | 59.334 | 62.071 |
| 1 | `{'H': 34, 'W_JERK': 8.15965, 'W_DU': 4.99513, 'W_U': 0.01318, 'D_GAIN': 0.16981}` | 49.392 | 60.709 | 63.064 |
| 2 | `{'H': 23, 'W_JERK': 10.18619, 'W_DU': 2.94751, 'W_U': 0.00132, 'D_GAIN': 0.13469}` | 49.679 | 60.891 | 61.680 |
| 3 ⭐ | `{'H': 38, 'W_JERK': 7.45933, 'W_DU': 3.75182, 'W_U': 0.01409, 'D_GAIN': 0.09704}` | 49.709 | 58.768 | 60.115 |
| 4 | `{'H': 19, 'W_JERK': 11.02247, 'W_DU': 1.37024, 'W_U': 0.00045, 'D_GAIN': 0.12914}` | 49.869 | 60.031 | 61.473 |

## Method

1. Identified the plant open-loop from the tinyphysics simulator with smooth random steer excitation: an ARX model with a ~2-step input dead-time, steady gain u→la≈1.6, roll passthrough≈0.95 (`model_coeffs.json`).

2. Built an MPC that, each step, expresses the previewed lataccel trajectory as `la = G u + f` (affine in the steer vector) and solves the quadratic challenge cost in closed form; a slow offset-free disturbance estimate rejects model bias.

3. Random-searched 5 weights (H, W_JERK, W_DU, W_U, D_GAIN) over 50 segments (`tune.py search --controller mpc`).

4. Re-validated the top 5 configs on 100, 500, 1000 segments and selected the winner on the 1000-segment score to avoid overfitting the small search set.

5. Confirmed the winner on 5000 segments (challenge submission scale).
