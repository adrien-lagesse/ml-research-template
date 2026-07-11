# dummy_regression

## Question

None, scientifically. This experiment exists to exercise the whole scaffold
end to end: the `core` contracts, the Hydra config, the training loop, the
eval schedules, and the loggers. If it misbehaves, the scaffold is broken,
not the science.

## Setup

Synthetic linear regression from `dummy_regression.SyntheticRegression`:
targets are `input @ w + b` plus Gaussian noise with `noise_std = 0.1`.
An MLP (`model/small_mlp.yaml`, with `model/big_mlp.yaml` as a capacity
check) trains with AdamW and a cosine schedule for 2000 epochs on 1024
samples.

## What success looks like

The task is linear, so the model should recover the signal almost exactly:

- `validation/loss` (MSE) approaches the noise floor, `noise_std**2 = 0.01`.
- `validation/explained_variance` climbs above 0.99.
- `test/*` at the end matches the validation metrics; a gap means the
  splits or the eval schedule are wired wrong.
