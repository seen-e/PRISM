# Validation status

The release tree was validated on 2026-10-09. Source-level tests use no
project datasets or private checkpoints; the full smoke checks below used one
local RLBench sample and the project's existing initialization checkpoints.

## Completed checks

- Installed the bundled pi0.5/OpenPI environment from its checked-in
  `uv.lock` using the public PyPI index.
- Ran the release test suite: **9 passed**.
- Ran selected upstream OpenPI model and data-loader tests: **11 passed,
  1 deselected** (the deselected test requires downloading a real dataset).
- Loaded the public `pi05_prism` recipe and constructed its model state,
  including the CPR precision head.
- Loaded all pi0.5 training, normalization-statistics, and serving entry
  points.
- Checked the pi0.5 PPS adapter for shape, range, dtype, and non-mutating
  sample behavior.
- Loaded the public StarVLA recipe and its training and serving entry points.
- Ran a small StarVLA LayerwiseFM forward/backward test with PRISM enabled;
  per-sample loss, clean-time output, and gradients were finite.
- Checked all public shell entry points with `bash -n`.
- Confirmed an existing OpenPI pi0.5 + PRISM training run starts with finite
  step-0 loss **0.7769**.
- Ran StarVLA QwenPI-v3 + PRISM on a real normalized RLBench sample using an
  action-dimension-matched StarVLA checkpoint: finite initial total loss
  **0.9440** (flow/PSW **0.9419**, CPR **0.0204**).
- Ran checkpoint inference on the same StarVLA model and checked that the
  returned `(1, 15, 8)` normalized action chunk is finite.

## Environment-dependent checks still required

A new StarVLA optimizer step was not run because all eight GPUs were occupied
by the user's production job. The completed real forward-loss check exercises
the training data/model/PRISM path, and the checkpoint inference check
exercises the denoising path. Before a public release on another platform,
repeat a short optimizer smoke test in that platform's CUDA environment.
