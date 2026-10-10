# PRISM

Code-only release of **PRISM: Precision-aware Reweighting and Internalization
for Skilled Manipulation**.

Project page: <https://seen-e.github.io/PRISM/>

This repository is self-contained at the source-code level. It vendors the
exact π0.5/OpenPI working tree used by the project and a PRISM-integrated
StarVLA tree. Users do **not** need to clone either upstream repository or
apply patches. Datasets, pretrained weights, experiment outputs, benchmarks,
and paper assets are intentionally excluded.

## Repository layout

- `openpi/`: bundled π0.5/OpenPI training, checkpointing, policy serving, and
  client code, including PPS, PSW, and CPR.
- `starvla/`: bundled StarVLA training and policy-serving code with PRISM
  integrated into QwenPI-v3/LayerwiseFM.
- `prism/`: framework-independent PPS and PSW utilities shared by the two
  implementations.
- `configs/`: public reference configurations.
- `scripts/`: installation, training, and serving entry points.
- `tests/`: numerical and data-contract tests.

## 1. Generate PPS targets

PPS is computed independently for each episode. The default implementation
expects the project dual-arm 22-D state layout: 14 joint values, two XYZ
end-effector positions, and two gripper values.

```bash
python -m pip install -e '.[data]'

prism-compute-pps \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir /path/to/pps_output
```

This writes both formats used by the bundled trainers:

- `pps_weights.npy`: flat global-frame order for π0.5/OpenPI;
- `<dataset_name>.npz`: identity-based `(trajectory_id, frame_index,
  pps_weight)` lookup for StarVLA.

For split state columns:

```bash
prism-compute-pps \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir /path/to/pps_output \
  --joint-key observation.joints \
  --eef-key observation.eef_xyz \
  --gripper-key observation.gripper
```

For another robot state layout, adapt the signal extraction in
`prism/pps.py` and validate it on complete episodes before training.

## 2. π0.5 + PRISM

Install the bundled implementation:

```bash
./scripts/install_pi05.sh
```

Configure the public `pi05_prism` recipe and train:

```bash
export PRISM_REPO_ID=org/your_lerobot_dataset
export PRISM_PPS_PATH=/absolute/path/to/pps_output/pps_weights.npy
export PRISM_LOSS_ACTION_DIM=14
export PRISM_BASE_CHECKPOINT=/path/to/pi05_base/params  # optional; defaults to the public OpenPI checkpoint
export PRISM_ASSETS_DIR=/path/to/norm_stats_assets
export PRISM_CHECKPOINT_DIR=/path/to/output_checkpoints
export EXP_NAME=prism_run

./scripts/train_pi05_prism.sh
```

The reference recipe uses action horizon 15, PSW `gamma=2`, `tau_0=1`,
`p=2`, SNR gating, CPR weight `0.1`, and 15,000 training steps. Optional
environment overrides are documented in `configs/pi05_prism.env.example`.

The public recipe uses OpenPI's ALOHA LeRobot schema. If a dataset has a
different observation/action schema, select or adapt the corresponding
`DataConfigFactory` in `openpi/src/openpi/training/config.py`; the bundled data
loader supports `pps_weight_path` for every LeRobot factory.

Serve a trained checkpoint:

```bash
export CHECKPOINT_DIR=/path/to/output_checkpoints/pi05_prism/prism_run/15000
./scripts/serve_pi05_prism.sh
```

The normal OpenPI websocket client is included under
`openpi/packages/openpi-client`.

## 3. StarVLA + PRISM

Create the StarVLA environment and install the bundled implementation:

```bash
./scripts/install_starvla.sh
```

Train QwenPI-v3/LayerwiseFM with PRISM:

```bash
export BASE_VLM=/path/to/Qwen3-VL-4B-Instruct
export DATA_ROOT=/path/to/starvla_lerobot_root
export DATA_MIX=your_registered_mix
export PPS_ROOT=/path/to/pps_output
export PRETRAINED_CHECKPOINT=/path/to/task_shaped_starvla_checkpoint.pt
export RUN_ID=prism_qwenpi_v3
export NUM_PROCESSES=8

./scripts/train_starvla_prism.sh
```

For the expected sub-1.0 initial loss, `PRETRAINED_CHECKPOINT` must already
use the dataset's configured action/state dimensions. For example, an 8-D
RLBench PRISM run should warm-start from an 8-D StarVLA RLBench checkpoint.
When a public checkpoint has fewer dimensions (for example, Bridge 7-D to
RLBench 8-D), the bundled loader can preserve the overlapping pretrained
rows/columns when `trainer.pretrained_allow_shape_mismatch: true`; the newly
added dimension still needs warm-up and can make the very first loss higher.

`DATA_MIX` and the robot modality definition must be registered in the bundled
StarVLA configuration exactly as required by its standard loader. Every
dataset directory in the mixture needs a matching `<dataset_name>.npz` under
`PPS_ROOT`.

Serve the trained checkpoint:

```bash
export CKPT_PATH=/path/to/steps_15000_pytorch_model.pt
./scripts/serve_starvla_prism.sh
```

PPS labels are training-only. Inference inputs, normalization, camera order,
and client/server protocols remain the native π0.5 or StarVLA contracts.

## Tests

The checks already completed for this release are recorded in
[`VALIDATION.md`](VALIDATION.md).

The source-only checks are:

```bash
python -m pytest -q
bash -n scripts/*.sh
python -m compileall -q prism openpi/src starvla/starVLA starvla/deployment
```

Full training requires compatible GPUs, the stated model initialization, the
dataset, its normalization statistics, and frame-aligned PPS targets.

## Source and licenses

See `SOURCE_VERSIONS.md` for the bundled source snapshots. The π0.5/OpenPI
files retain their Apache-2.0 and Gemma license notices. StarVLA retains its
MIT license. Add the intended license for the original PRISM modules before
public release.
