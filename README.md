# Breaking-wave crest detection

This repository trains semantic-segmentation models for breaking-wave crests
in rectified Argus video, runs production inference, extracts crest speeds, and
merges the results with environmental observations from the U.S. Army Corps of
Engineers Field Research Facility in Duck, North Carolina. The associated video
dataset is available from [Dryad](https://doi.org/10.5061/dryad.r2280gbsn).

## Production model

The production model is SegNeXt-T with learned upsampling and H/4 and H/2
decoder skips. Its configuration is
[`model/ablation/configs/segnext_t_learned_up_skip_4_2.yaml`](model/ablation/configs/segnext_t_learned_up_skip_4_2.yaml),
and its validation-selected decision threshold is `0.48`.

The released checkpoint bundle belongs at:

```text
model/outputs/promoted/segnext_t_learned_up_skip_4_2/
├── best_model.pth
├── complete.json
├── config.yaml
├── provenance.json
├── SHA256SUMS
└── thresholds.json
```

Model artifacts are excluded from Git history.

## Code layout

Reusable model code lives directly under `model/`:

```text
model/models/       model architectures
model/metrics/      pixel and crest metrics
model/augmentations.py training augmentations
model/checkpoint.py strict checkpoint loading
model/dataset.py    datasets and augmentations
model/losses.py     segmentation losses
model/tiling.py     tiled inference
model/train.py      training entry point
```

`model/ablation/` contains only study configuration and post-training
orchestration: manifests, evaluation, operating-point analysis, costs, and
report generation.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. From the
repository root:

```bash
uv sync --locked
```

The remaining commands use `uv run` and therefore execute in that locked
environment.

## External assets

| Asset | Expected location | Purpose |
|---|---|---|
| Production checkpoint bundle | `model/outputs/promoted/segnext_t_learned_up_skip_4_2/` | Video inference |
| SegNeXt-T encoder weights | `model/pretrained_weights/segnext_tiny_512x512_ade_160k.pth` | Training initialization |
| Canonical labeled dataset | `model/data/full_split/images/` and `model/data/full_split/masks/` | Training, validation, and holdout evaluation |
| Date-disjoint evaluation set | `model/data/ood_test/images/` and `model/data/ood_test/masks/` | Post-selection evaluation |
| Primary and supplemental videos | `data/video_dataset/argus/` and `data/video_dataset/extra/` | Production inference |
| Environmental observations | `data/bathy/`, `data/water_level/`, `data/8m_array_waves/`, and `data/met/` | Physical-data merge and analysis |

The canonical segmentation dataset contains 256 image/mask pairs under
`model/data/full_split/`; the date-disjoint set contains 20 pairs under
`model/data/ood_test/`. Dataset mappings and split assignments are versioned
beside the external PNG files.

The primary and supplemental Argus videos belong at:

```text
data/video_dataset/argus/
data/video_dataset/extra/
```

See [`data/README.md`](data/README.md) and
[`model/data/README.md`](model/data/README.md) for the complete inventories.
Set `DUNEX_DATA_ROOT` and `DUNEX_OUTPUTS_DIR` when data or model outputs live
outside the repository defaults.

## Verify the installation and inputs

```bash
uv run python -m ablation.verify
uv run python -m unittest discover -s model/tests -p 'test_*.py'
```

The first command validates model configurations, dataset layout, pretrained
weights, and package boundaries. The second runs the lightweight pipeline
contracts.

To validate the complete canonical dataset, including image/mask geometry and
split isolation:

```bash
uv run python model/data/full_split/verify_dataset.py
```

## Reproduce the model study

Run training, inference, evaluation, operating-point analysis, cost analysis,
and report generation on a CUDA system:

```bash
DEVICE=cuda INFERENCE_BATCH_SIZE=32 METRIC_WORKERS=8 \
  uv run bash model/ablation/run_all.sh
```

Run or resume the stages separately:

```bash
uv run python -m train --all --device cuda
uv run python -m ablation.infer --device cuda --batch-size 32
uv run python -m ablation.evaluate --workers 8
uv run python -m ablation.operating_points --workers 8
uv run python -m ablation.costs
uv run python -m ablation.report
uv run python -m ablation.verify_outputs
```

Train one configuration with:

```bash
uv run python -m train \
  --config ablation/configs/segnext_t_learned_up_skip_4_2.yaml \
  --device cuda
```

Add `--smoke` for one epoch with one patch per source. Runs, cached inference,
metrics, tables, and reports are written below `model/outputs/`.

The evaluation policy uses 224-pixel tiles with 86-pixel overlap, reflection
padding, and Hanning blending. Thresholds are selected on clean validation
images and frozen for missing-data, date-disjoint, and holdout evaluation. The
source-disjoint 41-image holdout split is evaluated only for the selected
model.

## Run production inference

Validate configuration and strict checkpoint compatibility without processing
video:

```bash
uv run python model/segmentation/predict_video.py \
  model/segmentation/configs/predict_argus.yaml --validate-only
```

Process the primary and supplemental videos:

```bash
uv run python model/segmentation/predict_video.py \
  model/segmentation/configs/predict_argus.yaml

uv run python model/segmentation/predict_video.py \
  model/segmentation/configs/predict_extra.yaml
```

The default `device: auto` selects CUDA, then MPS, then CPU. `--device`,
`--model-path`, `--input-path`, and `--output-path` provide command-line
overrides without editing the YAML files. Each input video produces a binary
prediction AVI, an overlay video, and a detection-density image.

## Extract crest speeds

```bash
uv run python model/calculate_wave_speeds.py \
  model/predictions/argus model/predictions/wave_speeds --completed-only

uv run python model/calculate_wave_speeds.py \
  model/predictions/extra model/predictions/wave_speeds --completed-only
```

Each prediction video produces a speed NetCDF, a pixel-speed map, and a
trajectory CSV. Complete output triplets are skipped on reruns.

## Build the merged physical dataset

```bash
uv run python create_merged_dataset.py \
  --speeds-dir model/predictions/wave_speeds \
  --out combined_dunex_dataset_lerp.nc
```

Linear interpolation is the default. Pass `--no-lerp` for nearest-neighbor
matching and choose a different output filename.

## Limitations

The model comparison uses one training seed (`42`). GPU retraining is seeded
but is not guaranteed to be bitwise identical across hardware, drivers, CUDA
libraries, or PyTorch versions.
