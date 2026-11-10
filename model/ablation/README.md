# Model ablation

This package contains the configurations and post-training analysis for the
single-seed architecture and augmentation study. Reusable architectures,
datasets, augmentations, losses, metrics, tiled inference, checkpoint loading, and training
live one level above in `model/`.

All runs use seed 42, the augmentations in `model/augmentations.py`,
foreground-IoU checkpoint selection, patience 12, and the common recipe in
`configs/common.yaml`.

The architecture table is ordered by `manifest.yaml`:

1. U-Net
2. Attention U-Net
3. DeepLabV3+
4. Swin U-Net
5. SegNeXt-T vanilla HAM
6. SegNeXt-T learned upsampling
7. SegNeXt-T learned upsampling + H/4 skip
8. SegNeXt-T learned upsampling + H/4 and H/2 skips

A ninth run disables training augmentation for the augmentation and
missing-data controls. SegNeXt-T with learned upsampling and H/4 + H/2 skips is
the selected model and the operating-point and holdout-inference target.

## Run

From `model/`:

```bash
bash ablation/run_all.sh
```

Individual stages:

```bash
python -m train --all --device cuda
python -m ablation.infer --device cuda --batch-size 32
python -m ablation.evaluate --workers 8
python -m ablation.operating_points --workers 8
python -m ablation.costs
python -m ablation.report
python -m ablation.verify_outputs
```

The study's runs, caches, metrics, tables, and released model are stored under
`model/outputs/model_ablation_final/final_aug_v1/`.

## Evaluation policy

- Complete-image inference uses 224-pixel tiles, 86-pixel overlap, reflection
  padding, and Hanning blending.
- Each model's threshold is calibrated on clean validation images and frozen
  for missing-data, date-disjoint diagnostic, and holdout evaluation.
- Pixel metrics are averaged across complete images.
- Crest metrics use optimized one-to-one crest matching and report both
  quality-weighted and conventional detection results.
- Operating-point curves target the selected H/4 + H/2 skip model.
- The source-disjoint 41-image holdout split is evaluated once at the frozen
  validation threshold and is never used for calibration.

Holdout commands:

```bash
python -m ablation.infer --device cuda --batch-size 32 \
  --manifest ablation/manifests/holdout_test.yaml --condition test
python -m ablation.evaluate --workers 8 \
  --manifest ablation/manifests/holdout_test.yaml --condition test \
  --thresholds-from outputs/model_ablation_final/final_aug_v1/released/segnext_t_learned_up_skip_4_2/thresholds.json
```

## Crest metrics

Crests are matched one-to-one and scored two ways: quality-weighted (skeleton
overlap at 2 px tolerance) and binary detection (50% mutual coverage at 2 px
dilation). Definitions are in `model/metrics/crest.py`.
