# Model ablation

This package contains the configurations and post-training analysis for the
single-seed architecture and augmentation study. Reusable architectures,
datasets, augmentations, losses, metrics, tiled inference, checkpoint loading, and training
live one level above in `model/`.

All runs use seed 42, the augmentation implementation in `model/dataset.py`,
foreground-IoU checkpoint selection, patience 12, and the common recipe in
`configs/common.yaml`. Configuration validation rejects unknown, retired, and
unused fields. Training provenance records source, dataset mapping, pretrained
weights, dependencies, and initial-model fingerprints.

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
python -m ablation.verify
python -m unittest discover -s tests -p 'test_*.py'
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

Generated runs, caches, metrics, tables, and the promoted model are stored
under `model/outputs/`.

## Evaluation policy

- Complete-image inference uses 224-pixel tiles, 86-pixel overlap, reflection
  padding, and Hanning blending.
- Each model's threshold is calibrated on clean validation images and frozen
  for missing-data, date-disjoint diagnostic, and holdout evaluation.
- Pixel metrics are averaged across complete images.
- Crest metrics use optimized one-to-one, quality-weighted crest matching.
- Operating-point curves target the selected H/4 + H/2 skip model.
- The source-disjoint 41-image holdout split is evaluated once at the frozen
  validation threshold and is never used for calibration.

Holdout commands:

```bash
python -m ablation.infer --device cuda --batch-size 32 \
  --manifest ablation/manifests/holdout_test.yaml --condition test
python -m ablation.evaluate --workers 8 \
  --manifest ablation/manifests/holdout_test.yaml --condition test \
  --thresholds-from outputs/promoted/segnext_t_learned_up_skip_4_2/thresholds.json
```
