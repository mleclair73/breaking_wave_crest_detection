# full_split

Canonical train/validation/test dataset for `WaveBreakingDataset`: 256 RGB
timestack crops with human-annotated one-pixel crest-centreline masks.

## Data

- `images/`, `masks/` — 256 crops and matching masks, 512 × 500. Images are
  named `<source_id>_y<yFRF>_f<start_frame>.png`; masks are `mask_<image_name>`.
- `annotations_canonical.xml` — CVAT annotations the masks were rasterized from.
- `image_mask_mapping.csv` — loader mapping with the frozen `split` column.
  Splits come from this file, never from filenames.
- `split_manifest.csv` — per-crop provenance and joined wave/wind/solar conditions.
- `split_assignment.csv` — source video to split.
- `verification.json` — SHA-256 inventory and dataset digest.

## Scripts

- `verify_dataset.py` — pairing, geometry, split leakage, hashes, loader output.
- `check_train_not_in_inference_pool.py` — no training video in `data/video_dataset/argus/`.
- `rasterize_masks.py` — rebuilds masks from the XML (`../utils/cvat_io.py`)
  and checks them pixel-for-pixel against `masks/`.
- `build_split_figures.py` — contact sheets and condition-coverage figures in `figures/`.

`ANNOTATION_WORKFLOW.md` describes crop extraction, preannotation, and CVAT review.

## Composition

- 172 train / 43 val / 41 test.
- 19 source videos: 11 train, 4 val, 4 test. Each video is in exactly one split.
- No training video is in the inference pool. Val and test contain pool
  videos: they are held out from training, not from inference.
- `val` is for model selection and threshold calibration. `test` is for final
  evaluation only.
- Images and masks are already in the model's vertical orientation; do not
  flip them again.
- Augmentation is applied by `dataset.WaveBreakingDataset`; no augmented copy
  is stored.

## Version

`full_split-2026-09-16`. The content digest is `dataset_sha256` in
`verification.json`; it covers images, masks, annotations, mapping, split
assignment, and manifest.

## Reproduce

From `model/`:

```bash
uv run python data/full_split/verify_dataset.py      # must report the digest in verification.json
uv run python data/full_split/rasterize_masks.py     # masks match annotations_canonical.xml
uv run python data/full_split/check_train_not_in_inference_pool.py  # needs data/video_dataset/argus/
cd data/full_split && uv run python build_split_figures.py
```
