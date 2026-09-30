# Date-disjoint OOD test set

This is a fixed, human-reviewed 20-image OOD evaluation set. It contains 20
500x512 RGB timestacks from nine source videos on eight dates. Those dates do
not occur in the combined training, validation, or regular test splits.

Use this set only after model and decision-threshold selection. Do not move its
images or labels into training/validation or use OOD results to tune a model.

## Contents

- `images/`: the 20 model-oriented PNG inputs.
- `manifest.csv`: exact source video, spatial row, frame interval, conditions,
  missing-data metrics, orientation transform, and source selection rank.
- `source_candidates.csv`: the exact nine-source, 20-crop input plan.
- `verification.json`: leakage, geometry, QA-policy, and source-AVI hashes.
- `contact_sheet.png`: visual orientation and content audit.
- `build_ood_test.py`: deterministic regeneration code.
- `cvat_job_22_export_20260918.zip`: preserved reviewed CVAT for images 1.1
  export.
- `annotations_human_canonical.xml`: manifest-ordered reviewed annotations,
  tagged `done` and `annotated` in model/CVAT orientation.
- `masks/` and `image_mask_mapping.csv`: one-pixel binary masks and the
  evaluation-loader mapping (`split=ood_test`).
- `annotation_status.json`: annotation provenance, hashes, counts, and per-image
  line totals.
- `annotation_overlay.png`: visual audit of the reviewed masks over all images.
- `import_reviewed_annotations.py`: validates the preserved export and
  deterministically regenerates canonical XML, masks, mapping, and status.
- `annotations_preseed.xml`, `ood_test_preannotations.zip`,
  `preannotation_overlay.png`, and `preannotation_summary.json`: original model
  suggestions retained for provenance; these are not ground truth.

All images use the same 500x512 geometry as the main dataset. The builder first
looks for a synchronized window that passes the frozen v6 gates. Where that is
impossible, it preserves the planned spatial rows and full width and chooses the
least-missing window. Rank 1 is explicitly fixed to middle-twilight frames
1280--1791 because its metadata-complete interval occurs after the sunset scene
has become nearly black. This retains limited missing columns instead of either
the heavily gapped early interval or unusable dark imagery. `selection_mode`
and all missingness values are explicit in `manifest.csv`. Rank 9 uses the
reviewed D4 option at yFRF 550 and frames 1884--2395 for better late-interval
contrast while retaining a strict quality pass.

## Regenerate images

From this directory (`model/data/ood_test`):

```bash
MPLCONFIGDIR=/tmp/mpl-ood-test python build_ood_test.py \
  --video-dir ../../../data/video_dataset/argus \
  --candidate-csv source_candidates.csv \
  --current-conditions ../full_split/figures/acquisition_conditions.csv \
  --output-dir .
```

The nine selected source AVIs must be present in `data/argus`. Their expected SHA-256
digests are recorded in `verification.json`. Rebuilding images leaves reviewed
label artifacts in place, but rewrites this README and the image-source
verification file.

## Regenerate reviewed labels

After the images exist, run:

```bash
python import_reviewed_annotations.py \
  --archive cvat_job_22_export_20260918.zip \
  --dataset .
```

This validates all 20 image identities, 512x500 geometry, `breaker` polylines,
and in-bounds vertices before writing labels. It preserves the single vertical
flip already applied during image extraction; annotation import applies no
additional orientation transform.
