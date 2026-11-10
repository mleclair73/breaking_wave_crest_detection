# Date-disjoint OOD test set

This is a fixed, human-reviewed 20-image OOD evaluation set. It contains 20
500x512 RGB timestacks from nine source videos on eight dates. Those dates do
not occur in the combined training, validation, or regular test splits, although
two are included in the evaluation set.

This dataset was only used after model and decision-threshold selection.

## Contents

- `images`: timestacks
- `masks/` and `image_mask_mapping.csv`: one-pixel binary masks and the
  evaluation-loader mapping (`split=ood_test`).
- `manifest.csv`: exact source video, spatial row, frame interval, conditions,
  missing-data metrics, orientation transform, and source selection rank.
- `build_ood_test.py`: code to build the test set.
- `import_reviewed_annotations.py`: validates the preserved export and
  deterministically regenerates canonical XML, masks, mapping, and status.
- `contact_sheet.png`: visual orientation and content audit.
- `annotation_overlay.png`: visual audit of the reviewed masks over all images.

All images use the same 500x512 geometry as the main dataset.
