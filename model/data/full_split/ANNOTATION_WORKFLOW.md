# Annotation workflow

How `full_split` images are extracted and annotated. `../ood_test/build_ood_test.py` and
`../ood_test/import_reviewed_annotations.py` implement steps 1 and 4; preannotation and
rasterization are in `../utils/`.

## 1. Extract

Each crop comes from an acquisition assigned to a single split, and has one manifest
row (source video, timestamp, yFRF, pixel row, frame interval, conditions, rationale,
source hash). No `(source_video, yfrf, t_start, t_end)` crop is duplicated.

- source frame: 1600 × 500 (cross-shore × time)
- crop: 512 consecutive frames at pixel row `1500 - yfrf`, transposed to
  `(cross_shore, time, RGB)`, then flipped once (`flipud`)
- output: 500 × 512 (height × width); no further flip or resize

Quality gates:

- top missing-data margin ≤ 78 px
- ≤ 2,048 zero pixels outside that margin
- no all-zero time columns
- saturated-pixel fraction ≤ 0.25

## 2. Preannotate (optional)

Model suggestions to speed up review; not ground truth. From `model/`:

```bash
uv run python data/utils/preannotate.py <images_dir> preannotations.xml
```

- tiled inference with the released `segnext_t_learned_up_skip_4_2`: 224 px patches,
  86 px overlap, threshold 0.48
- `cvat_io.mask_to_polylines`: drop components < 20 px, skeletonize, trace paths,
  RDP simplify (ε = 1.0)

## 3. Review in CVAT

1. Image task with one polyline label, `breaker`.
2. Upload images and the preannotations (CVAT for images 1.1).
3. Correct predicted lines, add missed crest centrelines, remove false positives.
   Empty images are allowed. Keep orientation and geometry unchanged.
4. Export as CVAT for images 1.1.

## 4. Import and add to the dataset

The export is validated (image identity, 512 × 500 geometry, `breaker` labels,
in-bounds vertices) and ordered to match the manifest. Reviewed polylines are
rasterized to one-pixel masks with `cvat_io.rasterize_polylines` (`rasterize_masks.py`
checks the stored masks against the XML), and `image_mask_mapping.csv`, `split_manifest.csv`,
figures (`build_split_figures.py`), and `verification.json` (`verify_dataset.py`)
are regenerated.
