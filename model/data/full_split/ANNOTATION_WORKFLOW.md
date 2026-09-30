# Annotation workflow

This is the standard process for adding one or more human-reviewed breaker
annotations to `full_split`. The same contract applies to training additions
and held-out extensions; only the destination split and provenance differ.

## 1. Select and extract the image

Use an acquisition assigned exclusively to the intended split. Record one
manifest row with the source video, acquisition timestamp, yFRF, pixel row,
512-frame interval, environmental conditions, selection rationale, and source
hash.

The frozen image contract is:

- source video frame geometry: 1600 × 500 (cross-shore × time)
- output image geometry: 500 × 512 (height × width)
- pixel row: `1500 - yfrf`
- crop: 512 consecutive frames, then transpose to `(cross_shore, time, RGB)`
- orientation: one vertical flip (`flipud`) into model/CVAT orientation
- no additional flip or resize during annotation or inference

Each crop must pass the existing quality gates:

- top missing-data margin ≤ 78 pixels
- zero pixels outside that margin ≤ 2,048
- no fully zero time columns
- saturated-pixel fraction ≤ 0.25

Do not duplicate an existing `(source_video, yfrf, t_start, t_end)` crop.

Current extraction implementation and six-image example:

- [build_train_extension.py](build_train_extension.py)
- [train_extension_v1/manifest.csv](sources/train_extension_v1/manifest.csv)
- [train_extension_v1/README.md](sources/train_extension_v1/README.md)

## 2. Generate optional model preannotations

Run the pinned SegNeXt checkpoint against the stored model-oriented PNG:

```bash
MPLCONFIGDIR=/tmp/mpl-train-extension \
python preseed_train_extension.py
```

Current preseed parameters:

- checkpoint: `outputs/segnext_base_skip_1_2/run_20260903_042920/best_model.pth`
- checkpoint SHA-256: `e782ccaec727f21df07af00136ea27d75fc7913c5a97e0d3c6c47c5acdb76fe0`
- tiled inference: 224-pixel patches, 32-pixel overlap
- foreground threshold: 0.50
- remove connected components smaller than 20 pixels
- skeletonize, trace graph paths, simplify with RDP epsilon 1.0

Preannotations are suggestions only. Human review must add missed breakers and
remove false positives.

Key outputs:

- `annotations_preseed.xml` — CVAT geometry with `source="auto"`
- `train_extension_v1_preannotations.zip` — uploadable CVAT 1.1 annotation archive
- `preannotation_overlay.png` — visual audit
- `preannotation_summary.json` — checkpoint, parameters, and per-image counts

## 3. Human review in CVAT

1. Create an image task with one polyline label named `breaker`.
2. Upload the image ZIP.
3. Upload the preannotation ZIP as **CVAT for images 1.1** annotations.
4. Preserve image orientation and geometry.
5. Correct every predicted line; add missed breaker centrelines; remove false
   positives; allow a legitimately empty image.
6. Export **CVAT for images 1.1**.

## 4. Normalize the reviewed export

Use the shared importer, setting the expected image count for the batch:

```bash
python import_extension_cvat.py \
  --archive cvat_export.zip \
  --manifest sources/train_extension_v1/manifest.csv \
  --expected-count 6 \
  --output sources/train_extension_v1/annotations_human_canonical.xml \
  --status-output sources/train_extension_v1/annotation_status.json
```

The importer validates image identity, 512 × 500 geometry, supported CVAT
shapes, `breaker` labels, and in-bounds vertices. It orders images to match the
manifest and adds `done` and `annotated` tags.

## 5. Add to the canonical dataset

Only after review/import:

- append the reviewed images and manifest rows to `full_split`
- append the canonical CVAT images to `annotations_canonical.xml`
- regenerate one-pixel masks from the reviewed polylines
- update `image_mask_mapping.csv`, `split_manifest.csv`, and split counts
- regenerate figures and verification metadata

Run the deterministic checks:

```bash
python build_combined_dataset.py
python build_split_figures.py
```

Acceptance requires one image/mask per manifest row, matching 512 × 500
geometry, source-video-disjoint splits, no unexpected files, valid annotation
tags, and a passing `verification.json`.

## File/status convention

Use the following lifecycle and keep intermediate artifacts separate from the
active dataset:

`pending extraction` → `pending_human_annotation` → `predicted_labels_pending_human_review` → `reviewed/canonical` → `active dataset`

The reviewed training-extension batch is under
[`sources/train_extension_v1/`](sources/train_extension_v1/). Once its reviewed
export is imported and the canonical rebuild succeeds, it is represented in
the six appended training rows of the active dataset.
