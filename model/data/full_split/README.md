# wave-crest combined dataset — `full_split`

Canonical dataset root for `WaveBreakingDataset`: 256 images, masks, canonical
labels, splits, and coverage figures. The one-time build/import/preseed pipeline
and its raw `sources/` bundle have been removed from the working tree; they are
archived in the pre-cleanup snapshot commit and remain recoverable from git
history. Everything needed to *use* the dataset and regenerate its figures lives
here.

## Contents

- `images/`, `masks/` — 256 canonical RGB crops and matching one-pixel
  human-centreline masks (512 × 500, one-to-one). Filenames are descriptive:
  `<source_id>_y<yFRF>_f<start_frame>.png` (e.g.
  `ArgusFF_20211007T210100Z_y0600_f1624.png`), with `mask_` prefixed for masks —
  `(source video, yFRF, start frame)` is a unique key. Masks are always
  `mask_<image_name>`.
- `image_mask_mapping.csv` — the loader's mapping, carrying the frozen `split`
  column (`train` / `val` / `test`). Split selection is explicit from this file;
  never inferred from filenames.
- `split_manifest.csv` — self-contained per-crop provenance for all 256 rows:
  `sample_id` (historical id), `image_name` (current file), `source_video`
  (bare source id, matching `split_assignment.csv`), acquisition, pixel/time
  bounds, `selection_seed`, `selection_stratum`, `checkpoint_sha256`, and the
  joined wave/wind/solar conditions.
- `split_assignment.csv` — acquisition-level split assignment (one source video
  per split).
- `verification.json` — deterministic whole-dataset SHA-256 inventory, pairing,
  geometry, leakage, and 224×224 loader checks. Regenerate it with
  `python data/full_split/verify_dataset.py` from `model/`.
- `annotations_canonical.xml` — canonical human CVAT geometry in the model's
  vertical orientation.
- `figures/` — train/val/test contact sheets and acquisition, wave, lighting,
  meteorological, and proposed-OOD coverage figures, plus the joined
  `acquisition_conditions.csv` (output) and the two figure inputs
  `conditions_input.csv` and `ood_candidates.csv`.
- `build_split_figures.py` — regenerates all figures and the joined condition
  table from the files above.
- `ANNOTATION_WORKFLOW.md` — historical record of the extraction, preannotation,
  CVAT-review, and rebuild procedure for adding images. Its commands reference
  build/import scripts retained only in git history.

## Composition

- 256 crops.
- Combined split: **172 train / 43 validation / 41 test**
  (67.2% / 16.8% / 16.0%).
- **19 distinct source videos: 11 train, 4 validation, 4 test — zero
  cross-split overlap** (every video belongs to exactly one split). The 20
  reviewed evaluation-extension images remain held out across validation and
  test.
- The `ArgusFF_20210919T163000Z` recording (14 train crops) was removed because
  it is present verbatim in the Stage-1 inference pool `data/video_dataset/argus/`,
  and replaced by 14 crops from `ArgusFF_20211026T160100Z` (a new day, absent
  from that pool). **No training recording appears in the inference pool.**
  (Val/test still contain recordings that overlap the pool by design; they are
  held out from training, not from downstream inference.)
- The validation split is for model selection; the in-domain test split is
  **evaluation-only** and must not be used for threshold selection or tuning.
- Images and masks share 512 × 500 geometry in the model's vertical
  orientation. Do not apply another data-level flip.

## Proposed future OOD holdout

`figures/ood_candidates.csv` reserves acquisition- and date-disjoint Argus
sources for a replacement OOD holdout, and drives `ood_candidate_conditions.png`
(existing joint-condition cloud vs. date-disjoint candidates, with pairwise
convex hulls). The candidates are joint-condition shifts outside the existing
multivariate cloud; edge rows bracket the usable high/low values for wave
height, peak frequency, direction, and solar elevation. Any replacement must
stay outside every training, validation, and test loader and be evaluated only
after model selection.

## Freeze

- Dataset version: **`full_split-2026-09-16-revised`**. The authoritative
  content digest is `dataset_sha256` in `verification.json`; it covers every
  mapped image and mask plus the canonical annotations, mapping, split
  assignment, and split manifest.

- Promoted from source commit `1de72f7`; the 230-image base was frozen under
  the annotated tag `wave-crest-v6-initial-submission`, which predates the
  current model-oriented dataset.
- Only the dataset **spec** is tracked in git: this README, the mapping,
  `split_assignment.csv`, `split_manifest.csv`, and `annotations_canonical.xml`.
  Pixel data (`images/`, `masks/`) and `figures/` are gitignored by repo policy
  and carried outside git (Dropbox / RunPod sync). Masks regenerate from
  `annotations_canonical.xml`; image crops are located by `split_manifest.csv`.
- Augmentation is applied by `dataset.WaveBreakingDataset`; this dataset
  does not carry a separate augmentation copy.

## Regenerating figures

Run from this directory:

```bash
python build_split_figures.py
```

It reads `image_mask_mapping.csv`, `split_manifest.csv`,
`figures/conditions_input.csv`, and `figures/ood_candidates.csv`, and rewrites
the three contact sheets, split-coverage, wave-condition, lighting/meteorological,
and proposed-OOD figures plus `figures/acquisition_conditions.csv`. It verifies
the 172/43/41 counts and source-video-disjoint splits, and has no dependency on
raw Argus videos or repository-level NetCDF files.

## Training

- Train: `python segmentation/train_segnext.py --config segmentation/configs/segnext_full_split.yaml`
- In-domain held-out test: `python segmentation/evaluate_test.py --checkpoint <run>/best_model.pth`
  (selects the decision threshold on `val`, applies it to `test`; `--threshold` to force one).
