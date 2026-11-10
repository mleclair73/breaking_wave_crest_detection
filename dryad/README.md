# Dryad dataset

The rectified Argus video, crest masks, and per-collection metadata are
published on Dryad: <https://doi.org/10.5061/dryad.r2280gbsn>. The dataset
description lives there.

## Verify

Download the archive into `data/dryad/` so it contains `imagery/`, `crests/`,
and `metadata/`, then run from the repository root (requires `ffprobe` and
NumPy):

```bash
python3 dryad/verify_submission.py data/dryad
```

Expected summary:

```text
collections: 40
frames: 105676
duration: 15.343 hours
failures: 0
```

## Where the files go

| Dryad directory | Repository path | Used by |
|---|---|---|
| `imagery/*_RectifiedVideo.mkv` | `data/video_dataset/argus/*.avi` | production inference |
| `metadata/*_headerData.pickle` | `data/argus/metadata/` | per-video dropout diagnostics |
| `crests/*_DetectedCrests.mkv` | `model/predictions/argus/*_RectifiedVideo_pred.avi` | crest speeds (skips inference) |

Dryad has headers for the 40 published collections only. The context figure's
Argus coverage shading (`figures/00_context.py`) and the full-archive dropout
statistics use all 332 DUNEX headers, which are not published.

The pipeline reads `.avi`, so remux the `.mkv` files losslessly. The input rate
must be set explicitly (four Sep 19 collections are 1 Hz, the rest 2 Hz); a
plain `-c copy` produces a 1000 fps AVI.

```bash
mkdir -p data/video_dataset/argus data/argus/metadata model/predictions/argus
remux() {  # remux <in.mkv> <out.avi>
  r=$(ffprobe -v error -select_streams v -show_entries stream=r_frame_rate -of csv=p=0 "$1")
  ffmpeg -n -r "$r" -i "$1" -c copy "$2"
}
for f in data/dryad/imagery/*.mkv; do
  remux "$f" "data/video_dataset/argus/$(basename "${f%.mkv}").avi"
done
for f in data/dryad/crests/*.mkv; do
  b=$(basename "$f" _DetectedCrests.mkv)
  remux "$f" "model/predictions/argus/${b}_RectifiedVideo_pred.avi"
done
cp data/dryad/metadata/*.pickle data/argus/metadata/
```

Then run the crest-speed step without `--completed-only`; that flag needs the
`_density.png` files that only inference writes:

```bash
uv run python model/calculate_wave_speeds.py model/predictions/argus model/predictions/wave_speeds
```
