# Argus rectified videos (DUNEX, FRF Duck NC)

Stage-1 input: 40 rectified Argus videos,
`ArgusFF_YYYYMMDDThhmmssZ_RectifiedVideo.avi`, Sep–Oct 2021.

## Source

Dryad, DOI [10.5061/dryad.r2280gbsn](https://doi.org/10.5061/dryad.r2280gbsn).
The release contains:

```text
imagery/   ArgusFF_<timestamp>_RectifiedVideo.mkv   input to this pipeline
crests/    ArgusFF_<timestamp>_DetectedCrests.mkv   published model crest masks
metadata/  ArgusFF_<timestamp>_headerData.pickle    camera capture metadata
```

Inference reads `.avi`. Remux the Dryad imagery into this folder losslessly
(the frames are unchanged; the input rate must be set explicitly):

```bash
for f in <dryad>/imagery/*.mkv; do
  r=$(ffprobe -v error -select_streams v -show_entries stream=r_frame_rate -of csv=p=0 "$f")
  ffmpeg -r "$r" -i "$f" -c copy "data/video_dataset/argus/$(basename "${f%.mkv}").avi"
done
```

Four Sep 19 collections are 1 Hz; all others are 2 Hz. Crest-speed extraction
takes the frame rate from the acquisition date.
