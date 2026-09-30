# Argus Coastal Imaging Header Metadata (DUNEX FRF)

---
This folder contains the collection-header metadata for the Argus optical imaging system operated at the U.S. Army Corps of Engineers Field Research Facility (FRF) in Duck, North Carolina, during the DUNEX (During Nearshore Events Experiment) field campaign. The Argus tower images the nearshore surf zone with a fixed array of up to six cameras (`c1`–`c6`). Each metadata file describes one ~32-minute imaging "collection," recording exactly which frames each camera captured, the precise capture timestamps, and which portions of the collection produced usable, fully-synchronized imagery. These header files do **not** contain the imagery itself; they are the per-collection bookkeeping needed to align, quality-control, and time-reference the raw Argus frames against the in-situ microSWIFT drifter and CDIP wave records. The collections span September 17 – October 29, 2021, covering the DUNEX deployment window. A companion `video_metadata.yaml` provides hand-scored quality assessments for a subset of collections used in analysis.

## Description of the data and file structure

Each `.pickle` file corresponds to a single Argus collection and is named by its UTC start time:

```
ArgusFF_<YYYYMMDD>T<HHMMSS>Z_headerData.pickle
```

A collection is a fixed ~32-minute window during which the camera array is triggered at 2 Hz, producing **3840 frame slots** per camera. The files are standard Python `pickle` objects holding a single dictionary. There are **332** header files spanning **2021-09-17 06:30 UTC** to **2021-10-29 16:03 UTC**.

Each pickle dictionary is organized as follows:

```
├── ArgusFF_<timestamp>_headerData.pickle  (dict)
    ├── cams_all (list)
    |    Names of the cameras present in this collection,
    |    a subset of ['c1','c2','c3','c4','c5','c6'].
    |    Most collections use all six; some use fewer
    |    when individual cameras were offline.
    ├── time (ndarray of datetime, length 3840)
    |    UTC timestamp of each frame slot. Spacing is
    |    0.5 s (2 Hz sampling) across the ~32-minute
    |    collection.
    ├── captureArray (ndarray, shape (3840, n_cams))
    |    Per-frame, per-camera capture flag. 1 if that
    |    camera recorded a frame in that slot, 0 if the
    |    frame is missing. Columns correspond to cams_all.
    ├── totalCaptured (ndarray, length 3840)
    |    Number of cameras that successfully captured each
    |    frame slot (row sum of captureArray). A value
    |    equal to len(cams_all) means all cameras were
    |    synchronized for that frame.
    ├── captures (list)
    |    Per-frame list of raw camera capture timestamps
    |    (Unix epoch seconds) for frames where imagery was
    |    recorded.
    ├── missingFrames (tuple of ndarray)
    |    Indices into the frame axis where one or more
    |    cameras dropped a frame. Used to mask gaps before
    |    rectification.
    ├── startSkipping / endSkipping (list, per camera)
    |    First and last frame index to skip at the head and
    |    tail of the collection for each camera (warm-up and
    |    shutdown frames that should be discarded).
    ├── beginGoodTimes (datetime)
    |    Start of the longest contiguous interval in which
    |    all cameras captured synchronously (usable window).
    ├── endGoodTimes (datetime)
    |    End of that contiguous fully-synchronized interval.
```

Notes:
* All times are UTC. The `time` array gives the nominal 2 Hz frame grid; `captures` holds the actual hardware capture timestamps for recorded frames.
* `captureArray`, `totalCaptured`, and `missingFrames` are three views of the same information (full frame grid, per-frame count, and gap indices) and together describe frame dropout across the array.
* The number of columns in `captureArray` and the length of `cams_all` vary between collections depending on how many cameras were online. Camera availability ranged from a single camera up to the full six-camera array.

### `video_metadata.yaml`

A YAML list of manually-reviewed quality assessments for **45** collections selected for analysis. Each entry has:

```
- timestamp: '<YYYYMMDD>T<HHMMSS>Z'   # matches a pickle collection
  ids: [<int>, ...]                    # associated mission / segment id(s)
  description:                         # free-text quality tags
    - frame rate issues
    - glare
    - ...
  quality: good | medium | bad         # overall usability rating
```

Overall quality distribution: **good: 11, medium: 9, bad: 25**. The most common quality tags are *frame rate issues* (19), *glare* (17), *exposure issues* (16), *blurring* (12), and *rectification problems* (11), with positive tags such as *good wave contrast*, *clear waves*, and *good rectification* applied to the higher-rated collections.

## Sharing/Access information

These metadata files describe imagery collected by the Argus system at the USACE FRF as part of the DUNEX field experiment, facilitated by the U.S. Coastal Research Program (USCRP). Use of the underlying Argus imagery and FRF data products is subject to USACE FRF data policies; please acknowledge the FRF and USCRP and contact the data authors before redistribution or commercial use.

Related data sources:
* Argus optical imaging system, USACE Field Research Facility, Duck, NC
* FRF / DUNEX project data portal: https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/Dunex/catalog.html

## Code/Software

The header files are Python `pickle` objects and are most easily read in Python:

```python
import pickle
md = pickle.load(open('ArgusFF_20210917T120000Z_headerData.pickle', 'rb'))
md.keys()
# captureArray, time, startSkipping, endSkipping, cams_all,
# totalCaptured, captures, missingFrames, beginGoodTimes, endGoodTimes
```
