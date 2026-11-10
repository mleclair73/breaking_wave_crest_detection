# Production inference configuration

`predict_argus.yaml` runs the selected SegNeXt-T model (learned upsampling,
H/4 and H/2 skips, validation-selected threshold `0.48`) on all 40 Argus videos.

Stage the released model bundle at
`model/outputs/released/segnext_t_learned_up_skip_4_2/`, then run from the
repository root:

```bash
python model/segmentation/predict_video.py \
  model/segmentation/configs/predict_argus.yaml
```

The default device is `auto`, which selects CUDA, then MPS, then CPU. Override
it without editing YAML when needed:

```bash
python model/segmentation/predict_video.py \
  model/segmentation/configs/predict_argus.yaml --device cpu
```

`--model-path`, `--input-path`, and `--output-path` provide portable command-line
overrides. YAML paths support the `${REPO_ROOT}`, `${DATA_ROOT}`,
`${OUTPUTS_DIR}`, and `${PREDICTIONS_DIR}` placeholders resolved by
`dunex_paths`. Training configuration belongs to the canonical `ablation`
package rather than this directory.
