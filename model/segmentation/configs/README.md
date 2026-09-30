# Production inference configuration

The two supported configurations use the selected SegNeXt-T model with learned
upsampling, H/4 and H/2 skips, and the validation-selected threshold `0.48`:

- `predict_argus.yaml`: the 36 primary Argus videos.
- `predict_extra.yaml`: the four supplemental videos.

Stage the released model bundle at
`model/outputs/promoted/segnext_t_learned_up_skip_4_2/`, then run from the
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
