# ML exploration figures

These scripts inspect what the selected SegNeXt model learns and how its
attention and Hamburger modules affect breaking-crest predictions. They use the
same production checkpoint and operating threshold (`0.48`) as video inference.

## Runtime

Run scripts from the repository root. The promoted checkpoint is external to
Git, so either stage it at the canonical location
`model/outputs/promoted/segnext_t_learned_up_skip_4_2/best_model.pth`, set
`OUTPUTS_DIR`, or pass `--checkpoint` explicitly:

```bash
python figures/ml-exploration/20_seg_grad_cam.py \
  --checkpoint /path/to/best_model.pth \
  --device cpu \
  --output-dir /tmp/ml-figures
```

Every script supports `--checkpoint`, `--device`, and `--output-dir`. Scripts
that operate on one timestack also support `--image`. PNG files are written to
the output directory; vector versions are written to its `pdf/` subdirectory.

## Figure inventory

| Script | Question |
|---|---|
| `13_attention_viz_ood.py` | Do attention patterns transfer to OOD frames? |
| `17_pca_pooled.py` | Which pooled encoder directions recur across images? |
| `18_nmf_atoms.py` | What spatial concepts do shared Hamburger-input atoms encode? |
| `19_erf.py` | Which pixels influence a crest prediction under cumulative ablations? |
| `20_seg_grad_cam.py` | Which stage activations support or suppress breaking logits? |
| `21_ham_variability_pooled.py` | How does the Hamburger change class geometry? |
| `21_ham_variability_sepnull.py` | Does that change improve Fisher separation? |
| `22_ham_edit_maps.py` | Where and in which feature directions does the Hamburger edit? |
| `23_ham_ablation_iou.py` | Does bypassing the Hamburger change labelled-mask IoU? |

For quick checks, use `--fast` where available. `--help` lists script-specific
sampling controls. Generated PNG/PDF files are analysis products and are not
tracked by Git.
