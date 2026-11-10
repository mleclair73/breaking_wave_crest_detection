# Model data inventory

Only dataset specifications, annotations, provenance, and rebuild/verification
code belong in Git. Image crops, masks, archives, and generated figures are
external assets staged into the directories described below.

## Canonical datasets

| Dataset | Required local pixels | Versioned specification | Role |
|---|---:|---|---|
| `full_split/` | 256 images and 256 masks | mapping, split manifests, README, verification/rebuild code | canonical train/validation/test dataset |
| `ood_test/` | 20 images and 20 masks | source manifest, mapping, README, build/import code | date-disjoint post-selection evaluation |

Canonical training and evaluation use `full_split/`.

After staging `full_split/images/` and `full_split/masks/`, validate the dataset
from the `model/` directory:

```bash
python data/full_split/verify_dataset.py
```

The command checks the frozen mapping, image/mask pairing, dimensions, split
leakage, content inventory, and loader behavior. See `full_split/README.md` and
`ood_test/README.md` for details.
