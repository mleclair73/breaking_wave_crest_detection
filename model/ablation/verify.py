"""Validate model-study configuration and required local inputs."""

from __future__ import annotations

import ast

from .config import PACKAGE_ROOT, load_all_configs, resolve_path


def verify_inputs() -> None:
    """Check configurations, datasets, pretrained weights, and import boundaries."""
    _, entries = load_all_configs()
    forbidden_roots = {
        "segmentation",
        "data",
        "model_ablation",
        "unified_focal_loss_pytorch",
    }
    violations = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".", 1)[0] in forbidden_roots:
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {name}")
    if violations:
        raise RuntimeError("Forbidden repository imports:\n" + "\n".join(violations))

    for _, config in entries:
        dataset = resolve_path(config["dataset_root"])
        for required in ("image_mask_mapping.csv", "images", "masks"):
            if not (dataset / required).exists():
                raise FileNotFoundError(dataset / required)
        if config["arch"] == "segnext":
            pretrained = resolve_path(config["pretrained_path"])
            if not pretrained.is_file():
                raise FileNotFoundError(pretrained)


def main() -> None:
    verify_inputs()
    print("Configuration, dataset, weight, and import-boundary verification passed.")


if __name__ == "__main__":
    main()
