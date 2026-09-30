#!/usr/bin/env python3
"""Generate the canonical CSV and LaTeX result tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import load_all_configs, resolve_path
from .evaluate import write_csv


ACCURACY_COLUMNS = (
    "boundary_f1",
    "iou_fg",
    "cldice",
    "crest_precision",
    "crest_recall",
    "crest_f1",
)
DISPLAY = {
    "dice": "Dice",
    "boundary_f1": "bF1",
    "iou_fg": r"IoU$_\mathrm{fg}$",
    "cldice": "clDice",
    "crest_precision": "Crest Prec.",
    "crest_recall": "Crest Recall",
    "crest_f1": "Crest F1",
    "parameters": "Params",
    "flops": "FLOPs",
    "threshold": "Threshold",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def indexed(path: Path) -> dict[str, dict[str, str]]:
    return {row["model_id"]: row for row in read_csv(path)}


def latex_escape(value: str) -> str:
    return (
        value.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def number(value: str | float, column: str) -> str:
    value = float(value)
    if column == "parameters":
        return f"{value / 1e6:.2f}M"
    if column == "flops":
        return f"{value / 1e9:.2f}G"
    if column == "threshold":
        return f"{value:.2f}"
    return f"{value:.3f}"


def write_latex(
    path: Path,
    rows: list[dict],
    leading: list[tuple[str, str]],
    numeric: list[str],
    caption: str,
    label: str,
    *,
    bold_accuracy: bool = True,
) -> None:
    best = {
        column: max(float(row[column]) for row in rows)
        for column in numeric
        if bold_accuracy and column in ACCURACY_COLUMNS
    }
    columns = "l" * len(leading) + "c" * len(numeric)
    header = [title for _, title in leading] + [DISPLAY[column] for column in numeric]
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{3pt}",
        f"    \\begin{{tabular}}{{{columns}}}",
        r"        \toprule",
        "        " + " & ".join(header) + r" \\",
        r"        \midrule",
    ]
    for row in rows:
        values = [latex_escape(str(row[key])) for key, _ in leading]
        for column in numeric:
            formatted = number(row[column], column)
            if column in best and abs(float(row[column]) - best[column]) < 5e-13:
                formatted = rf"\textbf{{{formatted}}}"
            values.append(formatted)
        lines.append("        " + " & ".join(values) + r" \\")
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines))


def metric_row(source: dict[str, str], **leading) -> dict:
    return {
        **leading,
        **{column: float(source[column]) for column in ACCURACY_COLUMNS},
        "threshold": float(source["threshold"]),
    }


def run_report() -> None:
    manifest, entries = load_all_configs()
    output_root = resolve_path(manifest["output_root"])
    metrics_root = output_root / "metrics"
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    validation = indexed(metrics_root / "validation" / "summary.csv")
    missing = indexed(metrics_root / "missing_data" / "summary.csv")
    ood = indexed(metrics_root / "ood" / "summary.csv")
    test = indexed(metrics_root / "test" / "summary.csv")
    costs = indexed(metrics_root / "costs.csv")

    model_rows = []
    for entry, config in entries:
        if not entry.get("table"):
            continue
        row = metric_row(validation[config["id"]], model=config["label"])
        row["parameters"] = int(costs[config["id"]]["parameters"])
        row["flops"] = int(costs[config["id"]]["flops"])
        model_rows.append(row)
    write_csv(tables / "model_comparison.csv", model_rows)
    write_latex(
        tables / "model_comparison.tex",
        model_rows,
        [("model", "Model")],
        [*ACCURACY_COLUMNS, "parameters", "flops"],
        "Single-seed architecture comparison using complete-image Hanning-blended tiled inference. Pixel metrics are means across images; crest metrics use dataset-level quality-weighted one-to-one crest matching. FLOPs are $2\\times$MACs for one $224\\times224$ RGB input.",
        "tab:model-comparison",
    )

    augmentation_rows = []
    for dataset_name, source in (
        ("Clean validation", validation),
        ("OOD diagnostic", ood),
    ):
        for entry, config in entries:
            role = entry.get("augmentation_role")
            if not role:
                continue
            augmentation_rows.append(metric_row(
                source[config["id"]],
                dataset=dataset_name,
                training="Augmented" if role == "augmented" else "No augmentation",
            ))
    write_csv(tables / "augmentation.csv", augmentation_rows)
    write_latex(
        tables / "augmentation.tex",
        augmentation_rows,
        [("dataset", "Dataset"), ("training", "Training")],
        list(ACCURACY_COLUMNS),
        "Effect of training augmentation on clean validation data and the date-disjoint OOD diagnostic set. Each model uses its independently calibrated clean-validation threshold, frozen for OOD evaluation.",
        "tab:augmentation",
        bold_accuracy=False,
    )

    missing_rows = []
    for entry, config in entries:
        role = entry.get("augmentation_role")
        if not role:
            continue
        training = "Augmented" if role == "augmented" else "No augmentation"
        missing_rows.append(metric_row(
            validation[config["id"]], training=training, condition="Clean"
        ))
        missing_rows.append(metric_row(
            missing[config["id"]], training=training, condition="Missing data"
        ))
    write_csv(tables / "missing_data.csv", missing_rows)
    write_latex(
        tables / "missing_data.tex",
        missing_rows,
        [("training", "Training"), ("condition", "Condition")],
        ["threshold", *ACCURACY_COLUMNS],
        "Missing-data robustness using deterministic mixed corruption before tiled inference. Each model's clean-validation threshold is frozen across both conditions.",
        "tab:missing-data",
        bold_accuracy=False,
    )

    ood_rows = [
        metric_row(ood[config["id"]], model=config["label"])
        for entry, config in entries
        if entry.get("ood_comparison")
    ]
    write_csv(tables / "ood_swin_segnext.csv", ood_rows)
    write_latex(
        tables / "ood_swin_segnext.tex",
        ood_rows,
        [("model", "Model")],
        list(ACCURACY_COLUMNS),
        "Date-disjoint OOD diagnostic comparison using thresholds frozen on clean validation data. This diagnostic set was not used for architecture selection and is distinct from the holdout test set.",
        "tab:ood-swin-segnext",
    )

    holdout_rows = []
    for entry, config in entries:
        if not entry.get("holdout_test"):
            continue
        source = test[config["id"]]
        row = metric_row(source, model=config["label"])
        row["dice"] = float(source["dice"])
        holdout_rows.append(row)
    write_csv(tables / "holdout_test.csv", holdout_rows)
    write_latex(
        tables / "holdout_test.tex",
        holdout_rows,
        [("model", "Model")],
        ["threshold", "dice", *ACCURACY_COLUMNS],
        "Final source-disjoint holdout-test performance of the selected model. The decision threshold was frozen from validation before the test set was evaluated.",
        "tab:holdout-test",
        bold_accuracy=False,
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_report()


if __name__ == "__main__":
    main()
