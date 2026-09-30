#!/usr/bin/env python3
"""Restartable training loop for the segmentation model study."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from losses import AsymmetricUnifiedFocalLoss

from ablation.checkpoint import (
    complete_run,
    latest_complete_run,
    latest_incomplete_run,
    run_matches_config,
)
from ablation.config import (
    MODEL_ROOT,
    load_all_configs,
    load_training_config,
    resolve_path,
)
from dataset import WaveBreakingDataset
from models import build_model
from ablation.provenance import write_or_validate_provenance
from common.normalization import IMAGENET_MEAN, IMAGENET_STD


THRESHOLDS = tuple(round(value / 100, 2) for value in range(10, 91, 5))
BOUNDARY_TOLERANCE = 2
IMAGE_MEAN = IMAGENET_MEAN
IMAGE_STD = IMAGENET_STD


@contextmanager
def exclusive_training_lock(config: dict) -> Iterator[None]:
    """Serialize concurrent launches of the same configured experiment."""
    output_root = resolve_path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".training.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def complete_run_since(config: dict, requested_at_ns: int) -> Path | None:
    """Find a matching run completed while this launch was waiting."""
    output_root = resolve_path(config["output_dir"])
    matches = sorted(
        path
        for path in output_root.glob("run_*")
        if (
            complete_run(path)
            and run_matches_config(path, config)
            and (path / "complete.json").stat().st_mtime_ns >= requested_at_ns
        )
    )
    return matches[-1] if matches else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_loss_inputs(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=1)
    two_class_targets = torch.cat((1.0 - targets, targets), dim=1)
    return probabilities, two_class_targets


def empty_boundary_counts(
    device: torch.device | str = "cpu",
) -> dict[float, torch.Tensor]:
    # Boundary hits/totals followed by TP/FP/FN/TN for cheap pixel metrics.
    return {
        threshold: torch.zeros(8, dtype=torch.float64, device=device)
        for threshold in THRESHOLDS
    }


def update_boundary_counts(
    counts: dict[float, torch.Tensor],
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    kernel = 2 * BOUNDARY_TOLERANCE + 1
    target_band = F.max_pool2d(
        targets, kernel, stride=1, padding=BOUNDARY_TOLERANCE
    )
    target_total = targets.sum(dtype=torch.float64)
    target_binary = targets > 0.5
    for threshold, values in counts.items():
        prediction = (probabilities > threshold).float()
        prediction_band = F.max_pool2d(
            prediction, kernel, stride=1, padding=BOUNDARY_TOLERANCE
        )
        values[0].add_((prediction * target_band).sum(dtype=torch.float64))
        values[1].add_(prediction.sum(dtype=torch.float64))
        values[2].add_((targets * prediction_band).sum(dtype=torch.float64))
        values[3] += target_total
        prediction_binary = prediction > 0.5
        values[4].add_((prediction_binary & target_binary).sum(dtype=torch.float64))
        values[5].add_((prediction_binary & ~target_binary).sum(dtype=torch.float64))
        values[6].add_((~prediction_binary & target_binary).sum(dtype=torch.float64))
        values[7].add_((~prediction_binary & ~target_binary).sum(dtype=torch.float64))


def select_iou_threshold(
    counts: dict[float, torch.Tensor],
) -> tuple[float, float, float, float, int, int, int, int]:
    scored = []
    threshold_values = list(counts)
    host_counts = torch.stack([counts[key] for key in threshold_values]).cpu().numpy()
    for threshold, values in zip(threshold_values, host_counts):
        pred_hit, pred_total, target_hit, target_total = values[:4]
        precision = pred_hit / (pred_total + 1e-8)
        recall = target_hit / (target_total + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        tp, fp, fn = values[4:7]
        iou_fg = tp / (tp + fp + fn + 1e-8)
        scored.append((
            iou_fg,
            -abs(threshold - 0.5),
            -threshold,
            threshold,
            f1,
            precision,
            recall,
        ))
    best = max(scored)
    threshold = best[3]
    selected_index = threshold_values.index(threshold)
    tp, fp, fn, tn = (int(value) for value in host_counts[selected_index, 4:])
    return threshold, best[4], best[5], best[6], tp, fp, fn, tn


def update_binary_counts(
    counts: torch.Tensor,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> None:
    """Accumulate TP/FP/FN/TN without per-batch CPU synchronization."""
    prediction = probabilities.detach() > threshold
    target = targets > 0.5
    counts[0].add_((prediction & target).sum(dtype=torch.float64))
    counts[1].add_((prediction & ~target).sum(dtype=torch.float64))
    counts[2].add_((~prediction & target).sum(dtype=torch.float64))
    counts[3].add_((~prediction & ~target).sum(dtype=torch.float64))


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Return inexpensive pixel metrics for training feedback."""
    eps = 1e-8
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": iou_fg,
        "iou_fg": iou_fg,
        "iou_bg": iou_bg,
        "miou": 0.5 * (iou_fg + iou_bg),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp / (fp + tn + eps),
        "pixel_false_negative_rate": fn / (tp + fn + eps),
    }


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGE_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def error_maps(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    error = predictions - targets
    red = torch.where(error >= 0, torch.ones_like(error), 1 + error)
    green = 1 - error.abs()
    blue = torch.where(error <= 0, torch.ones_like(error), 1 - error)
    return torch.cat((red, green, blue), dim=1)


def overlays(images: torch.Tensor, values: torch.Tensor, alpha: float = 0.6) -> torch.Tensor:
    images = denormalize_images(images)
    red = torch.zeros_like(images)
    red[:, 0] = 1
    blend = alpha * values.repeat(1, 3, 1, 1)
    return images * (1 - blend) + red * blend


def write_tensorboard_preview(
    writer: SummaryWriter,
    prefix: str,
    preview: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    threshold: float,
    epoch: int,
) -> None:
    """Write eight training diagnostics without another forward pass."""
    images, targets, probabilities = preview
    predictions_05 = (probabilities > 0.5).float()
    predictions_selected = (probabilities > threshold).float()
    writer.add_images(f"{prefix}/1_Images", denormalize_images(images), epoch)
    writer.add_images(f"{prefix}/2_GroundTruth", targets, epoch)
    writer.add_images(f"{prefix}/3_Predictions_0.5", predictions_05, epoch)
    writer.add_images(
        f"{prefix}/4_Predictions_Selected", predictions_selected, epoch
    )
    writer.add_images(f"{prefix}/5_Probabilities", probabilities, epoch)
    writer.add_images(
        f"{prefix}/6_Error_Map", error_maps(predictions_selected, targets), epoch
    )
    writer.add_images(
        f"{prefix}/7_Overlay_Probs", overlays(images, probabilities), epoch
    )
    writer.add_images(
        f"{prefix}/8_Overlay_Preds", overlays(images, predictions_selected), epoch
    )


def write_history(metrics_dir: Path, history: list[dict]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if history:
        temporary = metrics_dir / "history.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        temporary.replace(metrics_dir / "history.csv")


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
    history: list[dict],
    selection_metric: str,
    best_selection_value: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "history": history,
        "selection_metric": selection_metric,
        "selection_value": best_selection_value,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
    }


def make_loaders(
    config: dict, device: torch.device
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    workers = int(config["num_workers"])
    if device.type == "mps" and workers > 4:
        workers = 4
    persistent = bool(config["persistent_workers"]) and workers > 0
    common = dict(
        dataset_root=resolve_path(config["dataset_root"]),
        transform_size=int(config["image_size"]),
        patches_per_image=int(config["patches_per_image"]),
        seed=int(config["seed"]),
        cache_images=bool(config["cache_images"]),
    )
    train_dataset = WaveBreakingDataset(
        split="train", augment=bool(config["augment_train"]), **common
    )
    validation_dataset = WaveBreakingDataset(split="val", augment=False, **common)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_common = dict(
        batch_size=int(config["batch_size"]),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent,
    )
    if workers:
        loader_common["prefetch_factor"] = int(config["prefetch_factor"])
    training_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )
    return training_loader, validation_loader, generator


def initialize_model(config: dict) -> torch.nn.Module:
    model = build_model(config)
    if config["arch"] == "segnext" and config.get("pretrained_path"):
        pretrained = resolve_path(config["pretrained_path"])
        if not pretrained.is_file():
            raise FileNotFoundError(pretrained)
        model.load_pretrained_backbone(pretrained)
    return model


def run_training(
    config: dict,
    device: torch.device,
    *,
    smoke: bool = False,
    resume: Path | None = None,
) -> Path:
    config = dict(config)
    selection_metric = str(config.get("select_metric", "iou_fg"))
    if selection_metric != "iou_fg":
        raise ValueError("The study selects checkpoints by foreground IoU")
    if smoke:
        config["epochs"] = 1
        config["patches_per_image"] = 1
        config["num_workers"] = 0
        config["output_dir"] = str(Path(config["output_dir"]) / "smoke")
    torch.set_num_threads(int(config["torch_num_threads"]))
    seed_everything(int(config["seed"]))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    training_loader, validation_loader, loader_generator = make_loaders(config, device)
    model = initialize_model(config).to(device)
    criterion = AsymmetricUnifiedFocalLoss(**config["loss_params"]).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["lr"]),
        epochs=int(config["epochs"]),
        steps_per_epoch=len(training_loader),
        pct_start=0.15,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    start_epoch = 0
    history: list[dict] = []
    best_selection_value = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    if resume is not None:
        run_dir = resume.parent
        write_or_validate_provenance(run_dir, config, model)
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload["config"]["id"] != config["id"]:
            raise ValueError("Resume checkpoint belongs to another model")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        history = list(payload["history"])
        checkpoint_metric = str(payload["selection_metric"])
        if checkpoint_metric != selection_metric:
            raise ValueError(
                f"Cannot resume a {checkpoint_metric!r}-selected checkpoint "
                f"with select_metric={selection_metric!r}"
            )
        best_selection_value = float(payload["selection_value"])
        best_epoch = int(payload["best_epoch"])
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
    else:
        output_root = resolve_path(config["output_dir"])
        run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_or_validate_provenance(run_dir, config, model)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tensorboard")
    print(
        f"Training {config['label']} on {device}; params="
        f"{sum(parameter.numel() for parameter in model.parameters()):,}; "
        f"train_batches={len(training_loader)}; val_batches={len(validation_loader)}",
        flush=True,
    )

    for epoch in range(start_epoch, int(config["epochs"])):
        visualize = epoch == 0 or (
            (epoch + 1) % int(config.get("visualize_every", 5)) == 0
        )
        training_loader.dataset.set_epoch(epoch)
        loader_generator.manual_seed(int(config["seed"]) + epoch)
        model.train()
        training_loss_sum = torch.zeros((), device=device)
        training_bce_sum = torch.zeros((), device=device)
        training_counts = torch.zeros(4, dtype=torch.float64, device=device)
        training_preview = None
        for images, targets in tqdm(
            training_loader, desc=f"epoch {epoch + 1} train", leave=False
        ):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss_inputs, loss_targets = prepare_loss_inputs(logits, targets)
            loss = criterion(loss_inputs, loss_targets)
            foreground = loss_inputs[:, 1:2].detach()
            training_bce_sum.add_(F.binary_cross_entropy(
                foreground.clamp(1e-7, 1 - 1e-7), targets
            ))
            update_binary_counts(training_counts, foreground, targets)
            if visualize and training_preview is None:
                training_preview = (
                    images[:8].detach().cpu(),
                    targets[:8].detach().cpu(),
                    foreground[:8].cpu(),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch + 1}")
            optimizer.step()
            scheduler.step()
            training_loss_sum.add_(loss.detach())
        training_loss = float(training_loss_sum / len(training_loader))
        training_bce = float(training_bce_sum / len(training_loader))
        train_tp, train_fp, train_fn, train_tn = (
            int(value) for value in training_counts.cpu().numpy()
        )
        train_metrics = binary_metrics(train_tp, train_fp, train_fn, train_tn)

        model.eval()
        validation_loss_sum = torch.zeros((), device=device)
        validation_bce_sum = torch.zeros((), device=device)
        counts = empty_boundary_counts(device)
        validation_preview = None
        with torch.inference_mode():
            for images, targets in tqdm(
                validation_loader, desc=f"epoch {epoch + 1} validate", leave=False
            ):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss_inputs, loss_targets = prepare_loss_inputs(logits, targets)
                validation_loss_sum.add_(criterion(loss_inputs, loss_targets))
                foreground = loss_inputs[:, 1:2]
                validation_bce_sum.add_(F.binary_cross_entropy(
                    foreground.clamp(1e-7, 1 - 1e-7), targets
                ))
                update_boundary_counts(counts, foreground, targets)
                if visualize and validation_preview is None:
                    validation_preview = (
                        images[:8].detach().cpu(),
                        targets[:8].detach().cpu(),
                        foreground[:8].detach().cpu(),
                    )
        validation_loss = float(validation_loss_sum / len(validation_loader))
        validation_bce = float(validation_bce_sum / len(validation_loader))
        (
            threshold,
            boundary_f1,
            boundary_precision,
            boundary_recall,
            tp,
            fp,
            fn,
            tn,
        ) = (
            select_iou_threshold(counts)
        )
        validation_metrics = binary_metrics(tp, fp, fn, tn)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": training_loss,
            "train_bce": training_bce,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            "val_loss": validation_loss,
            "val_bce": validation_bce,
            "val_threshold": threshold,
            "val_boundary_f1": boundary_f1,
            "val_boundary_precision": boundary_precision,
            "val_boundary_recall": boundary_recall,
            **{f"val_{name}": value for name, value in validation_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        writer.add_scalar("Loss/train", training_loss, epoch + 1)
        writer.add_scalar("Loss/train_bce", training_bce, epoch + 1)
        writer.add_scalar("Loss/val", validation_loss, epoch + 1)
        writer.add_scalar("Loss/val_bce", validation_bce, epoch + 1)
        for name, value in train_metrics.items():
            writer.add_scalar(f"Metrics/train_{name}", value, epoch + 1)
        for name, value in validation_metrics.items():
            writer.add_scalar(f"Metrics/val_{name}", value, epoch + 1)
        writer.add_scalar("Metrics/val_boundary_f1", boundary_f1, epoch + 1)
        writer.add_scalar(
            "Metrics/val_boundary_precision", boundary_precision, epoch + 1
        )
        writer.add_scalar("Metrics/val_boundary_recall", boundary_recall, epoch + 1)
        writer.add_scalar("Metrics/val_optimal_threshold", threshold, epoch + 1)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch + 1)
        if visualize:
            if training_preview is not None:
                write_tensorboard_preview(
                    writer, "Train_Random", training_preview, threshold, epoch + 1
                )
            if validation_preview is not None:
                write_tensorboard_preview(
                    writer, "Val_Fixed", validation_preview, threshold, epoch + 1
                )
        writer.flush()

        current_selection_value = validation_metrics["iou_fg"]
        improved = current_selection_value > best_selection_value + 1e-5
        if improved:
            best_selection_value = current_selection_value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = checkpoint_payload(
            epoch,
            model,
            optimizer,
            scheduler,
            config,
            history,
            selection_metric,
            best_selection_value,
            best_epoch,
            epochs_without_improvement,
        )
        atomic_torch_save(payload, run_dir / "last_model.pth")
        if improved:
            atomic_torch_save(payload, run_dir / "best_model.pth")
        write_history(metrics_dir, history)
        print(
            f"epoch={epoch + 1} train_loss={training_loss:.5f} "
            f"val_loss={validation_loss:.5f} bF1={boundary_f1:.4f} "
            f"IoU={validation_metrics['iou_fg']:.4f} "
            f"Dice={validation_metrics['dice']:.4f} threshold={threshold:.2f}",
            flush=True,
        )
        if epochs_without_improvement >= int(config["early_stop_patience"]):
            print(
                f"Early stopping after {epochs_without_improvement} epochs without "
                f"{selection_metric} improvement",
                flush=True,
            )
            break

    writer.close()
    (metrics_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch + 1,
                "selection_metric": selection_metric,
                "best_selection_value": best_selection_value,
                "best_boundary_f1": max(
                    (float(row["val_boundary_f1"]) for row in history),
                    default=None,
                ),
                "best_iou_fg": max(
                    (float(row["val_iou_fg"]) for row in history),
                    default=None,
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "model_id": config["id"],
            },
            indent=2,
        )
        + "\n"
    )
    return run_dir


def append_status(output_root: Path, event: str, detail: str = "") -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat()
    with (output_root / "status.tsv").open("a") as handle:
        handle.write(f"{event}\t{timestamp}\t{detail}\n")


def train_all(device: str, *, smoke: bool = False) -> None:
    """Train every model in manifest order in isolated subprocesses."""
    manifest, entries = load_all_configs()
    entries = sorted(
        entries,
        key=lambda pair: not bool(pair[0].get("training_priority")),
    )
    output_root = resolve_path(manifest["output_root"])
    append_status(output_root, "training_started", f"device={device} smoke={smoke}")

    for entry, config in entries:
        try:
            run = latest_complete_run(config)
        except FileNotFoundError:
            run = None
        if run is not None and not smoke:
            append_status(output_root, f"{config['id']}_skipped", str(run))
            continue

        arguments = [
            sys.executable,
            "-m",
            "train",
            "--config",
            entry["config"],
            "--device",
            device,
        ]
        if smoke:
            arguments.append("--smoke")
        elif (incomplete := latest_incomplete_run(config)) is not None:
            arguments.extend(("--resume", str(incomplete / "last_model.pth")))

        append_status(output_root, f"{config['id']}_started", entry["config"])
        subprocess.run(arguments, cwd=MODEL_ROOT, check=True)
        append_status(output_root, f"{config['id']}_complete")
        gc.collect()

    append_status(output_root, "training_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--config")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.all:
        if args.resume is not None:
            parser.error("--resume requires --config")
        from ablation.verify import verify_inputs

        verify_inputs()
        train_all(args.device or "cuda", smoke=args.smoke)
        return

    config = load_training_config(args.config)
    requested_at_ns = time.time_ns()
    with exclusive_training_lock(config):
        run = None if args.resume is not None else complete_run_since(
            config, requested_at_ns
        )
        if run is None:
            run = run_training(
                config,
                choose_device(args.device),
                smoke=args.smoke,
                resume=(
                    resolve_path(args.resume) if args.resume is not None else None
                ),
            )
        else:
            print(
                "Reusing matching run completed by a concurrent launch: "
                f"{run.relative_to(MODEL_ROOT)}",
                flush=True,
            )
    print(f"Completed run: {run.relative_to(MODEL_ROOT)}")


if __name__ == "__main__":
    main()
