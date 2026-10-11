from __future__ import annotations

import csv
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from .config import dump_yaml_file
from .distributed import (
    DistributedContext,
    all_gather_object,
    barrier,
    is_distributed,
    is_main_process,
    unwrap_model,
)
from .metrics import evaluate_detection, save_predictions
from .missbank import compute_missbank_stability_metrics, merge_missbank_epoch_snapshots
from .module_metadata import collect_enabled_module_configs

_MISSBANK_STATE_KEY = "missbank._extra_state"


def fit(
    model: torch.nn.Module,
    model_config: dict[str, Any],
    model_config_path: str | Path,
    runtime_config: dict[str, Any],
    runtime_config_path: str | Path,
    train_loader,
    val_loader,
    device: torch.device,
    arch: str,
    distributed: DistributedContext | None = None,
    module_config_paths: Mapping[str, str | Path] | None = None,
) -> list[dict[str, Any]]:
    distributed = distributed or DistributedContext(enabled=False, device=device)
    main_process = is_main_process(distributed)
    train_config = runtime_config["train"]
    checkpoint_config = runtime_config["checkpoint"]
    output_dir = Path(runtime_config["output_dir"])
    checkpoint_dir = Path(checkpoint_config["dir"])
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        probe_path = checkpoint_dir / ".write_probe"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    barrier(distributed)

    rank_offset = distributed.rank if is_distributed(distributed) else 0
    seed_everything(int(runtime_config["seed"]) + rank_offset)
    model.to(device)
    if is_distributed(distributed):
        if device.type != "cuda" or device.index is None:
            raise RuntimeError("DDP training requires an explicit CUDA device per rank.")
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )

    amp = runtime_config["amp"]
    total_epochs = train_config["epochs"]
    max_iterations = train_config["max_iterations"]
    optimizer = build_optimizer(model, runtime_config["optimizer"])
    scheduler = build_scheduler(optimizer, runtime_config["scheduler"])
    step_scheduler_per_iteration = str(runtime_config["scheduler"]["unit"]).lower() == "iteration"
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    history: list[dict[str, Any]] = []
    hard_replay_history: list[dict[str, Any]] = []
    missbank_history: list[dict[str, Any]] = []
    previous_missbank_snapshot: dict[str, Any] | None = None
    best_metric = float("-inf")
    start_epoch = 0
    global_iteration = 0

    resume_path = checkpoint_config["resume"]
    if resume_path:
        resume_scheduler = checkpoint_config["resume_scheduler"]
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer if checkpoint_config["resume_optimizer"] else None,
            scheduler=scheduler if resume_scheduler else None,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        global_iteration = int(checkpoint.get("iteration", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))
        if checkpoint_config["reset_optimizer_lr"]:
            for group in optimizer.param_groups:
                group["lr"] = float(runtime_config["optimizer"]["lr"])
        if not resume_scheduler:
            _align_scheduler_to_step(
                scheduler,
                step=global_iteration if step_scheduler_per_iteration else start_epoch,
            )

    if main_process:
        if resume_path:
            hard_replay_history = _read_json_list(
                output_dir / "hard-replay" / "hard_replay_epoch.json"
            )
            missbank_history = _read_json_list(output_dir / "missbank" / "missbank_epoch.json")
        persist_run_metadata(
            output_dir=output_dir,
            arch=arch,
            model_config=model_config,
            model_config_path=model_config_path,
            runtime_config=runtime_config,
            runtime_config_path=runtime_config_path,
            module_configs=collect_enabled_module_configs(
                arch,
                config_paths=module_config_paths or runtime_config.get("_module_config_paths"),
            ),
        )
    barrier(distributed)

    log_interval = train_config["log_interval"]
    monitor = checkpoint_config["monitor"]
    primary_metric = runtime_config["metrics"]["primary"]
    eval_every_epochs = train_config["eval_every_epochs"]
    eval_every_iterations = train_config["eval_every_iterations"]
    save_every_epochs = checkpoint_config["save_every_epochs"]
    last_eval_iteration = -1
    iteration_timer_start = time.perf_counter()

    def save(name: str, epoch: int) -> None:
        save_checkpoint(
            checkpoint_dir / name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            iteration=global_iteration,
            best_metric=best_metric,
        )

    def validate(stage_label: str, epoch_index: int) -> dict[str, float]:
        nonlocal best_metric
        val_metrics, _ = evaluate(
            model=unwrap_model(model),
            runtime_config=runtime_config,
            data_loader=val_loader,
            device=device,
            output_dir=None,
            log_interval=log_interval,
            stage_label=stage_label,
            epoch_index=epoch_index,
            total_epochs=total_epochs,
        )
        current_value = val_metrics.get(monitor)
        if current_value is None:
            raise KeyError(f"Monitored metric {monitor!r} was not found in validation output.")
        if current_value > best_metric:
            best_metric = current_value
            if checkpoint_config["save_best"]:
                save("best.pt", epoch_index + 1)
        return val_metrics

    for epoch in range(start_epoch, total_epochs):
        if max_iterations is not None and global_iteration >= max_iterations:
            break

        _set_data_loader_epoch(train_loader, epoch + 1)
        _set_missbank_epoch(model, epoch + 1)
        _refresh_hard_replay(train_loader, model, epoch + 1)

        max_steps = None if max_iterations is None else max_iterations - global_iteration
        num_steps = len(train_loader)
        epoch_completed = max_steps is None or max_steps >= num_steps

        def on_iteration(iteration: int, train_snapshot: dict[str, float]) -> None:
            nonlocal global_iteration, last_eval_iteration, iteration_timer_start
            global_iteration = iteration
            if eval_every_iterations is None or val_loader is None:
                return
            reached_max = max_iterations is not None and global_iteration >= max_iterations
            if not (global_iteration % eval_every_iterations == 0 or reached_max):
                return
            if last_eval_iteration == global_iteration:
                return

            train_time_sec = _max_distributed_float(
                time.perf_counter() - iteration_timer_start,
                distributed,
            )
            barrier(distributed)
            if main_process:
                train_snapshot["epoch_time_sec"] = float(train_time_sec)
                record: dict[str, Any] = {
                    "record_type": "iteration_eval",
                    "epoch": epoch + 1,
                    "iteration": global_iteration,
                    "train": train_snapshot,
                    "val": validate(f"val@iter {global_iteration}", epoch),
                }
                history.append(record)
                _write_history_outputs(output_dir, history)
                print(f"[iter {global_iteration}] {_format_metrics(record, primary_metric)}")
                last_eval_iteration = global_iteration
            barrier(distributed)
            iteration_timer_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler if step_scheduler_per_iteration else None,
            data_loader=train_loader,
            device=device,
            amp=amp,
            scaler=scaler,
            log_interval=log_interval if main_process else 0,
            grad_clip_norm=train_config["grad_clip_norm"],
            epoch_index=epoch,
            total_epochs=total_epochs,
            max_steps=max_steps,
            start_iteration=global_iteration,
            total_iterations=max_iterations,
            on_iteration=on_iteration,
            distributed=distributed,
        )

        remiss_mining_time_sec = None
        hard_replay_summary = None
        if epoch_completed:
            remiss_mining_time_sec = _run_missbank_offline_mining(
                model=model,
                data_loader=train_loader,
                device=device,
                amp=amp,
                epoch=epoch + 1,
                total_epochs=total_epochs,
                log_interval=log_interval if main_process else 0,
                distributed=distributed,
            )
            missbank_snapshot = _collect_missbank_epoch_snapshot(model, epoch + 1, distributed)
            if main_process and missbank_snapshot is not None:
                missbank_history.append(
                    _build_missbank_epoch_metrics(
                        missbank_snapshot,
                        previous_snapshot=previous_missbank_snapshot,
                    )
                )
                _write_missbank_outputs(output_dir, missbank_history)
                previous_missbank_snapshot = missbank_snapshot
            hard_replay_summary = _get_hard_replay_summary(train_loader)

        record: dict[str, Any] = {
            "record_type": "epoch_summary",
            "epoch": epoch + 1,
            "iteration": global_iteration,
            "train": train_metrics,
        }
        if remiss_mining_time_sec is not None:
            record["remiss_mining_time_sec"] = remiss_mining_time_sec
        if main_process and hard_replay_summary is not None:
            hard_replay_history.append(hard_replay_summary)
            _write_hard_replay_outputs(output_dir, hard_replay_history, hard_replay_summary)

        reached_max_iterations = max_iterations is not None and global_iteration >= max_iterations
        should_eval = (
            val_loader is not None
            and last_eval_iteration != global_iteration
            and (
                (eval_every_epochs is not None and (epoch + 1) % eval_every_epochs == 0)
                or epoch + 1 == total_epochs
                or reached_max_iterations
            )
        )
        if should_eval and main_process:
            record["val"] = validate("val", epoch)
            last_eval_iteration = global_iteration
        barrier(distributed)

        if main_process:
            if checkpoint_config["save_last"]:
                save("last.pt", epoch + 1)
            if save_every_epochs is not None and (epoch + 1) % save_every_epochs == 0:
                save(f"epoch_{epoch + 1:04d}.pt", epoch + 1)

        if not step_scheduler_per_iteration:
            scheduler.step()

        if main_process:
            history.append(record)
            _write_history_outputs(output_dir, history)
            summary = _format_metrics(record, primary_metric, hard_replay=hard_replay_summary)
            if max_iterations is not None:
                progress = f"iter {global_iteration}/{max_iterations}"
            else:
                progress = f"epoch {epoch + 1}/{total_epochs}"
            print(f"[{progress}] {summary}")

    if main_process and val_loader is not None:
        for name in ("last", "best"):
            _evaluate_saved_checkpoint(
                name,
                model=unwrap_model(model),
                runtime_config=runtime_config,
                data_loader=val_loader,
                device=device,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
            )

    barrier(distributed)
    return history


def _evaluate_saved_checkpoint(
    name: str,
    *,
    model: torch.nn.Module,
    runtime_config: dict[str, Any],
    data_loader,
    device: torch.device,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    checkpoint_path = checkpoint_dir / f"{name}.pt"
    if not checkpoint_path.is_file():
        return
    print(f"[{name}_val] Loading {name} checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    epoch = int(checkpoint.get("epoch", 0))
    metrics, _ = evaluate_detection(
        model=model,
        data_loader=data_loader,
        device=device,
        amp=runtime_config["amp"],
        metrics_type=runtime_config["metrics"]["type"],
        log_interval=runtime_config["train"]["log_interval"],
        stage_label=f"{name}_val",
    )
    payload: dict[str, Any] = {"epoch": epoch}
    if "iteration" in checkpoint:
        payload["iteration"] = int(checkpoint["iteration"])
    payload.update(metrics)
    metrics_path = output_dir / f"{name}_val_metrics.json"
    _write_json(metrics_path, payload)
    primary = runtime_config["metrics"]["primary"]
    print(
        f"[{name}_val] epoch={epoch} "
        f"{primary}={metrics.get(primary, float('nan')):.4f} "
        f"saved to {metrics_path}"
    )


def evaluate(
    model: torch.nn.Module,
    runtime_config: dict[str, Any],
    data_loader,
    device: torch.device,
    output_dir: str | Path | None,
    log_interval: int = 0,
    stage_label: str = "eval",
    epoch_index: int | None = None,
    total_epochs: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    checkpoint_path = runtime_config["checkpoint"].get("path")
    if checkpoint_path:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model=model, map_location=device)

    model.to(device)
    metrics, predictions = evaluate_detection(
        model=model,
        data_loader=data_loader,
        device=device,
        amp=runtime_config["amp"],
        metrics_type=runtime_config["metrics"]["type"],
        log_interval=log_interval,
        stage_label=stage_label,
        epoch_index=epoch_index,
        total_epochs=total_epochs,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "metrics.json", metrics)
        eval_config = runtime_config["eval"]
        if eval_config["save_predictions"]:
            predictions_path = eval_config["predictions_path"]
            save_predictions(
                Path(predictions_path) if predictions_path else output_dir / "predictions.json",
                predictions,
            )

    return metrics, predictions


def train_one_epoch(
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    data_loader,
    device: torch.device,
    amp: bool,
    scaler: torch.amp.GradScaler,
    log_interval: int,
    grad_clip_norm: float | None,
    epoch_index: int,
    total_epochs: int,
    max_steps: int | None,
    start_iteration: int,
    total_iterations: int | None,
    on_iteration: Callable[[int, dict[str, float]], None],
    distributed: DistributedContext,
) -> dict[str, float]:
    model.train()
    loss_sums: defaultdict[str, float] = defaultdict(float)
    num_batches = 0
    start_time = time.perf_counter()
    total_steps = len(data_loader)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    for step, (images, targets) in enumerate(data_loader, start=1):
        if max_steps is not None and step > max_steps:
            break

        images = [image.to(device) for image in images]
        targets = [_move_target_to_device(target, device) for target in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            if not loss_dict:
                raise RuntimeError("Model returned an empty loss dict during training.")
            total_loss = sum(loss_dict.values())

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss encountered: {total_loss.item()}")

        if scaler.is_enabled():
            scale_before_step = scaler.get_scale()
            scaler.scale(total_loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_ran = scaler.get_scale() >= scale_before_step
        else:
            total_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer_step_ran = True

        num_batches += 1
        loss_sums["loss"] += total_loss.item()
        for name, value in loss_dict.items():
            loss_sums[name] += value.item()

        if scheduler is not None and optimizer_step_ran:
            scheduler.step()

        global_iteration = start_iteration + num_batches
        on_iteration(
            global_iteration,
            _summarize_train_metrics(
                loss_sums,
                num_batches=num_batches,
                lr=optimizer.param_groups[0]["lr"],
                duration=time.perf_counter() - start_time,
            ),
        )

        if log_interval > 0 and (step % log_interval == 0 or step == total_steps):
            elapsed = time.perf_counter() - start_time
            epoch_eta = _format_eta(elapsed / step * max(total_steps - step, 0))
            components = ", ".join(
                f"{name}={loss_sums[name] / num_batches:.4f}" for name in sorted(loss_dict)
            )
            iteration_label = ""
            if total_iterations is not None:
                iteration_label = f" iter {global_iteration}/{total_iterations}"
            print(
                f"[train] epoch {epoch_index + 1}/{total_epochs} "
                f"step {step}/{total_steps}{iteration_label} "
                f"epoch_eta={epoch_eta} "
                f"total_loss={loss_sums['loss'] / num_batches:.4f}"
                + (f" {components}" if components else "")
            )

    duration = time.perf_counter() - start_time
    lr = optimizer.param_groups[0]["lr"]
    if is_distributed(distributed):
        gathered = all_gather_object(
            {"loss_sums": dict(loss_sums), "num_batches": num_batches, "duration": duration},
            distributed,
        )
        loss_sums = defaultdict(float)
        for item in gathered:
            for key, value in item["loss_sums"].items():
                loss_sums[key] += value
        num_batches = sum(item["num_batches"] for item in gathered)
        duration = max(item["duration"] for item in gathered)
    return _summarize_train_metrics(loss_sums, num_batches=num_batches, lr=lr, duration=duration)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = config["name"].lower()
    if name == "sgd":
        return optim.SGD(
            params,
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=config["nesterov"],
        )
    if name == "adamw":
        return optim.AdamW(
            params,
            lr=config["lr"],
            betas=tuple(config["betas"]),
            weight_decay=config["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer.name: {config['name']!r}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: dict[str, Any],
) -> optim.lr_scheduler.MultiStepLR:
    if config["name"].lower() != "multistep":
        raise ValueError(f"Unsupported scheduler.name: {config['name']!r}")
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["milestones"],
        gamma=config["gamma"],
    )


def _align_scheduler_to_step(scheduler: optim.lr_scheduler.MultiStepLR, *, step: int) -> None:
    step = max(int(step), 0)
    decay_count = sum(
        int(count) for milestone, count in scheduler.milestones.items() if int(milestone) <= step
    )
    factor = float(scheduler.gamma) ** decay_count
    base_lrs = [float(base_lr) for base_lr in scheduler.base_lrs]
    lrs = [base_lr * factor for base_lr in base_lrs]
    for group, base_lr, lr in zip(scheduler.optimizer.param_groups, base_lrs, lrs):
        group["initial_lr"] = base_lr
        group["lr"] = lr
    scheduler.last_epoch = step
    scheduler._last_lr = list(lrs)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    result = unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing_keys = [key for key in result.missing_keys if key != _MISSBANK_STATE_KEY]
    unexpected_keys = [key for key in result.unexpected_keys if key != _MISSBANK_STATE_KEY]
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"missing_keys={missing_keys}")
        if unexpected_keys:
            details.append(f"unexpected_keys={unexpected_keys}")
        raise RuntimeError(
            f"Checkpoint state_dict is incompatible with the current model: {'; '.join(details)}"
        )

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    iteration: int,
    best_metric: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "iteration": int(iteration),
        "best_metric": best_metric,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f"{target.stem}-", suffix=".tmp")
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def persist_run_metadata(
    output_dir: Path,
    arch: str,
    model_config: dict[str, Any],
    model_config_path: str | Path,
    runtime_config: dict[str, Any],
    runtime_config_path: str | Path,
    module_configs: dict[str, Any] | None = None,
) -> None:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml_file(metadata_dir / "model.yaml", model_config)
    dump_yaml_file(
        metadata_dir / f"{runtime_config['_mode']}.yaml",
        {key: value for key, value in runtime_config.items() if not key.startswith("_")},
    )
    if module_configs is not None:
        dump_yaml_file(metadata_dir / "modules.yaml", module_configs)

    metadata = {
        "arch": arch,
        "model_config_path": str(Path(model_config_path).resolve()),
        "runtime_config_path": str(Path(runtime_config_path).resolve()),
    }
    module_config_paths = runtime_config.get("_module_config_paths")
    if isinstance(module_config_paths, Mapping):
        metadata["module_config_paths"] = {
            str(name): str(Path(path).expanduser().resolve())
            for name, path in module_config_paths.items()
        }
    _write_json(metadata_dir / "run.json", metadata)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_metrics(
    record: Mapping[str, Any],
    primary_metric: str,
    hard_replay: Mapping[str, Any] | None = None,
) -> str:
    parts = []
    if "loss" in record["train"]:
        parts.append(f"train_loss={record['train']['loss']:.4f}")
    if "val" in record and record["val"].get(primary_metric) is not None:
        parts.append(f"val_{primary_metric}={record['val'][primary_metric]:.4f}")
    if hard_replay is not None and hard_replay.get("enabled", False):
        parts.append(f"replay_images={int(hard_replay.get('replay_num_images', 0))}")
        parts.append(f"replay_ratio={float(hard_replay.get('replay_ratio_effective', 0.0)):.3f}")
    return " ".join(parts)


def _summarize_train_metrics(
    loss_sums: Mapping[str, float],
    *,
    num_batches: int,
    lr: float,
    duration: float,
) -> dict[str, float]:
    metrics = {key: value / max(num_batches, 1) for key, value in loss_sums.items()}
    metrics["lr"] = float(lr)
    metrics["epoch_time_sec"] = float(duration)
    return metrics


def _max_distributed_float(value: float, distributed: DistributedContext) -> float:
    return max(all_gather_object(float(value), distributed))


def _move_target_to_device(target: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def _set_data_loader_epoch(data_loader, epoch: int) -> None:
    for sampler in (data_loader.sampler, data_loader.batch_sampler):
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


def _refresh_hard_replay(data_loader, model: torch.nn.Module, epoch: int) -> None:
    controller = getattr(data_loader, "hard_replay", None)
    if controller is not None:
        controller.refresh(missbank=_get_missbank(model), epoch=epoch)


def _get_hard_replay_summary(data_loader) -> dict[str, Any] | None:
    controller = getattr(data_loader, "hard_replay", None)
    return None if controller is None else dict(controller.summary())


def _get_missbank(model: torch.nn.Module):
    return getattr(unwrap_model(model), "missbank", None)


def _set_missbank_epoch(model: torch.nn.Module, epoch: int) -> None:
    missbank = _get_missbank(model)
    if missbank is not None:
        missbank.start_epoch(epoch)


def _missbank_mining_due(missbank: Any, epoch: int) -> bool:
    if missbank is None:
        return False
    mining = missbank.config.mining
    return epoch >= mining.start_epoch and (epoch - mining.start_epoch) % mining.interval_epoch == 0


def _run_missbank_offline_mining(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool,
    epoch: int,
    total_epochs: int,
    log_interval: int,
    distributed: DistributedContext,
) -> float | None:
    missbank = _get_missbank(model)
    if not _missbank_mining_due(missbank, epoch):
        return None

    was_training = model.training
    controller = getattr(data_loader, "hard_replay", None)
    previous_base_only = None
    if controller is not None:
        previous_base_only = bool(controller.batch_sampler.base_only)
        controller.set_base_only(True)
    model.eval()
    start_time = time.perf_counter()
    total_steps = len(data_loader)
    try:
        for step, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [_move_target_to_device(target, device) for target in targets]
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"),
            ):
                detections = model(images)
            missbank.update(targets=targets, detections=detections, epoch=epoch, step=step)

            if log_interval > 0 and (step % log_interval == 0 or step == total_steps):
                elapsed = time.perf_counter() - start_time
                epoch_eta = _format_eta(elapsed / step * max(total_steps - step, 0))
                print(
                    f"[remiss-mining] epoch {epoch}/{total_epochs} "
                    f"step {step}/{total_steps} epoch_eta={epoch_eta}"
                )
    finally:
        if controller is not None:
            controller.set_base_only(previous_base_only)
        if was_training:
            model.train()
    duration = time.perf_counter() - start_time
    barrier(distributed)
    return _max_distributed_float(duration, distributed)


def _collect_missbank_epoch_snapshot(
    model: torch.nn.Module,
    epoch: int,
    distributed: DistributedContext,
) -> dict[str, Any] | None:
    missbank = _get_missbank(model)
    snapshot = (
        missbank.epoch_snapshot(epoch=epoch) if _missbank_mining_due(missbank, epoch) else None
    )
    return merge_missbank_epoch_snapshots(all_gather_object(snapshot, distributed))


def _build_missbank_epoch_metrics(
    snapshot: Mapping[str, Any],
    *,
    previous_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stability = compute_missbank_stability_metrics(snapshot, previous_snapshot=previous_snapshot)
    num_seen_gts = int(snapshot["num_seen_gts"])
    num_missed_gts = int(stability["num_missed_gts"])
    num_target_gts = int(stability["num_target_gts"])
    num_images_seen = int(snapshot["num_images_seen"])
    num_images_with_miss = int(snapshot["num_images_with_miss"])
    num_images_with_target = int(snapshot["num_images_with_target"])
    return {
        "epoch": int(stability["epoch"]),
        "mining_type": snapshot["mining_type"],
        "num_seen_gts": num_seen_gts,
        "num_missed_gts": num_missed_gts,
        "num_target_gts": num_target_gts,
        "num_images_seen": num_images_seen,
        "num_images_with_miss": num_images_with_miss,
        "num_images_with_target": num_images_with_target,
        "missed_gt_ratio": _ratio(num_missed_gts, num_seen_gts),
        "target_gt_ratio": _ratio(num_target_gts, num_seen_gts),
        "image_miss_ratio": _ratio(num_images_with_miss, num_images_seen),
        "image_target_ratio": _ratio(num_images_with_target, num_images_seen),
        "mean_miss_count": float(snapshot["mean_miss_count"]),
        "max_miss_count": int(snapshot["max_miss_count"]),
        "mean_gt_miss_rate": float(snapshot["mean_gt_miss_rate"]),
        "miss_gt_jaccard_stability": stability["miss_gt_jaccard_stability"],
        "miss_gt_churn_rate": stability["miss_gt_churn_rate"],
        "new_miss_rate": stability["new_miss_rate"],
        "persistent_miss_ratio": stability["persistent_miss_ratio"],
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator > 0 else None


def _write_hard_replay_outputs(
    output_dir: Path,
    history: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> None:
    hard_replay_dir = output_dir / "hard-replay"
    _write_json(hard_replay_dir / "hard_replay_epoch.json", history)
    _write_csv(hard_replay_dir / "hard_replay_epoch.csv", history)
    _write_json(hard_replay_dir / "hard_replay_state.json", {"snapshot": snapshot})


def _write_missbank_outputs(output_dir: Path, history: list[dict[str, Any]]) -> None:
    _write_json(output_dir / "missbank" / "missbank_epoch.json", history)
    _write_csv(output_dir / "missbank" / "missbank_epoch.csv", history)


def _write_history_outputs(output_dir: Path, history: list[dict[str, Any]]) -> None:
    iteration_history = [record for record in history if record["record_type"] == "iteration_eval"]
    epoch_history = [record for record in history if record["record_type"] != "iteration_eval"]
    _write_json(output_dir / "history.json", history)
    _write_csv(output_dir / "results.csv", history)
    _write_json(output_dir / "history_iteration.json", iteration_history)
    _write_csv(output_dir / "results_iteration.csv", iteration_history)
    _write_json(output_dir / "history_epoch.json", epoch_history)
    _write_csv(output_dir / "results_epoch.csv", epoch_history)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = [_flatten_record(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    leading = [key for key in ("epoch", "iteration") if key in keys]
    fieldnames = leading + [key for key in keys if key not in leading]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _flatten_record(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        flat_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_record(value, flat_key))
        else:
            flattened[flat_key] = value
    return flattened


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_eta(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
