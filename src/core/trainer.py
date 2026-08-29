"""Training and evaluation loop with MLflow tracking."""

import os
import random
import shutil

import mlflow
import numpy as np
import torch
from torch.amp import GradScaler, autocast

from src.core.metrics import compute_metrics
from src.schemas.training import TrainingConfig


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility.

    Args:
        seed: The random seed to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_best_model(config: TrainingConfig, model: torch.nn.Module) -> None:
    """Persist the best model weights and processor config.

    Args:
        config: Resolved training configuration.
        model: The model to serialize.
    """
    model_dir = config.model_dir()
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    preprocessor_src = os.path.join(
        config.model_local_path, "preprocessor_config.json"
    )
    if os.path.exists(preprocessor_src):
        shutil.copy(preprocessor_src, os.path.join(model_dir, "preprocessor_config.json"))


def run_training(
    config: TrainingConfig,
    model: torch.nn.Module,
    train_loader,
    val_loader,
) -> None:
    """Run the fine-tuning routine and track results with MLflow.

    Args:
        config: Resolved training configuration.
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    """
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    amp_dtype = torch.float16

    model.to(device)
    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    best_f1 = -1.0
    best_metrics: dict = {}

    with mlflow.start_run() as run:
        mlflow.log_params(config.model_dump())
        for epoch in range(config.num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            steps = 0
            for step, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                running_loss += outputs.loss.item()
                steps += 1
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            train_loss = running_loss / max(steps, 1)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        outputs = model(pixel_values=pixel_values)
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

            predictions = np.concatenate(all_preds)
            ground_truth = np.concatenate(all_labels)
            metrics = compute_metrics(predictions=predictions, labels=ground_truth)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metrics(
                {f"val_{key}": value for key, value in metrics.items()},
                step=epoch,
            )
            print(
                f"epoch {epoch + 1}/{config.num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_f1={metrics['f1']:.4f} | "
                f"val_acc={metrics['accuracy']:.4f}"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_metrics = dict(metrics)
                _save_best_model(config=config, model=model)

        mlflow.log_metrics(
            {f"best_{key}": value for key, value in best_metrics.items()}
        )
        mlflow.log_artifacts(config.model_dir())
        print(f"MLflow run: {run.info.run_id} | best_f1={best_f1:.4f}")
