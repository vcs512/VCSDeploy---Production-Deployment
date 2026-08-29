"""Training service entrypoint for image classification."""

import argparse

from src.core.dataset import load_classification_dataset
from src.core.model import build_model
from src.core.trainer import run_training
from src.schemas.training import TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line namespace.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a local image classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.json",
        help="Path to the training JSON configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the configuration and run the training routine."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = TrainingConfig.model_validate_json(handle.read())

    train_loader, val_loader, num_labels, id2label, label2id = (
        load_classification_dataset(config)
    )
    model = build_model(
        config=config,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    run_training(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == "__main__":
    main()
