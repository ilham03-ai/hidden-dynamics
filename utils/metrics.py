from __future__ import annotations

import numpy as np
import torch


def mse_numpy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def mse_torch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def rollout_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def per_step_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((prediction - target) ** 2, axis=(0, 2))


def cumulative_rollout_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(np.mean((prediction[:, : step + 1] - target[:, : step + 1]) ** 2))
            for step in range(prediction.shape[1])
        ],
        dtype=np.float32,
    )


def binary_classification_metrics_from_logits(logits: np.ndarray, targets: np.ndarray) -> dict:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(np.float32)

    accuracy = float(np.mean(predictions == targets))
    positive = targets == 1.0
    negative = ~positive
    true_positive = float(np.sum((predictions == 1.0) & positive))
    true_negative = float(np.sum((predictions == 0.0) & negative))
    false_positive = float(np.sum((predictions == 1.0) & negative))
    false_negative = float(np.sum((predictions == 0.0) & positive))

    recall_pos = true_positive / max(true_positive + false_negative, 1.0)
    recall_neg = true_negative / max(true_negative + false_positive, 1.0)
    precision = true_positive / max(true_positive + false_positive, 1.0)
    balanced_accuracy = 0.5 * (recall_pos + recall_neg)
    bce = float(
        np.mean(
            np.maximum(logits, 0.0)
            - logits * targets
            + np.log1p(np.exp(-np.abs(logits)))
        )
    )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall_pos),
        "bce": bce,
        "positive_rate": float(np.mean(targets)),
    }
