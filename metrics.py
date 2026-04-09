from __future__ import annotations

import numpy as np


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def per_step_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((prediction - target) ** 2, axis=(0, 2))


def uncertainty_error_correlation(predicted_uncertainty: np.ndarray, actual_error: np.ndarray) -> float:
    flat_uncertainty = predicted_uncertainty.reshape(-1)
    flat_error = actual_error.reshape(-1)
    if np.allclose(flat_uncertainty.std(), 0.0) or np.allclose(flat_error.std(), 0.0):
        return 0.0
    return float(np.corrcoef(flat_uncertainty, flat_error)[0, 1])
