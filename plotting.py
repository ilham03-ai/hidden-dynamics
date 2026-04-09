from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from world.rendering import plot_observation


def plot_loss_curves(history: dict, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["train_total_loss"]) + 1)

    axes[0].plot(epochs, history["train_total_loss"], label="train")
    axes[0].plot(epochs, history["val_total_loss"], label="val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_reconstruction_loss"], label="train recon")
    axes[1].plot(epochs, history["val_reconstruction_loss"], label="val recon")
    axes[1].plot(epochs, history["train_rollout_loss"], label="train rollout")
    axes[1].plot(epochs, history["val_rollout_loss"], label="val rollout")
    if "train_beacon_loss" in history:
        axes[1].plot(epochs, history["train_beacon_loss"], label="train beacon")
        axes[1].plot(epochs, history["val_beacon_loss"], label="val beacon")
    axes[1].plot(epochs, history["train_latent_loss"], label="train latent")
    axes[1].plot(epochs, history["val_latent_loss"], label="val latent")
    axes[1].plot(epochs, history["train_uncertainty_loss"], label="train uncertainty")
    axes[1].plot(epochs, history["val_uncertainty_loss"], label="val uncertainty")
    axes[1].set_title("Loss Components")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_frames(
    true_sequence: np.ndarray,
    predicted_sequence: np.ndarray,
    grid_size: int,
    output_path: str | Path,
    title: str,
) -> None:
    step_indices = sorted(set(np.linspace(0, len(true_sequence) - 1, num=min(4, len(true_sequence)), dtype=int).tolist()))
    fig, axes = plt.subplots(2, len(step_indices), figsize=(3.2 * len(step_indices), 6))
    axes = np.asarray(axes).reshape(2, len(step_indices))
    for column, step_index in enumerate(step_indices):
        plot_observation(axes[0, column], true_sequence[step_index], grid_size, title=f"True t={step_index}")
        plot_observation(axes[1, column], predicted_sequence[step_index], grid_size, title=f"Pred t={step_index}")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_counterfactual(scenario_outputs: dict, grid_size: int, output_path: str | Path) -> None:
    num_columns = 4
    row_count = len(scenario_outputs) * 2
    fig, axes = plt.subplots(row_count, num_columns, figsize=(3.2 * num_columns, 2.8 * row_count))
    axes = np.asarray(axes).reshape(row_count, num_columns)

    for scenario_index, (name, outputs) in enumerate(scenario_outputs.items()):
        step_indices = sorted(set(np.linspace(0, len(outputs["true"]) - 1, num=min(num_columns, len(outputs["true"])), dtype=int).tolist()))
        for column, step_index in enumerate(step_indices):
            plot_observation(axes[scenario_index * 2, column], outputs["true"][step_index], grid_size, title=f"{name} true t={step_index}")
            plot_observation(axes[scenario_index * 2 + 1, column], outputs["pred"][step_index], grid_size, title=f"{name} pred t={step_index}")
        for column in range(len(step_indices), num_columns):
            axes[scenario_index * 2, column].axis("off")
            axes[scenario_index * 2 + 1, column].axis("off")

    fig.suptitle("Counterfactual rollouts", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_latent_projection(points: np.ndarray, labels: np.ndarray, output_path: str | Path, title: str, label_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="coolwarm", s=14, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(label_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_curve(mean_uncertainty: np.ndarray, mean_error: np.ndarray, output_path: str | Path, title: str) -> None:
    steps = np.arange(1, len(mean_uncertainty) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, mean_uncertainty, label="predicted uncertainty")
    ax.plot(steps, mean_error, label="actual squared error")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_noise_bar(clean_value: float, noisy_value: float, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["clean", "noisy"], [clean_value, noisy_value], color=["#4e79a7", "#e15759"])
    ax.set_ylabel("One-step MSE")
    ax.set_title("Noise robustness")
    for index, value in enumerate([clean_value, noisy_value]):
        ax.text(index, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_exploration_comparison(summary: dict, output_path: str | Path) -> None:
    labels = list(summary.keys())
    informative_interactions = [summary[label]["mean_informative_interactions"] for label in labels]
    uncertainty = [summary[label]["mean_predicted_uncertainty"] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, informative_interactions, color=["#4e79a7", "#f28e2b"])
    axes[0].set_title("Targeted interactions")
    axes[0].set_ylabel("Pad / terminal interactions")

    axes[1].bar(labels, uncertainty, color=["#4e79a7", "#f28e2b"])
    axes[1].set_title("Predicted uncertainty")
    axes[1].set_ylabel("Mean uncertainty")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
