from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from world.rendering import plot_observation


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    }
)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _select_indices(length: int, max_frames: int = 4) -> list[int]:
    if length <= 1:
        return [0]
    return sorted(set(np.linspace(0, length - 1, num=min(max_frames, length), dtype=int).tolist()))


def _actions_to_string(actions: Iterable[int]) -> str:
    action_names = {0: "U", 1: "D", 2: "L", 3: "R"}
    return " ".join(action_names.get(int(action), "?") for action in actions)


def plot_loss_curves(history: dict, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = np.arange(1, len(history["train_total_loss"]) + 1)

    axes[0].plot(epochs, history["train_total_loss"], color="#1f77b4", linewidth=2.0, label="Train")
    axes[0].plot(epochs, history["val_total_loss"], color="#d62728", linewidth=2.0, label="Validation")
    axes[0].set_title("Optimization objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    component_specs = [
        ("reconstruction", "#1f77b4"),
        ("rollout", "#ff7f0e"),
        ("latent", "#2ca02c"),
        ("armed", "#9467bd"),
        ("beacon", "#d62728"),
    ]
    for component, color in component_specs:
        train_key = f"train_{component}_loss"
        val_key = f"val_{component}_loss"
        if train_key in history:
            axes[1].plot(epochs, history[train_key], color=color, linewidth=2.0, label=f"Train {component}")
        if val_key in history:
            axes[1].plot(epochs, history[val_key], color=color, linestyle="--", linewidth=1.6, label=f"Val {component}")
    axes[1].set_title("Loss decomposition")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False, ncol=2)
    _style_axis(axes[1])

    fig.suptitle("Training dynamics", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_frames(
    true_observations: np.ndarray,
    predicted_observations: np.ndarray,
    grid_size: int,
    output_path: str | Path,
    title: str,
    subtitle: str | None = None,
) -> None:
    horizon = min(len(true_observations), len(predicted_observations))
    step_indices = _select_indices(horizon, max_frames=4)
    fig, axes = plt.subplots(2, len(step_indices), figsize=(3.1 * len(step_indices), 5.8))
    if len(step_indices) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, step_index in enumerate(step_indices):
        plot_observation(axes[0, column], true_observations[step_index], grid_size, title=f"True, t={step_index}")
        plot_observation(axes[1, column], predicted_observations[step_index], grid_size, title=f"Model, t={step_index}")

    axes[0, 0].set_ylabel("Ground truth", fontsize=11)
    axes[1, 0].set_ylabel("Open-loop model", fontsize=11)
    fig.suptitle(title, y=1.02)
    if subtitle:
        fig.text(0.5, 0.99, subtitle, ha="center", va="top", fontsize=10, color="#444444")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_counterfactual(
    scenario_outputs: dict,
    grid_size: int,
    output_path: str | Path,
    title: str = "Counterfactual rollout from a shared starting state",
    caption: str | None = None,
) -> None:
    num_scenarios = len(scenario_outputs)
    row_count = num_scenarios * 2
    num_columns = 4
    fig, axes = plt.subplots(row_count, num_columns, figsize=(3.1 * num_columns, 2.8 * row_count))
    axes = np.asarray(axes).reshape(row_count, num_columns)

    summary_lines = []
    for scenario_index, (name, outputs) in enumerate(scenario_outputs.items()):
        true_sequence = outputs["true"]
        pred_sequence = outputs["pred"]
        true_indices = _select_indices(len(true_sequence), max_frames=num_columns)
        pred_indices = _select_indices(len(pred_sequence), max_frames=num_columns)

        for column in range(num_columns):
            true_step = true_indices[min(column, len(true_indices) - 1)]
            pred_step = pred_indices[min(column, len(pred_indices) - 1)]
            plot_observation(
                axes[scenario_index * 2, column],
                true_sequence[true_step],
                grid_size,
                title=f"True, t={true_step}",
            )
            plot_observation(
                axes[scenario_index * 2 + 1, column],
                pred_sequence[pred_step],
                grid_size,
                title=f"Model, t={pred_step}",
            )

        axes[scenario_index * 2, 0].set_ylabel(f"{name}\nGround truth", fontsize=11)
        axes[scenario_index * 2 + 1, 0].set_ylabel(f"{name}\nModel", fontsize=11)

        summary = outputs.get("summary", {})
        summary_lines.append(
            (
                f"{name}: actions={_actions_to_string(outputs.get('actions', []))} | "
                f"final beacon true={int(summary.get('true_final_beacon_lit', 0.0))}, "
                f"model={int(summary.get('pred_final_beacon_lit', 0.0))}"
            )
        )

    fig.suptitle(title, y=1.02)
    fig.text(
        0.5,
        0.985,
        "\n".join(summary_lines),
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )
    if caption:
        fig.text(
            0.5,
            0.01,
            caption,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latent_pca(
    latents: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
    label_name: str,
    discrete: bool = True,
) -> None:
    centered = latents - latents.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if discrete:
        scatter = ax.scatter(
            components[:, 0],
            components[:, 1],
            c=labels,
            cmap="coolwarm",
            s=18,
            alpha=0.78,
            edgecolors="none",
        )
        colorbar = fig.colorbar(scatter, ax=ax, ticks=[0.0, 1.0])
        colorbar.set_label(label_name)
    else:
        scatter = ax.scatter(
            components[:, 0],
            components[:, 1],
            c=labels,
            cmap="viridis",
            s=18,
            alpha=0.78,
            edgecolors="none",
        )
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label(label_name)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latent_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
    label_name: str,
    discrete: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if discrete:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="coolwarm", s=18, alpha=0.78, edgecolors="none")
        colorbar = fig.colorbar(scatter, ax=ax, ticks=[0.0, 1.0])
        colorbar.set_label(label_name)
    else:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="viridis", s=18, alpha=0.78, edgecolors="none")
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label(label_name)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_noise_robustness(clean_mse: float, noisy_mse: float, output_path: str | Path) -> None:
    degradation = 100.0 * max(noisy_mse - clean_mse, 0.0) / max(clean_mse, 1e-8)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    colors = ["#4e79a7", "#e15759"]
    values = [clean_mse, noisy_mse]
    ax.bar(["Clean observations", "Noisy observations"], values, color=colors, width=0.62)
    ax.set_ylabel("One-step prediction MSE")
    ax.set_title("Noise robustness summary")
    ax.set_ylim(0.0, max(values) * 1.25 if max(values) > 0 else 1.0)
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.03, f"{value:.4f}", ha="center", va="bottom")
    ax.text(
        0.5,
        0.93,
        f"Relative degradation: {degradation:.1f}%",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_error_curve(
    stepwise_errors: Iterable[float],
    cumulative_errors: Iterable[float],
    output_path: str | Path,
) -> None:
    stepwise_errors = np.asarray(list(stepwise_errors), dtype=np.float32)
    cumulative_errors = np.asarray(list(cumulative_errors), dtype=np.float32)
    horizons = np.arange(1, len(stepwise_errors) + 1)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(horizons, stepwise_errors, marker="o", linewidth=2.0, color="#d62728", label="Stepwise MSE")
    ax.plot(horizons, cumulative_errors, marker="s", linewidth=2.0, color="#1f77b4", label="Cumulative MSE")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Error")
    ax.set_title("Open-loop rollout error vs. horizon")
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_failure_cases(
    failure_cases: list[dict],
    grid_size: int,
    output_path: str | Path,
    title: str,
) -> None:
    if not failure_cases:
        return

    num_cases = len(failure_cases)
    step_indices = _select_indices(len(failure_cases[0]["true"]), max_frames=4)
    fig, axes = plt.subplots(num_cases * 2, len(step_indices), figsize=(3.1 * len(step_indices), 2.8 * num_cases * 2))
    axes = np.asarray(axes).reshape(num_cases * 2, len(step_indices))

    for case_index, case in enumerate(failure_cases):
        for column, step_index in enumerate(step_indices):
            true_step = min(step_index, len(case["true"]) - 1)
            pred_step = min(step_index, len(case["pred"]) - 1)
            plot_observation(axes[case_index * 2, column], case["true"][true_step], grid_size, title=f"True, t={true_step}")
            plot_observation(axes[case_index * 2 + 1, column], case["pred"][pred_step], grid_size, title=f"Model, t={pred_step}")
        axes[case_index * 2, 0].set_ylabel(case["label"], fontsize=10)
        axes[case_index * 2 + 1, 0].set_ylabel(f"Error={case['error']:.4f}", fontsize=10)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
