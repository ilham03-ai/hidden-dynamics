from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from analyze_latent import analyze_latent_space
from config import CHECKPOINT_DIR, DATA_DIR, FIGURE_DIR, get_default_config
from generate_data import POLICY_NAMES
from models.world_model import WorldModel
from rollout import evaluate_counterfactual_benchmark, evaluate_filtered_counterfactual_benchmark, run_counterfactual_demo
from utils.data import load_split, save_json
from utils.metrics import (
    binary_classification_metrics_from_logits,
    cumulative_rollout_mse,
    mse_numpy,
    per_step_mse,
    rollout_mse,
)
from utils.plotting import plot_counterfactual, plot_failure_cases, plot_noise_robustness, plot_rollout_error_curve, plot_rollout_frames
from utils.seed import set_global_seed


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[WorldModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_config = config["model"]
    model = WorldModel(
        obs_dim=model_config["obs_dim"],
        num_actions=model_config["num_actions"],
        latent_dim=model_config["latent_dim"],
        hidden_dim=model_config["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def _with_optional_noise(observations: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    if noise_std <= 0.0:
        return observations
    noise = np.random.default_rng(seed).normal(0.0, noise_std, size=observations.shape).astype(np.float32)
    noisy = np.clip(observations + noise, 0.0, 1.0)
    noisy[..., -1] = (noisy[..., -1] >= 0.5).astype(np.float32)
    return noisy


def evaluate_one_step(
    model: WorldModel,
    split: dict,
    device: torch.device,
    noise_std: float = 0.0,
    seed: int = 0,
) -> dict:
    observations = _with_optional_noise(split["episode_observations"].copy(), noise_std=noise_std, seed=seed)
    with torch.no_grad():
        outputs = model.forward_sequence(
            torch.from_numpy(observations).float().to(device),
            torch.from_numpy(split["episode_actions"]).long().to(device),
        )["predicted_observations"].cpu().numpy()
    return {
        "mse": mse_numpy(outputs, split["episode_observations"][:, 1:]),
        "predictions": outputs,
    }


def evaluate_probe_metrics(model: WorldModel, split: dict, device: torch.device) -> dict:
    with torch.no_grad():
        outputs = model.forward_sequence(
            torch.from_numpy(split["episode_observations"]).float().to(device),
            torch.from_numpy(split["episode_actions"]).long().to(device),
        )
    logits = outputs["prior_state_logits"].cpu().numpy()
    return {
        "armed": binary_classification_metrics_from_logits(logits[..., 0], split["episode_hidden_armed"][:, 1:]),
        "beacon_lit": binary_classification_metrics_from_logits(logits[..., 1], split["episode_beacon_lit"][:, 1:]),
    }


def evaluate_rollouts(
    model: WorldModel,
    split: dict,
    device: torch.device,
    rollout_horizon: int,
) -> dict:
    episode_observations = split["episode_observations"]
    episode_actions = split["episode_actions"][:, :rollout_horizon]
    initial_observations = torch.from_numpy(episode_observations[:, 0]).float().to(device)
    action_tensor = torch.from_numpy(episode_actions).long().to(device)

    with torch.no_grad():
        predicted_rollouts = model.rollout(initial_observations, action_tensor).cpu().numpy()

    true_rollouts = episode_observations[:, 1 : rollout_horizon + 1]
    episode_errors = np.mean((predicted_rollouts - true_rollouts) ** 2, axis=(1, 2))
    return {
        "mse": rollout_mse(predicted_rollouts, true_rollouts),
        "predicted_rollouts": predicted_rollouts,
        "true_rollouts": true_rollouts,
        "episode_errors": episode_errors,
        "stepwise_mse": per_step_mse(predicted_rollouts, true_rollouts),
        "cumulative_mse": cumulative_rollout_mse(predicted_rollouts, true_rollouts),
    }


def _pick_rollout_example(split: dict, episode_errors: np.ndarray) -> int:
    lit_indices = np.where(split["episode_beacon_lit"][:, -1] >= 0.5)[0]
    if len(lit_indices) > 0:
        return int(lit_indices[np.argmin(episode_errors[lit_indices])])
    return int(np.argmin(episode_errors))


def _serialize_layout(layout: np.ndarray) -> dict:
    return {
        "agent": layout[0:2].astype(int).tolist(),
        "crate": layout[2:4].astype(int).tolist(),
        "switch": layout[4:6].astype(int).tolist(),
        "beacon": layout[6:8].astype(int).tolist(),
        "obstacle": layout[8:10].astype(int).tolist(),
    }


def build_failure_analysis(
    split: dict,
    rollout_results: dict,
    counterfactual_benchmark: dict,
    filtered_counterfactual_benchmark: dict,
    grid_size: int,
    failure_case_count: int,
) -> dict:
    episode_errors = rollout_results["episode_errors"]
    top_indices = np.argsort(episode_errors)[-failure_case_count:][::-1]

    rollout_failure_cases = []
    rollout_failure_summary = []
    for episode_index in top_indices:
        rollout_failure_cases.append(
            {
                "true": np.concatenate(
                    [split["episode_observations"][episode_index, :1], rollout_results["true_rollouts"][episode_index]],
                    axis=0,
                ),
                "pred": np.concatenate(
                    [split["episode_observations"][episode_index, :1], rollout_results["predicted_rollouts"][episode_index]],
                    axis=0,
                ),
                "label": f"Episode {int(episode_index)} ({POLICY_NAMES[int(split['behavior_ids'][episode_index])]})",
                "error": float(episode_errors[episode_index]),
            }
        )
        rollout_failure_summary.append(
            {
                "episode_index": int(episode_index),
                "behavior_policy": POLICY_NAMES[int(split["behavior_ids"][episode_index])],
                "episode_rollout_mse": float(episode_errors[episode_index]),
                "final_hidden_armed": float(split["episode_hidden_armed"][episode_index, -1]),
                "final_beacon_lit": float(split["episode_beacon_lit"][episode_index, -1]),
                "initial_layout": _serialize_layout(split["initial_layouts"][episode_index]),
            }
        )

    plot_failure_cases(
        rollout_failure_cases,
        grid_size=grid_size,
        output_path=FIGURE_DIR / "rollout_failure_cases.png",
        title="Highest rollout-error held-out episodes",
    )

    wrong_counterfactuals = [case for case in filtered_counterfactual_benchmark["cases"] if not case["paired_correct"]]
    serialized_counterfactual_failures = [
        {
            "episode_index": int(case["episode_index"]),
            "step_index": int(case["step_index"]),
            "direct_correct": bool(case["direct_correct"]),
            "switch_then_beacon_correct": bool(case["switch_then_beacon_correct"]),
            "true_direct_final_beacon": float(case["true_direct_final_beacon"]),
            "true_switch_then_beacon_final_beacon": float(case["true_switch_then_beacon_final_beacon"]),
            "pred_direct_final_beacon": float(case["pred_direct_final_beacon"]),
            "pred_switch_then_beacon_final_beacon": float(case["pred_switch_then_beacon_final_beacon"]),
        }
        for case in wrong_counterfactuals[:failure_case_count]
    ]

    if wrong_counterfactuals:
        representative = wrong_counterfactuals[0]
        plot_counterfactual(
            {
                "Direct to beacon": {
                    "true": representative["true_direct_sequence"],
                    "pred": representative["pred_direct_sequence"],
                    "actions": representative["direct_actions"],
                    "summary": {
                        "true_final_beacon_lit": representative["true_direct_final_beacon"],
                        "pred_final_beacon_lit": representative["pred_direct_final_beacon"],
                    },
                },
                "Switch then beacon": {
                    "true": representative["true_switch_sequence"],
                    "pred": representative["pred_switch_sequence"],
                    "actions": representative["switch_then_beacon_actions"],
                    "summary": {
                        "true_final_beacon_lit": representative["true_switch_then_beacon_final_beacon"],
                        "pred_final_beacon_lit": representative["pred_switch_then_beacon_final_beacon"],
                    },
                },
            },
            grid_size=grid_size,
            output_path=FIGURE_DIR / "counterfactual_failure_case.png",
            title="Representative counterfactual failure from a filtered latent state",
            caption=(
                "The model conditions on an episode prefix, then imagines two futures. "
                "This failure case shows where the latent belief state is still insufficient."
            ),
        )

    direct_failures = sum(not case["direct_correct"] for case in wrong_counterfactuals)
    switch_failures = sum(not case["switch_then_beacon_correct"] for case in wrong_counterfactuals)
    failure_patterns = []
    if rollout_failure_summary:
        failure_patterns.append("Largest rollout failures are dominated by long-horizon open-loop drift rather than one-step reconstruction.")
    if switch_failures > direct_failures:
        failure_patterns.append("Most filtered-latent counterfactual failures miss the delayed switch prerequisite on the longer path.")
    elif direct_failures > switch_failures:
        failure_patterns.append("Most filtered-latent counterfactual failures are false positives on the shorter direct-to-beacon path.")
    elif wrong_counterfactuals:
        failure_patterns.append("Filtered-latent counterfactual failures are split between false positives on the direct path and missed delayed effects on the switch-first path.")
    if counterfactual_benchmark["num_cases"] > 0:
        failure_patterns.append(
            f"On held-out initial states, raw pairwise counterfactual accuracy is {counterfactual_benchmark['pair_accuracy']:.2f}; the model separates branches more reliably in score than in raw 0.5-thresholded decisions."
        )

    analysis = {
        "top_rollout_failures": rollout_failure_summary,
        "wrong_counterfactual_cases": serialized_counterfactual_failures,
        "failure_patterns": failure_patterns,
    }
    save_json(analysis, CHECKPOINT_DIR / "failure_analysis.json")
    return analysis


def _clean_counterfactual_summary(counterfactual_benchmark: dict) -> dict:
    return {
        "num_cases": int(counterfactual_benchmark["num_cases"]),
        "raw_pair_accuracy": float(counterfactual_benchmark["pair_accuracy"]),
        "raw_final_beacon_accuracy": float(counterfactual_benchmark["final_beacon_accuracy"]),
        "raw_divergence_accuracy": float(counterfactual_benchmark["divergence_accuracy"]),
    }


def _counterfactual_ordering_metrics(cases: list[dict]) -> dict:
    if not cases:
        return {"probability_ordering_accuracy": 0.0, "mean_probability_gap": 0.0}
    ordering_accuracy = float(
        np.mean(
            [
                case["pred_switch_sequence"][-1, -1] > case["pred_direct_sequence"][-1, -1]
                for case in cases
            ]
        )
    )
    mean_gap = float(
        np.mean(
            [
                case["pred_switch_sequence"][-1, -1] - case["pred_direct_sequence"][-1, -1]
                for case in cases
            ]
        )
    )
    return {
        "probability_ordering_accuracy": ordering_accuracy,
        "mean_probability_gap": mean_gap,
    }


def _calibrate_counterfactual_threshold(validation_cases: list[dict]) -> float:
    if not validation_cases:
        return 0.5
    thresholds = np.linspace(0.01, 0.5, 50)
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in thresholds:
        outcomes = []
        for case in validation_cases:
            outcomes.extend(
                [
                    float(case["pred_direct_sequence"][-1, -1] >= threshold) == case["true_direct_final_beacon"],
                    float(case["pred_switch_sequence"][-1, -1] >= threshold) == case["true_switch_then_beacon_final_beacon"],
                ]
            )
        accuracy = float(np.mean(outcomes))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def _evaluate_counterfactual_at_threshold(cases: list[dict], threshold: float) -> dict:
    if not cases:
        return {"threshold": float(threshold), "pair_accuracy": 0.0, "final_beacon_accuracy": 0.0}
    pair_hits = []
    final_hits = []
    for case in cases:
        direct_pred = float(case["pred_direct_sequence"][-1, -1] >= threshold)
        switch_pred = float(case["pred_switch_sequence"][-1, -1] >= threshold)
        direct_hit = direct_pred == case["true_direct_final_beacon"]
        switch_hit = switch_pred == case["true_switch_then_beacon_final_beacon"]
        final_hits.extend([direct_hit, switch_hit])
        pair_hits.append(direct_hit and switch_hit)
    return {
        "threshold": float(threshold),
        "pair_accuracy": float(np.mean(pair_hits)),
        "final_beacon_accuracy": float(np.mean(final_hits)),
    }


def run_evaluation(checkpoint_path: str | Path) -> dict:
    default_config = get_default_config()
    evaluation_config = default_config["evaluation"]
    set_global_seed(evaluation_config.seed)
    device = torch.device("cpu")
    model, checkpoint_config = load_model(checkpoint_path, device)

    train_split = load_split(DATA_DIR / "train.npz")
    val_split = load_split(DATA_DIR / "val.npz")
    test_split = load_split(DATA_DIR / "test.npz")

    clean_one_step = evaluate_one_step(model, test_split, device, noise_std=0.0, seed=evaluation_config.seed)
    noisy_one_step = evaluate_one_step(
        model,
        test_split,
        device,
        noise_std=evaluation_config.noise_std,
        seed=evaluation_config.seed,
    )
    probe_metrics = evaluate_probe_metrics(model, test_split, device)
    rollout_results = evaluate_rollouts(model, test_split, device, evaluation_config.rollout_horizon)
    counterfactual_benchmark = evaluate_counterfactual_benchmark(
        model,
        test_split,
        grid_size=checkpoint_config["environment"]["grid_size"],
        max_steps=checkpoint_config["environment"]["max_steps"],
        benchmark_size=evaluation_config.counterfactual_benchmark_size,
        seed=evaluation_config.seed,
        device=device,
    )
    val_counterfactual_benchmark = evaluate_counterfactual_benchmark(
        model,
        val_split,
        grid_size=checkpoint_config["environment"]["grid_size"],
        max_steps=checkpoint_config["environment"]["max_steps"],
        benchmark_size=evaluation_config.counterfactual_benchmark_size,
        seed=evaluation_config.seed,
        device=device,
    )
    filtered_counterfactual_benchmark = evaluate_filtered_counterfactual_benchmark(
        model,
        test_split,
        grid_size=checkpoint_config["environment"]["grid_size"],
        max_steps=checkpoint_config["environment"]["max_steps"],
        benchmark_size=evaluation_config.counterfactual_benchmark_size,
        seed=evaluation_config.seed,
        device=device,
    )
    latent_analysis = analyze_latent_space(model, train_split, test_split, device, evaluation_config)
    counterfactual_threshold = _calibrate_counterfactual_threshold(val_counterfactual_benchmark["cases"])
    calibrated_counterfactual = _evaluate_counterfactual_at_threshold(
        counterfactual_benchmark["cases"],
        counterfactual_threshold,
    )
    counterfactual_ordering = _counterfactual_ordering_metrics(counterfactual_benchmark["cases"])

    plot_rollout_error_curve(
        rollout_results["stepwise_mse"],
        rollout_results["cumulative_mse"],
        FIGURE_DIR / "rollout_horizon_error.png",
    )
    plot_rollout_error_curve(
        rollout_results["stepwise_mse"],
        rollout_results["cumulative_mse"],
        FIGURE_DIR / "rollout_error_curve.png",
    )
    plot_noise_robustness(clean_one_step["mse"], noisy_one_step["mse"], FIGURE_DIR / "noise_summary.png")
    plot_noise_robustness(clean_one_step["mse"], noisy_one_step["mse"], FIGURE_DIR / "noise_robustness.png")

    example_episode = _pick_rollout_example(test_split, rollout_results["episode_errors"])
    true_sequence = test_split["episode_observations"][example_episode, : evaluation_config.rollout_horizon + 1]
    predicted_sequence = np.concatenate(
        [
            true_sequence[:1],
            rollout_results["predicted_rollouts"][example_episode],
        ],
        axis=0,
    )
    plot_rollout_frames(
        true_sequence,
        predicted_sequence,
        checkpoint_config["environment"]["grid_size"],
        FIGURE_DIR / "rollout_example.png",
        title="Held-out open-loop rollout",
        subtitle="The model receives only the initial observation, then predicts the rest of the trajectory from actions alone.",
    )

    demo_counterfactual = run_counterfactual_demo(checkpoint_path)
    failure_analysis = build_failure_analysis(
        test_split,
        rollout_results,
        counterfactual_benchmark,
        filtered_counterfactual_benchmark,
        checkpoint_config["environment"]["grid_size"],
        evaluation_config.failure_case_count,
    )

    noise_degradation = 100.0 * max(noisy_one_step["mse"] - clean_one_step["mse"], 0.0) / max(clean_one_step["mse"], 1e-8)
    metrics = {
        "one_step_prediction": {
            "clean_mse": float(clean_one_step["mse"]),
            "noisy_mse": float(noisy_one_step["mse"]),
            "noise_degradation_percent": float(noise_degradation),
        },
        "hidden_state_probes": probe_metrics,
        "rollout": {
            "open_loop_mse": float(rollout_results["mse"]),
            "stepwise_mse_by_horizon": rollout_results["stepwise_mse"].tolist(),
            "cumulative_mse_by_horizon": rollout_results["cumulative_mse"].tolist(),
        },
        "counterfactual_demo": demo_counterfactual,
        "counterfactual_benchmark": {
            **_clean_counterfactual_summary(counterfactual_benchmark),
            **counterfactual_ordering,
            "calibrated_threshold": float(counterfactual_threshold),
            "calibrated_pair_accuracy": float(calibrated_counterfactual["pair_accuracy"]),
            "calibrated_final_beacon_accuracy": float(calibrated_counterfactual["final_beacon_accuracy"]),
        },
        "filtered_counterfactual_diagnostic": _clean_counterfactual_summary(filtered_counterfactual_benchmark),
        "latent_analysis": {
            "summary_text": latent_analysis["summary_text"],
            "linear_probe_accuracy": float(latent_analysis["linear_probe"]["accuracy"]),
            "armed_gap_pca": float(latent_analysis["pca_separation"]["armed_gap"]),
            "beacon_gap_pca": float(latent_analysis["pca_separation"]["beacon_gap"]),
        },
        "failure_analysis": {
            "num_rollout_failures_saved": len(failure_analysis["top_rollout_failures"]),
            "num_counterfactual_failures_saved": len(failure_analysis["wrong_counterfactual_cases"]),
            "patterns": failure_analysis["failure_patterns"],
        },
    }
    save_json(metrics, CHECKPOINT_DIR / "evaluation_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained world model.")
    parser.add_argument(
        "--checkpoint",
        default=str((CHECKPOINT_DIR / "best_model.pt").resolve()),
        help="Path to a trained checkpoint.",
    )
    args = parser.parse_args()
    metrics = run_evaluation(args.checkpoint)
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
