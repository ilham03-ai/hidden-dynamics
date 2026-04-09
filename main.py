from __future__ import annotations

import argparse

from analyze_latent import run_latent_analysis
from evaluate import run_evaluation
from explore import run_exploration
from generate_data import run_generation
from rollout import run_counterfactual_demo
from train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run world-model-lab-v2 end to end.")
    parser.add_argument(
        "stage",
        choices=["generate", "train", "evaluate", "rollout", "analyze", "explore", "all"],
        nargs="?",
        default="all",
    )
    args = parser.parse_args()

    if args.stage in {"generate", "all"}:
        print(run_generation())
    if args.stage in {"train", "all"}:
        print(run_training())
    if args.stage in {"evaluate", "all"}:
        print(run_evaluation("results/checkpoints/best_model.pt"))
    if args.stage in {"rollout", "all"}:
        print(run_counterfactual_demo("results/checkpoints/best_model.pt"))
    if args.stage in {"analyze", "all"}:
        print(run_latent_analysis("results/checkpoints/best_model.pt"))
    if args.stage in {"explore", "all"}:
        print(run_exploration("results/checkpoints/best_model.pt"))


if __name__ == "__main__":
    main()
