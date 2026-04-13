from __future__ import annotations

import argparse

from analyze_latent import run_latent_analysis
from config import CHECKPOINT_DIR
from evaluate import run_evaluation
from generate_data import run_generation
from rollout import run_counterfactual_demo
from train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hidden Dynamics research pipeline.")
    parser.add_argument(
        "stage",
        choices=["generate", "train", "evaluate", "rollout", "analyze", "all"],
        nargs="?",
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str((CHECKPOINT_DIR / "best_model.pt").resolve()),
        help="Path to a trained checkpoint for evaluation, rollout, or latent analysis.",
    )
    args = parser.parse_args()

    if args.stage in {"generate", "all"}:
        print(run_generation())
    if args.stage in {"train", "all"}:
        print(run_training())
    if args.stage in {"evaluate", "all"}:
        print(run_evaluation(args.checkpoint))
    if args.stage == "rollout":
        print(run_counterfactual_demo(args.checkpoint))
    if args.stage == "analyze":
        print(run_latent_analysis(args.checkpoint))


if __name__ == "__main__":
    main()
