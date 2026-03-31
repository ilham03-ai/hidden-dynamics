from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURE_DIR = RESULTS_DIR / "figures"


@dataclass
class EnvironmentConfig:
    grid_size: int = 6
    max_steps: int = 18
    action_noise: float = 0.0
    observation_noise: float = 0.0


@dataclass
class DataConfig:
    seed: int = 7
    train_episodes: int = 512
    val_episodes: int = 128
    test_episodes: int = 128
    horizon: int = 18
    noisy_eval_std: float = 0.04


@dataclass
class ModelConfig:
    obs_dim: int = 11
    num_actions: int = 4
    latent_dim: int = 64
    hidden_dim: int = 128


@dataclass
class TrainingConfig:
    seed: int = 7
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    latent_loss_weight: float = 0.25
    rollout_loss_weight: float = 1.0
    armed_loss_weight: float = 2.0
    beacon_loss_weight: float = 4.0
    grad_clip_norm: float = 5.0
    device: str = "cpu"


@dataclass
class EvaluationConfig:
    seed: int = 17
    rollout_horizon: int = 10
    latent_samples: int = 1000
    noise_std: float = 0.04
    counterfactual_benchmark_size: int = 24
    failure_case_count: int = 3
    latent_probe_epochs: int = 200
    latent_probe_learning_rate: float = 0.05
    enable_tsne: bool = False


def ensure_directories() -> None:
    for path in [DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR, FIGURE_DIR, PROJECT_ROOT / "notebooks"]:
        path.mkdir(parents=True, exist_ok=True)


def get_default_config() -> dict:
    ensure_directories()
    return {
        "environment": EnvironmentConfig(),
        "data": DataConfig(),
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "evaluation": EvaluationConfig(),
    }


def config_to_dict(config: dict) -> dict:
    return {name: asdict(section) for name, section in config.items()}
