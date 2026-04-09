from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURE_DIR = RESULTS_DIR / "figures"


@dataclass
class EnvironmentConfig:
    grid_size: int = 7
    max_steps: int = 15
    charge_duration: int = 4
    observation_noise: float = 0.0
    action_noise: float = 0.0


@dataclass
class DataConfig:
    seed: int = 13
    train_episodes: int = 400
    val_episodes: int = 100
    test_episodes: int = 100
    horizon: int = 15
    noisy_observation_std: float = 0.035


@dataclass
class ModelConfig:
    obs_dim: int = 14
    num_actions: int = 5
    latent_dim: int = 64
    hidden_dim: int = 128


@dataclass
class TrainingConfig:
    seed: int = 13
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    latent_loss_weight: float = 0.15
    rollout_loss_weight: float = 0.5
    beacon_loss_weight: float = 0.0
    uncertainty_loss_weight: float = 0.5
    grad_clip_norm: float = 5.0
    device: str = "cpu"


@dataclass
class EvaluationConfig:
    rollout_horizon: int = 10
    latent_sample_limit: int = 1200
    counterfactual_trials: int = 5000


@dataclass
class ExplorationConfig:
    seed: int = 23
    num_episodes: int = 20
    planning_horizon: int = 5
    num_candidates: int = 64
    episode_horizon: int = 12


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
        "exploration": ExplorationConfig(),
    }


def config_to_dict(config: dict) -> dict:
    return {name: asdict(section) for name, section in config.items()}
