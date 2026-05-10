"""Teaching project for moving from data science notebooks to production ML."""

__version__ = "0.1.0"

from .workflow import TrainingConfig, build_model, load_config, run_training

__all__ = ["TrainingConfig", "build_model", "load_config", "run_training"]
