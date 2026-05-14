"""Agnostic training runner for the demo.

This file intentionally knows only how to load configuration and start a run.
Dataset, model, training, tracking, and artifact behavior live in the package.
"""

try:
    from ipcv.workflow import cli_main
except ModuleNotFoundError as exc:
    optional_modules = {"segmentation_models_pytorch", "torch", "torchvision"}
    if exc.name in optional_modules:
        raise SystemExit(
            "The segmentation appendix requires optional dependencies. "
            "Run `uv sync --dev --group segmentation` first."
        ) from exc
    raise


if __name__ == "__main__":
    raise SystemExit(cli_main())
