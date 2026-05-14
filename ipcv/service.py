"""Optional FastAPI handoff for local localization replay and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ipcv.localization_workflow import (
    load_failure_scenario,
    load_localization_config,
    resolve_scenario_path,
    run_localization_workflow,
)


def list_scenario_names(root: str | Path = "configs/failures") -> list[str]:
    scenario_root = Path(root)
    if not scenario_root.exists():
        return ["baseline"]
    return ["baseline", *sorted(path.stem for path in scenario_root.glob("*.yaml"))]


def create_app() -> Any:
    """Create the optional FastAPI app.

    FastAPI is intentionally imported inside this factory so importing `ipcv`
    remains part of the lightweight core workflow.
    """

    from fastapi import FastAPI
    from pydantic import BaseModel

    class ReplayRequest(BaseModel):
        config_path: str = "configs/localization.yaml"
        scenario: str = "baseline"
        output_dir: str | None = None
        enable_mlflow: bool = False

    app = FastAPI(title="IPCV Localization Service")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "ipcv-localization"}

    @app.get("/scenarios")
    def scenarios() -> dict[str, list[str]]:
        return {"scenarios": list_scenario_names()}

    @app.post("/replay")
    def replay(request: ReplayRequest) -> dict[str, Any]:
        config = load_localization_config(request.config_path)
        if request.output_dir is not None:
            config = config.with_overrides(output_dir=request.output_dir)
        scenario = load_failure_scenario(resolve_scenario_path(request.scenario))
        run = run_localization_workflow(
            config,
            scenario=scenario,
            enable_mlflow=request.enable_mlflow,
        )
        return {
            "metrics": run.evaluation.metrics,
            "artifacts": [str(path) for path in run.artifacts],
        }

    @app.post("/evaluate")
    def evaluate(request: ReplayRequest) -> dict[str, Any]:
        return replay(request)

    return app


app = create_app()
