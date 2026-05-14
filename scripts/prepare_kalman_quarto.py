"""Prepare static-safe Kalman notebooks for Quarto rendering.

The upstream Kalman book notebooks contain stored ipywidgets state from older
widget versions. Static GitHub Pages builds can render those as browser-side
"module not found" errors. This script keeps the upstream submodule untouched
and writes sanitized render copies under generated/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "kalman-bayesian-filters"
DEST = ROOT / "generated" / "kalman-bayesian-filters"

NOTEBOOKS = [
    "table_of_contents.ipynb",
    "00-Preface.ipynb",
    "01-g-h-filter.ipynb",
    "02-Discrete-Bayes.ipynb",
    "03-Gaussians.ipynb",
    "04-One-Dimensional-Kalman-Filters.ipynb",
    "05-Multivariate-Gaussians.ipynb",
    "06-Multivariate-Kalman-Filters.ipynb",
    "07-Kalman-Filter-Math.ipynb",
    "08-Designing-Kalman-Filters.ipynb",
    "09-Nonlinear-Filtering.ipynb",
    "10-Unscented-Kalman-Filter.ipynb",
    "11-Extended-Kalman-Filters.ipynb",
    "12-Particle-Filters.ipynb",
    "13-Smoothing.ipynb",
    "14-Adaptive-Filtering.ipynb",
    "Appendix-A-Installation.ipynb",
    "Appendix-B-Symbols-and-Notations.ipynb",
    "Appendix-D-HInfinity-Filters.ipynb",
    "Appendix-E-Ensemble-Kalman-Filters.ipynb",
    "Appendix-G-Designing-Nonlinear-Kalman-Filters.ipynb",
    "Appendix-H-Least-Squares-Filters.ipynb",
    "Appendix-I-Analytic-Evaluation-of-Performance.ipynb",
]


def strip_widget_output(output: dict[str, Any]) -> dict[str, Any] | None:
    data = output.get("data")
    if isinstance(data, dict):
        for key in list(data):
            if "jupyter.widget" in key:
                data.pop(key)
        if not data and output.get("output_type") in {"display_data", "execute_result"}:
            return None
    return output


def sanitize_notebook(source: Path, dest: Path) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    notebook.get("metadata", {}).pop("widgets", None)
    for cell in notebook.get("cells", []):
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue
        cleaned = []
        for output in outputs:
            stripped = strip_widget_output(output)
            if stripped is not None:
                cleaned.append(stripped)
        cell["outputs"] = cleaned
    dest.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def copy_assets() -> None:
    for dirname in ["animations", "figs"]:
        source = SOURCE / dirname
        target = DEST / dirname
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(
            "Kalman submodule is missing. Run `git submodule update --init --recursive`."
        )
    DEST.mkdir(parents=True, exist_ok=True)
    copy_assets()
    for notebook in NOTEBOOKS:
        sanitize_notebook(SOURCE / notebook, DEST / notebook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
