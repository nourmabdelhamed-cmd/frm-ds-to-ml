"""Prepare render-safe MLOps companion pages for Quarto.

The upstream MLOps tutorial is kept as a Git submodule. This script leaves that
submodule untouched and writes Quarto source copies under generated/ so the
course site can render a stable sidebar section.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "mlops-hands-on-tutorial"
DEST = ROOT / "generated" / "mlops-hands-on-tutorial"

PAGES = [
    ("README.md", "overview.qmd", "MLOps Hands On Tutorial"),
    (
        "week-0-setups-pytorch-lightning/README.md",
        "week-0-project-setup.qmd",
        "Week 0 Project Setup And Train Model",
    ),
    (
        "week-2-hydra-configuration/README.md",
        "week-2-hydra-configuration.qmd",
        "Week 2 Hydra Configuration",
    ),
    ("week-3-dvc/README.md", "week-3-dvc.qmd", "Week 3 Data Version Control"),
    ("week-4-onnx/README.md", "week-4-onnx.qmd", "Week 4 ONNX Packaging"),
    ("week-5-docker/README.md", "week-5-docker.qmd", "Week 5 Docker Packaging"),
    (
        "week-6-github-actions/README.md",
        "week-6-github-actions.qmd",
        "Week 6 GitHub Actions",
    ),
    ("week-7-ecr/README.md", "week-7-ecr.qmd", "Week 7 AWS ECR"),
    (
        "week-8-serverless-lambda/README.md",
        "week-8-serverless-lambda.qmd",
        "Week 8 Serverless Lambda",
    ),
    ("week-9-monitoring/README.md", "week-9-monitoring.qmd", "Week 9 Monitoring"),
]


def strip_leading_h1(markdown: str, title: str) -> str:
    pattern = re.compile(rf"^#\s+{re.escape(title)}\s*\n+", flags=re.IGNORECASE)
    return pattern.sub("", markdown, count=1)


def write_page(source_relative: str, dest_name: str, title: str) -> None:
    source = SOURCE / source_relative
    if not source.exists():
        raise SystemExit(f"MLOps source page is missing: {source}")
    markdown = source.read_text(encoding="utf-8")
    markdown = strip_leading_h1(markdown, title).lstrip()
    rendered = "\n".join(
        [
            "---",
            f'title: "{title}"',
            "format: html",
            "execute:",
            "  eval: false",
            "---",
            "",
            markdown,
        ]
    )
    (DEST / dest_name).write_text(rendered, encoding="utf-8")


def copy_assets() -> None:
    images = SOURCE / "images"
    if images.exists():
        shutil.copytree(images, DEST / "images")


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(
            "MLOps submodule is missing. Run "
            "`git submodule update --init external/mlops-hands-on-tutorial`."
        )
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True, exist_ok=True)
    copy_assets()
    for source_relative, dest_name, title in PAGES:
        write_page(source_relative, dest_name, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
