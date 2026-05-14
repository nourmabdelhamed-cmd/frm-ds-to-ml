#!/usr/bin/env python3
"""Scaffold a small software-engineering course project."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent


def slug_to_package(slug: str) -> str:
    return slug.replace("-", "_")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


def scaffold(root: Path, title: str, domain: str, package: str) -> None:
    package_dir = root / package
    for directory in [
        root / "docs",
        package_dir,
        root / "tests",
        root / ".github" / "workflows",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    write(
        root / "README.md",
        f"""
        # {title}

        Hands-on course project showing how a prototype becomes maintainable, testable, team-ready software.

        ## Setup

        ```bash
        uv sync --dev
        ```

        ## Run

        ```bash
        uv run python -m {package}
        ```

        ## Test

        ```bash
        uv run pytest
        ```

        ## Teaching Arc

        1. Prototype
        2. Reusable structure
        3. Configuration
        4. Tests
        5. Automation
        6. Observability
        7. Handoff
        """,
    )
    write(
        root / "pyproject.toml",
        f"""
        [project]
        name = "{package.replace('_', '-')}"
        version = "0.1.0"
        description = "{title}"
        requires-python = ">=3.12"
        dependencies = []

        [dependency-groups]
        dev = ["pytest>=8.0"]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["{package}"]

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        addopts = "-q"
        """,
    )
    write(
        package_dir / "__init__.py",
        f'"""Course package for {title}."""\n',
    )
    write(
        package_dir / "__main__.py",
        """
        def main() -> int:
            print("Prototype is now a runnable package.")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """,
    )
    write(
        root / "tests" / "test_smoke.py",
        f"""
        from {package}.__main__ import main


        def test_main_returns_success():
            assert main() == 0
        """,
    )
    write(
        root / ".github" / "workflows" / "ci.yml",
        """
        name: CI

        on:
          push:
          pull_request:

        jobs:
          tests:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
                with:
                  python-version: "3.12"
              - uses: astral-sh/setup-uv@v5
                with:
                  enable-cache: true
              - run: uv sync --dev
              - run: uv run pytest
        """,
    )
    write(
        root / ".gitignore",
        """
        .DS_Store
        .venv/
        __pycache__/
        *.py[cod]
        .pytest_cache/
        .mypy_cache/
        .ruff_cache/
        _site/
        _docs/
        .quarto/
        *.db
        *.log
        """,
    )
    write(
        root / "docs" / "01_prototype.md",
        f"""
        # Stage 1: Prototype

        Domain: {domain}

        Start from the smallest example that makes the idea concrete.
        """,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--title", default="Software Engineering Course Project")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--package-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = args.package_name or slug_to_package(args.path.name)
    scaffold(args.path, args.title, args.domain, package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
