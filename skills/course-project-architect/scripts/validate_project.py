#!/usr/bin/env python3
"""Validate the shape of a hands-on course repository."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_ANY = [
    ("README", ["README.md"]),
    ("dependency config", ["pyproject.toml", "package.json", "requirements.txt"]),
    ("tests", ["tests"]),
    ("CI workflow", [".github/workflows/ci.yml"]),
    ("lessons/docs", ["docs", "nbs"]),
]

SHOULD_IGNORE = [
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    "_site/",
    ".quarto/",
    "*.db",
    "*.log",
]


def exists_any(root: Path, paths: list[str]) -> bool:
    return any((root / path).exists() for path in paths)


def validate(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for label, paths in REQUIRED_ANY:
        if not exists_any(root, paths):
            errors.append(f"missing {label}: expected one of {', '.join(paths)}")

    gitignore = root / ".gitignore"
    if not gitignore.exists():
        errors.append("missing .gitignore")
    else:
        text = gitignore.read_text(encoding="utf-8")
        for pattern in SHOULD_IGNORE:
            if pattern not in text:
                warnings.append(f".gitignore should include {pattern}")

    readme = root / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8").lower()
        for section in ["setup", "run", "test"]:
            if section not in text:
                warnings.append(f"README should explain {section}")

    return errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors, warnings = validate(args.path)
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")
    if errors:
        return 1
    print("course project shape looks usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
