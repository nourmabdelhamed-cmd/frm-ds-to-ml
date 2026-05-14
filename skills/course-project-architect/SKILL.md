---
name: course-project-architect
description: Use when creating, refactoring, or reviewing a hands-on teaching repository that demonstrates software engineering best practices through a concrete technical case study. Applies to ML, data engineering, backend APIs, frontend apps, Git workflows, automation, and similar courses.
---

# Course Project Architect

## Goal

Create teaching repositories that move learners from "my prototype works" to "a team can understand, test, run, change, and trust it."

The skill should produce a concrete project, not only slides. Every course should have runnable examples, a visible teaching arc, and a small engineering workflow that students can repeat.

## Default Teaching Arc

Use this progression unless the user gives a different one:

1. Prototype or manual workflow
2. Reusable structure
3. Configuration or explicit inputs
4. Tests and validation
5. Automation
6. Observability or feedback
7. Artifact, data, or state boundaries
8. Documentation
9. Deployment, publishing, or handoff

## Repository Principles

- Keep the first example concrete and runnable.
- Prefer boring, widely used open-source tools.
- Keep setup minimal and beginner-friendly.
- Separate source code from generated/runtime artifacts.
- Make tests fast and deterministic.
- Make expensive, external, or long-running work optional or clearly bounded.
- Add clear commands for local development.
- Add CI that verifies the workflow contract.
- Use documentation as part of the learning path, not an afterthought.
- Keep the Git history and review surface clean.

## Standard Deliverables

For most course repos, create or maintain:

- staged lesson notebooks, markdown pages, or docs
- reusable package/application code
- config-driven entry point or explicit command interface
- README with setup, run, test, preview, and deploy instructions
- tests
- CI workflow
- `.gitignore`
- optional artifact/state management
- optional publishing workflow

## Domain Selection

Read only the relevant reference files:

- General structure and lesson shape: `references/teaching-arc.md`
- Engineering practices checklist: `references/software-engineering-practices.md`
- ML or AI workflow courses: `references/ml-specialization.md`
- Git branching, rebasing, and collaboration courses: `references/git-specialization.md`
- Backend/API courses: `references/backend-api-specialization.md`

## Project Workflow

1. Identify the target learner and concrete case study.
2. Define the staged arc in 4-6 lessons.
3. Choose the smallest realistic toolchain.
4. Scaffold the repo and runnable examples.
5. Add tests and CI early.
6. Add documentation and local preview commands.
7. Add deployment/publishing only when it serves the teaching goal.
8. Validate with the scripts when available.

## Tooling Defaults

For Python projects:

- Use `uv`.
- Prefer `pytest`.
- Put reusable code in a package.
- Keep notebooks or docs as the teaching interface.
- Keep runtime output ignored by Git.

For docs:

- Prefer Quarto, nbdev, MkDocs, or a similarly reproducible static docs workflow.
- Keep generated output ignored unless the user explicitly wants committed static output.

For CI:

- Keep test CI separate from publishing/deployment.
- Avoid large dataset downloads, long training, cloud costs, or external services in CI.

## Validation Checklist

Before final response, check as much as applies:

- setup command works or is documented
- tests pass
- package/app imports
- demo command runs on a small local path
- generated/runtime artifacts are ignored
- README includes setup, run, test, preview, and publishing/deployment notes
- CI avoids expensive or flaky work
- staged lessons match the actual project files

## Scripts

Use `scripts/scaffold_project.py` when the user wants a new repo skeleton.

Use `scripts/validate_project.py` when reviewing whether a course repo has the expected engineering shape.
