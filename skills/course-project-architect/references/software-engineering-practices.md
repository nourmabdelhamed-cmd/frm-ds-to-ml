# Software Engineering Practices

Use this as the general checklist for course repositories.

## Version Control

Teach:

- small commits
- branch names with intent
- pull requests for review
- clean diffs
- avoiding generated/runtime files in Git

Project should include:

- `.gitignore`
- README with Git workflow notes when relevant
- generated artifacts excluded from source control

## Project Structure

Teach:

- source code belongs in a predictable package/app folder
- examples and docs should call the same reusable code
- scripts should be thin entry points

Prefer:

```text
package_name/
tests/
docs/ or nbs/
scripts/
README.md
```

## Dependency Management

Teach:

- reproducible installs
- committed lock files when the tool supports them
- minimal dependency sets

Python default:

```text
pyproject.toml
uv.lock
```

## Configuration

Teach:

- separate code from experiment/runtime choices
- use command-line overrides for common teaching variants
- avoid editing internals to change behavior

Good examples:

```text
params.yaml
config.yaml
.env.example
CLI flags
```

Do not commit secrets.

## Testing

Teach:

- fast tests
- deterministic fixtures
- smoke tests for command-line entry points
- no expensive external work in CI

Tests should prove workflow contracts, not only implementation details.

## Automation

Teach:

- CI as a team quality gate
- separate test workflow from publish/deploy workflow
- small, reliable jobs

Default GitHub Actions shape:

```text
.github/workflows/ci.yml
.github/workflows/deploy-pages.yml  # optional
```

## Observability

Choose domain-appropriate feedback:

- logs for scripts and services
- metrics for training or performance
- traces for agents and distributed flows
- visual artifacts for image/video workflows
- reports for data quality

The point is to make progress and failure inspectable.

## Artifact And State Boundaries

Teach what Git should and should not track.

Track:

```text
source code
tests
docs
configuration
small metadata
workflow definitions
```

Do not track:

```text
datasets
checkpoints
databases
caches
logs
rendered docs
build outputs
secrets
```

Use DVC, object storage, package registries, or deployment platforms where appropriate.

## Documentation

README should include:

- project goal
- setup
- run commands
- test commands
- local preview
- deployment/publishing if applicable
- what to version-control
- what to ignore

Docs or notebooks should teach the staged arc, not duplicate the README.
