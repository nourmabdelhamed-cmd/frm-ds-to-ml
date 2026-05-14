# Backend API Specialization

Use this reference when the course teaches web APIs, services, backend design, or production service workflows.

## Mapping

Generic course arc to backend/API:

```text
prototype              -> single script or notebook request handler
reusable structure     -> app package
configuration          -> environment/config file
tests                  -> unit and API tests
automation             -> CI
observability          -> structured logs and health endpoints
artifact/state boundary-> database/migrations/local state ignored
documentation          -> OpenAPI and README
handoff                -> deployment or container
```

## Defaults

- Prefer FastAPI for Python courses unless the user gives another framework.
- Keep the first service local and easy to run.
- Include a `/health` endpoint.
- Include tests that do not require external services.
- Use `.env.example`, never commit `.env`.
- Keep database files and local volumes ignored.

## Recommended Files

```text
app/
tests/
docs/
pyproject.toml
.env.example
README.md
.github/workflows/ci.yml
```

## Teaching Points

- request/response contract
- app factory or clear entry point
- config by environment
- validation
- tests with client fixtures
- structured logging
- CI quality gate
- deployment boundary

## Avoid

- adding Docker or cloud deployment before the local workflow is clear
- requiring paid services for core tests
- hiding important behavior in framework magic without explanation
