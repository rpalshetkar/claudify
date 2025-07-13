# Python Technology Stack

Approved packages by category. No alternatives unless listed. For implementation patterns, see [PYTHON_MUST.md](./PYTHON_MUST.md).

## Core Requirements

- **Python**: 3.12+ (No exceptions)
- **Package Manager**: uv (No pip, poetry, pipenv)

## Web Frameworks

- **FastAPI**: All REST APIs

## CLI Development

- **Typer**: All command-line interfaces

## Data Validation & Settings

- **Pydantic 2.10+**: All data validation
- **DynaConf**: Configuration management (multi-environment support)
- **python-dotenv**: Environment file loading (used by DynaConf)

## Dependency Injection

- **dependency-injector**: IoC container and dependency injection

## Database ORMs & Drivers

### ORMs

- **SQLAlchemy 2.0+**: Primary ORM
- **SQLModel**: When Pydantic models = DB models
- **Tortoise-ORM**: Async-first projects only

### Drivers

- **asyncpg**: PostgreSQL async driver
- **psycopg3**: PostgreSQL sync driver
- **aiosqlite**: SQLite async driver
- **motor**: MongoDB async driver
- **redis-py**: Redis client

## Testing Framework

### Core Testing

- **pytest**: Test runner
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-benchmark**: Performance testing

### Test Data

- **factory-boy**: Test factories
- **faker**: Fake data generation

### Load Testing

- **locust**: Load testing (when needed)

## Code Quality (ALL REQUIRED)

- **ruff**: Linting and formatting
- **mypy**: Static type checking (--strict mode)
- **pre-commit**: Git hooks automation

## Async & Concurrency

- **anyio**: Async compatibility layer
- **httpx**: HTTP client (async/sync)
- **aiofiles**: Async file operations
- **dramatiq**: Background task queue

## Authentication & Security

- **passlib[bcrypt]**: Password hashing
- **python-jose[cryptography]**: JWT tokens
- **python-multipart**: Form data parsing
- **slowapi**: Rate limiting

## Logging & Output

- **loguru**: Structured logging
- **rich**: Terminal formatting
- **icecream**: Debug output (dev only)

## Utilities

### Date & Time

- **pendulum**: All datetime operations

### Data Processing

- **orjson**: Fast JSON parsing
- **polars**: DataFrames (preferred)
- **pandas**: Legacy projects only
- **numpy**: Numerical computing
- **glom**: Numerical computing

### General

- **structlog**: Structured logging (alternative)

## API Development

### Documentation

- **uvicorn**: ASGI server
- **gunicorn**: WSGI server (Django)

## Development Tools

### Type Stubs

- **types-\***: Type stubs as needed

### Schema Translation

- **pydantic2ts**: Pydantic → TypeScript
- **datamodel-code-generator**: OpenAPI → Pydantic

## Infrastructure

### Deployment

- **uvicorn**: ASGI server
- **gunicorn**: WSGI server

### Secret Management

- **hvac**: HashiCorp Vault client

## BANNED - NEVER USE

### Package Managers

- ❌ pip, poetry, pipenv, conda

### Code Quality

- ❌ black, isort, flake8, pylint, autopep8

### HTTP Clients

- ❌ requests, urllib, urllib3, aiohttp

### CLI Tools

- ❌ click, argparse, fire

### Date/Time

- ❌ datetime, dateutil, arrow, maya

### Logging

- ❌ logging module, print statements

### Testing

- ❌ unittest, nose, nose2

### Task Queues

- ❌ celery, rq, huey

### Web Frameworks

- ❌ flask, bottle, tornado, sanic, starlette

### ORMs

- ❌ peewee, orator, pony

### JSON

- ❌ json, simplejson, ujson

## Version Pinning

Always specify minimum versions in pyproject.toml:

```toml
dependencies = [
    "fastapi>=0.115.0",
    "pydantic>=2.10.0",
    "sqlalchemy>=2.0.0",
]
```

## Exceptions

Project-specific packages require:

1. Documentation in project README
2. Justification for not using standard stack
3. Team approval

## Migration Notes

When migrating existing projects:

- requests → httpx
- click → typer
- datetime → pendulum
- logging → loguru
- unittest → pytest
