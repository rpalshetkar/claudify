# Python Must-Haves

Essential patterns for every Python project. Start here, scale later.

**Prerequisites**: 
- Package choices: See [PYTHON_STACK.md](./PYTHON_STACK.md)
- Code style rules: See [PYTHON_STANDARDS.md](./PYTHON_STANDARDS.md)
- Code examples: See [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md)
- Advanced features: See [PYTHON_LATER.md](./PYTHON_LATER.md)

## Project Structure

```
project/
├── src/
│   └── project/            # Your package name
│       ├── __init__.py
│       ├── main.py         # Entry point
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py
│       │   └── deps.py     # Dependencies
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py   # Settings
│       │   ├── container.py # DI container
│       │   ├── database.py
│       │   └── security.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── user.py
│       └── services/
│           ├── __init__.py
│           └── user.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Fixtures
│   ├── factories.py        # Test factories
│   └── unit/
│       └── __init__.py
├── .env.example
├── settings.toml           # DynaConf settings
├── .secrets.toml          # Git-ignored secrets
├── pyproject.toml
└── ruff.toml               # See below for config
```

### pyproject.toml Setup

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    # See PYTHON_STACK.md for approved packages
    "fastapi>=0.115.0",
    "pydantic>=2.10.0",
    "dynaconf>=3.2.0",
    "dependency-injector>=4.41.0",
    "httpx>=0.27.0",
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.29.0",
    "passlib[bcrypt]>=1.7.4",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "factory-boy>=3.3.0",
    "faker>=24.0.0",
    "loguru>=0.7.0",
    "rich>=13.0.0",
    "icecream>=2.1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/project"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-xvs --tb=short --strict-markers"
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.12"
strict = true
mypy_path = "src"
```

### ruff.toml

```toml
line-length = 88
target-version = "py312"

[lint]
select = ["ALL"]
ignore = [
    "D",      # docstrings
    "ANN101", # self type
    "ANN102", # cls type
    "COM812", # trailing comma
    "ISC001", # single line concat
]

[format]
quote-style = "single"
indent-style = "space"
```

## Testing Infrastructure

- **Fixtures**: See [Test Database Fixture](./PYTHON_SNIPPETS.md#test-database-fixture)
- **Factories**: See [Test Factories](./PYTHON_SNIPPETS.md#test-factories)
- **DI Testing**: See [Testing with DI Override](./PYTHON_SNIPPETS.md#testing-with-di-override)
- **Location**: `tests/conftest.py`, `tests/factories.py`

## API Essentials

- **Dependencies**: See [API Dependencies](./PYTHON_SNIPPETS.md#dependencies)
- **Models**: See [Request/Response Models](./PYTHON_SNIPPETS.md#requestresponse-models)
- **Routes**: See [API Routes](./PYTHON_SNIPPETS.md#api-routes)
- **Error Handling**: See [Standardized Error Handling](./PYTHON_SNIPPETS.md#standardized-error-handling) (MANDATORY)
- **Request Tracking**: See [Request ID Middleware](./PYTHON_SNIPPETS.md#request-id-middleware) (MANDATORY)
- **Files**: `src/project/api/deps.py`, `src/project/api/routes.py`, `src/project/main.py`

## Dependency Injection

- **Container**: See [Container Setup](./PYTHON_SNIPPETS.md#container-setup)
- **Wiring**: See [Wiring FastAPI](./PYTHON_SNIPPETS.md#wiring-fastapi)
- **Services**: See [Base Service Pattern](./PYTHON_SNIPPETS.md#base-service-pattern)
- **Files**: `src/project/core/container.py`, `src/project/services/user.py`

## Security Basics (MANDATORY)

### Required Security Components

1. **Security Headers Middleware** (MUST have on ALL APIs)
   - See [Security Headers Middleware](./PYTHON_SNIPPETS.md#security-headers-middleware)
   - Files: `src/project/core/security_headers.py`, `src/project/main.py`
   - Add to main.py: `app.add_middleware(SecurityHeadersMiddleware)`

2. **Input Validation & Sanitization** (MUST validate ALL inputs)
   - See [Enhanced Input Validation](./PYTHON_SNIPPETS.md#enhanced-input-validation)
   - Files: `src/project/api/schemas.py`
   - Use field validators for sanitization
   - Enforce password complexity (12+ chars, uppercase, number, special)

3. **Configuration & Secrets**
   - **Configuration**: See [DynaConf Setup](./PYTHON_SNIPPETS.md#dynaconf-setup)
   - **Settings File**: See [Settings TOML](./PYTHON_SNIPPETS.md#settings-toml)
   - **Security Service**: See [Security Service](./PYTHON_SNIPPETS.md#security-service)
   - **Files**: `src/project/core/config.py`, `settings.toml`, `src/project/core/security.py`

### .env.example

```bash
# Environment selection
PROJECT_ENV=development  # or production

# Secrets (production only)
PROJECT_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/dbname
PROJECT_SECRET_KEY=your-secret-key-here-generate-with-openssl

# Optional overrides
PROJECT_DEBUG=false
PROJECT_PROJECT_NAME=MyProject
```

### .gitignore additions

```
.secrets.toml
.env
settings.local.toml
```

## Database Foundation

- **Session Management**: See [Database Session](./PYTHON_SNIPPETS.md#database-session)
- **Base Model**: See [Base Model](./PYTHON_SNIPPETS.md#base-model)
- **Advanced Queries**: See [Advanced Queries](./PYTHON_SNIPPETS.md#advanced-queries)
- **Files**: `src/project/core/database.py`, `src/project/models/user.py`

## Commands

```bash
# Install dependencies
uv venv
uv pip install -e .

# Run tests
pytest
pytest -x  # Stop on first failure
pytest -k test_create  # Run specific tests

# Type checking
mypy src

# Linting
ruff check . --fix
ruff format .

# Run application
python -m project.main
```

## Key Principles

### Security MUST-HAVES
1. **ALWAYS add security headers** to ALL APIs (see Security Basics above)
2. **ALWAYS validate AND sanitize** ALL user inputs
3. **ALWAYS use standardized error responses** (see PYTHON_STANDARDS.md)
4. **ALWAYS track request IDs** for debugging and correlation
5. **NEVER trust user input** without validation
6. **NEVER expose internal errors** to users

### Development Standards
7. **Always use factories** for test data
8. **Always use fixtures** for common setup
9. **Always validate input** with Pydantic field validators
10. **Always use typed Python** (mypy strict)
11. **Always handle errors** with APIError class
12. **Never use print()** - use loguru/rich (see PYTHON_STANDARDS.md)
13. **Never hardcode secrets** - use .env
14. **Never concatenate SQL** - use parameters
15. **Follow import rules** from PYTHON_STANDARDS.md
16. **Use only approved packages** from PYTHON_STACK.md

## Quick Implementation Guide

All code examples have been moved to [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md) for easy reference and reuse. The snippets include:

1. **Configuration**: DynaConf setup and settings files
2. **Dependency Injection**: Container setup and wiring
3. **Database**: Sessions, models, and queries
4. **Testing**: Fixtures, factories, and test patterns
5. **API**: Dependencies, routes, and error handling
6. **Security**: Services and authentication
7. **Services**: Business logic patterns

## Next Steps

When you need more features, see:
- [PYTHON_LATER.md](./PYTHON_LATER.md) - Advanced patterns
- [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md) - All code examples
- [PYTHON_STACK.md](./PYTHON_STACK.md) - Package reference