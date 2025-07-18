# Python Must-Haves

Essential patterns for every Python package. Start here, scale later.

**Prerequisites**:

- Package choices: See [PYTHON_STACK.md](./PYTHON_STACK.md)
- Code style rules: See [PYTHON_STANDARDS.md](./PYTHON_STANDARDS.md)
- Code examples: See [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md)
- Advanced features: See [PYTHON_LATER.md](./PYTHON_LATER.md)

## Project Structure

```
# When using /xinit, this structure is created:
folder/
├── src/
│   └── module/              # Your module name
│       ├── __init__.py
│       ├── main.py          # Entry point
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
└── ruff.toml
```

**Note**: The `src/module/` structure keeps your module code separate from project configuration files. This is Python packaging best practice.

### Configuration Files

- **pyproject.toml**: See [pyproject.toml Setup](./PYTHON_SNIPPETS.md#pyprojecttoml-setup)
- **ruff.toml**: See [Ruff Configuration](./PYTHON_SNIPPETS.md#rufftoml)
- **settings.toml**: See [Settings TOML](./PYTHON_SNIPPETS.md#settings-toml)
- **.env.example**: See below for environment variables


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
- **Files**: `src/package/api/deps.py`, `src/package/api/routes.py`, `src/package/main.py`

## Dependency Injection

- **Container**: See [Container Setup](./PYTHON_SNIPPETS.md#container-setup)
- **Wiring**: See [Wiring FastAPI](./PYTHON_SNIPPETS.md#wiring-fastapi)
- **Services**: See [Base Service Pattern](./PYTHON_SNIPPETS.md#base-service-pattern)
- **Files**: `src/package/core/container.py`, `src/package/services/user.py`

## Security Basics (MANDATORY)

### Required Security Components

1. **Security Headers Middleware** (MUST have on ALL APIs)

   - See [Security Headers Middleware](./PYTHON_SNIPPETS.md#security-headers-middleware)
   - Files: `src/package/core/security_headers.py`, `src/package/main.py`
   - Add to main.py: `app.add_middleware(SecurityHeadersMiddleware)`

2. **Input Validation & Sanitization** (MUST validate ALL inputs)

   - See [Enhanced Input Validation](./PYTHON_SNIPPETS.md#enhanced-input-validation)
   - Files: `src/package/api/schemas.py`
   - Use field validators for sanitization
   - Enforce password complexity (12+ chars, uppercase, number, special)

3. **Configuration & Secrets**
   - **Configuration**: See [DynaConf Setup](./PYTHON_SNIPPETS.md#dynaconf-setup)
   - **Settings File**: See [Settings TOML](./PYTHON_SNIPPETS.md#settings-toml)
   - **Security Service**: See [Security Service](./PYTHON_SNIPPETS.md#security-service)
   - **Files**: `src/package/core/config.py`, `settings.toml`, `src/package/core/security.py`

### .env.example

```bash
# Environment selection
MODULE_ENV=development  # or production

# Secrets (production only)
MODULE_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/dbname
MODULE_SECRET_KEY=your-secret-key-here-generate-with-openssl

# Optional overrides
MODULE_DEBUG=false
MODULE_APP_NAME=MyApp
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
- **Files**: `src/package/core/database.py`, `src/package/models/user.py`

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
python -m module.main
```

## Build Automation

See [BUILD_AUTOMATION.md](./BUILD_AUTOMATION.md) for modern build practices.

**Quick start - add to pyproject.toml:**
```toml
[tool.uv.scripts]
dev = "python -m myapp.main --reload"
test = "pytest -v"
lint = "ruff check . --fix"
format = "ruff format ."
typecheck = "mypy src"
check = ["lint", "format", "typecheck", "test"]
```

Then use: `uv run dev`, `uv run test`, `uv run check`

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
17. Latest python type annotations - no Any, Optional, Union etc

## Quick Implementation Guide

All code examples have been moved to [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md) for easy reference and reuse. The snippets include:

1. **Configuration**: DynaConf setup and settings files
2. **Dependency Injection**: Container setup and wiring
3. **Database**: Sessions, models, and queries
4. **Testing**: Fixtures, factories, and test patterns
5. **API**: Dependencies, routes, and error handling
6. **Security**: Services and authentication
7. **Services**: Business logic patterns

## Related Documents

### Essential References
- **Code Examples**: [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md) - All implementation examples
- **Package Choices**: [PYTHON_STACK.md](./PYTHON_STACK.md) - Approved packages only
- **Coding Standards**: [PYTHON_STANDARDS.md](./PYTHON_STANDARDS.md) - Style and conventions
- **Advanced Features**: [PYTHON_LATER.md](./PYTHON_LATER.md) - When you need more

### Quick Links
- [pyproject.toml Setup](./PYTHON_SNIPPETS.md#pyprojecttoml-setup)
- [DynaConf Configuration](./PYTHON_SNIPPETS.md#dynaconf-setup)
- [Security Middleware](./PYTHON_SNIPPETS.md#security-headers-middleware)
- [Testing Patterns](./PYTHON_SNIPPETS.md#test-database-fixture)
- [Error Handling](./PYTHON_SNIPPETS.md#standardized-error-handling)
