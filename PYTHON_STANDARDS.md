# Python Code Standards

Pure code style guide. For project setup, see [PYTHON_MUST.md](./PYTHON_MUST.md). For package choices, see [PYTHON_STACK.md](./PYTHON_STACK.md).

## File Template - COPY EXACTLY:

```python
"""Module description in one line."""
from __future__ import annotations

# Standard library
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar

# Third-party
from icecream import ic
from loguru import logger
from rich.console import Console

# Local (no src prefix needed)
from myproject.config import settings

# Types
T = TypeVar("T")
PathLike = str | Path

# Constants
DEFAULT_TIMEOUT = 30.0
console = Console()


def xfunc_name(
    required: str,
    *,
    optional: int = 0,
    flag: bool = False,
) -> dict[str, Any]:
    """One line description.
    
    Args:
        required: What it is
        optional: What it is
        flag: What it is
        
    Returns:
        What it returns
        
    Raises:
        ValueError: When invalid
    """
    try:
        result = process(required)
        ic(result)  # Debug with icecream
    except SpecificError as e:
        logger.error(f'Failed processing {required}: {e}')
        raise ValueError(f'Cannot process {required}') from e
    
    console.print('[green]Success![/green]')
    return {'status': 'ok', 'data': result}


async def xfunc_async(url: str) -> str:
    """Async functions follow same rules."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

## MANDATORY Output Rules

✅ USE:
- `console.print()` for user-facing output
- `logger.{info,error,debug}()` for logs
- `ic()` for debugging (auto-removed in prod)
- Rich tables, panels, progress bars

❌ NEVER:
- `print()` - Use console.print() instead
- Direct stdout writes
- Unformatted output

## MANDATORY Code Rules

✅ DO:
- `from __future__ import annotations` FIRST
- `str | None` not `Optional[str]`
- `list[str]` not `List[str]`  
- `except E as e: raise NewE() from e`
- ALL functions have `-> ReturnType`
- Single quotes for strings
- Keyword-only args after `*`

❌ DON'T:
- `from typing import List, Optional, Union`
- Missing return types
- `raise` without `from e`
- Functions >50 lines
- Import paths with 'src' prefix

## Import Rules
- Package in src/myproject/
- Import as: `from myproject.module import thing`
- Tests import the same way
- NO relative imports with dots
- NO 'src.' prefix ever

## Import Order
```python
# 1. Future
from __future__ import annotations

# 2. Standard library
import json
from pathlib import Path

# 3. Third-party
import httpx
from pydantic import BaseModel

# 4. Local application
from myproject.core import settings
from myproject.models import User
```

## Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`
- Type aliases: `PascalCase`

## Docstring Format (Google Style)
```python
def complex_function(
    param1: str,
    param2: int,
    *,
    optional: bool = False,
) -> dict[str, Any]:
    """Short description.
    
    Longer description if needed. Can span multiple lines
    and explain complex behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        optional: Description of optional
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not positive
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result["status"])
        ok
    """
```

## Output Examples

### Rich Console:
```python
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# Formatted output
console.print('[bold green]Processing complete![/bold green]')

# Tables
table = Table(title='Results')
table.add_column('Name', style='cyan')
table.add_column('Value', style='magenta')
table.add_row('Count', '42')
console.print(table)

# Progress
for item in track(items, description='Processing...'):
    process(item)
```

### Logging:
```python
from loguru import logger

# Configure once in main
logger.add('app.log', rotation='1 day')

# Use throughout
logger.info('Starting process')
logger.error(f'Failed: {error}')
logger.debug(f'Details: {data}')
```

### Debugging:
```python
from icecream import ic

# Debug values
ic(variable)
ic(function_call())

# Disable in production
ic.disable()  # In config based on env
```

## Type Hints

### Basic Types:
```python
name: str = 'Alice'
age: int = 30
price: float = 19.99
active: bool = True
data: bytes = b'raw data'
```

### Collections:
```python
names: list[str] = ['Alice', 'Bob']
scores: dict[str, int] = {'Alice': 100}
unique: set[int] = {1, 2, 3}
coords: tuple[float, float] = (1.0, 2.0)
```

### Optional & Union:
```python
# Modern syntax only
maybe_name: str | None = None
number: int | float = 42
result: str | int | None = get_result()
```

### Advanced:
```python
from collections.abc import Callable, Iterator
from typing import Any, TypeVar, Generic

T = TypeVar('T')
Handler = Callable[[Request], Response]
DataMap = dict[str, list[int | str]]

class Container(Generic[T]):
    value: T
```

## Security Standards

### MANDATORY Security Requirements

✅ **ALWAYS:**
- Add security headers middleware to ALL APIs
- Validate and sanitize ALL user inputs
- Use parameterized queries (SQLAlchemy handles this)
- Hash passwords with bcrypt/argon2
- Implement rate limiting on public endpoints
- Track request IDs for correlation
- Use HTTPS in production

❌ **NEVER:**
- Trust user input without validation
- Store passwords in plain text
- Expose internal error details to users
- Use string formatting for SQL queries
- Skip CORS configuration
- Log sensitive data (passwords, tokens)

### Input Validation Requirements

```python
from pydantic import BaseModel, field_validator, EmailStr
import re

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    
    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        # Remove special chars, limit length
        cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', '', v)
        return cleaned[:100]
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain special character')
        return v
```

### Security Headers
All APIs MUST implement these headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy: default-src 'self'`

## Error Handling Standards

### API Error Response Format

ALL errors MUST follow this structure:

```json
{
    "error": "ERROR_CODE",
    "detail": "Human readable message",
    "timestamp": "2024-01-01T00:00:00Z",
    "path": "/api/endpoint",
    "request_id": "uuid-here"
}
```

### Error Codes Convention
- `AUTH_*` - Authentication errors
- `VALIDATION_*` - Input validation errors
- `PERMISSION_*` - Authorization errors
- `RESOURCE_*` - Resource errors (not found, conflict)
- `SYSTEM_*` - System errors (database, external service)

### Example Implementation

```python
from datetime import datetime
from fastapi import Request
from fastapi.responses import JSONResponse

class APIError(Exception):
    def __init__(self, status_code: int, error_code: str, detail: str):
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.error_code,
            'detail': exc.detail,
            'timestamp': datetime.utcnow().isoformat(),
            'path': request.url.path,
            'request_id': request.state.request_id
        }
    )

# Usage
raise APIError(400, 'VALIDATION_INVALID_EMAIL', 'Email format is invalid')
```

## Testing Requirements

### Coverage Standards
- **Minimum**: 80% code coverage
- **Target**: 90% for critical paths
- **Exclude**: Migrations, admin, config files

### Test Categories Required

1. **Unit Tests** (tests/unit/)
   - Test individual functions/methods
   - Mock external dependencies
   - Fast execution (<1s per test)

2. **Integration Tests** (tests/integration/)
   - Test API endpoints end-to-end
   - Use test database
   - Verify full request/response cycle

3. **Security Tests** (tests/security/)
   - SQL injection attempts
   - XSS prevention
   - Authentication bypass attempts
   - Rate limiting verification

4. **Performance Tests** (tests/performance/)
   - Response time benchmarks
   - Concurrent request handling
   - Memory usage under load

### Test Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/migrations/*", "*/admin.py"]

[tool.coverage.report]
fail_under = 80
show_missing = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

## Configuration Files

### pyproject.toml Template
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "your-project-name"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
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
packages = ["src/your-project"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-xvs --tb=short --strict-markers"
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"

[tool.mypy]
python_version = "3.12"
strict = true
mypy_path = "src"
```

### ruff.toml Template
```toml
line-length = 88
target-version = "py312"

[lint]
select = ["ALL"]
ignore = [
    "D",      # docstrings
    "COM812", # trailing comma
    "ISC001", # single line concat
    "TC002",  # third-party type-checking imports
    "TC003",  # standard library type-checking imports
    "Q000",   # double quotes (incompatible with single quote formatting)
    "Q003",   # double quotes (incompatible with single quote formatting)
]

[lint.flake8-type-checking]
runtime-evaluated-base-classes = [
    "pydantic.BaseModel",
    "fastapi.APIRouter",
]
runtime-evaluated-decorators = [
    "pytest.fixture",
    "fastapi.Depends",
]

[format]
quote-style = "single"
indent-style = "space"
```

## Verification Commands
```bash
# Run from project root
ruff check . --fix && ruff format .
mypy . --strict
pytest --cov=src --cov-report=term-missing
bandit -r src/  # Security checks
```

## That's it. Security first, errors standardized, tests comprehensive.