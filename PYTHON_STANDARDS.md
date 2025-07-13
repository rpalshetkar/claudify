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
        msg = f'Cannot process {required}'
        raise ValueError(msg) from e
    
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
- `from __future__ import annotations` FIRST (in EVERY Python file)
- `str | None` not `Optional[str]`
- `list[str]` not `List[str]`  
- `except E as e: raise NewE() from e`
- ALL functions have `-> ReturnType`
- Single quotes for strings
- Keyword-only args after `*`
- Assign error messages to variables before raising

❌ DON'T:
- `from typing import List, Optional, Union`
- Missing return types
- `raise` without `from e`
- Functions >50 lines
- Import paths with 'src' prefix
- String literals in exceptions: `raise ValueError('msg')` → use `msg = 'msg'; raise ValueError(msg)`

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

For examples of formatted output, tables, and progress bars, see [Rich Console Examples](./PYTHON_SNIPPETS.md#rich-console-examples).

### Logging:

For logging configuration and usage examples, see [Logging Examples](./PYTHON_SNIPPETS.md#logging-examples).

### Debugging:

For debugging with icecream, see [Debug Output Examples](./PYTHON_SNIPPETS.md#debug-output-examples).

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

### Implementation

For error handling implementation with APIError class and exception handlers, see [Standardized Error Handling](./PYTHON_SNIPPETS.md#standardized-error-handling).

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

For pytest and coverage configuration, see [Test Configuration](./PYTHON_SNIPPETS.md#test-configuration).

## Configuration Files

For complete configuration templates:
- **pyproject.toml (Minimal)**: See [pyproject.toml Setup](./PYTHON_SNIPPETS.md#pyprojecttoml-setup)
- **pyproject.toml (Full API)**: See [pyproject.toml - Full API](./PYTHON_SNIPPETS.md#pyprojecttoml-full-api)
- **Ruff Configuration**: See [Ruff Configuration](./PYTHON_SNIPPETS.md#rufftoml)

## Verification Commands

For verification commands and setup, see [Verification Commands](./PYTHON_SNIPPETS.md#verification-commands).

## Test File Template

For complete test file templates and patterns, see [Test File Template](./PYTHON_SNIPPETS.md#test-file-template).

## Common Pitfalls to Avoid

1. **Docstring formatting**: Ensure blank lines in docstrings don't have trailing whitespace
2. **Exception messages**: Always assign to variable first: `msg = 'error'; raise ValueError(msg)`
3. **Test mocking**: Use combined `with` statements for multiple patches
4. **Import unused variables**: Remove or use underscore prefix for intentionally unused
5. **Type annotations**: Include for ALL function parameters and returns, including test fixtures

## Related Documents

### Essential References
- **Project Setup**: [PYTHON_MUST.md](./PYTHON_MUST.md) - Start here for new projects
- **Code Examples**: [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md) - All implementation examples
- **Package Choices**: [PYTHON_STACK.md](./PYTHON_STACK.md) - Approved packages only
- **Advanced Features**: [PYTHON_LATER.md](./PYTHON_LATER.md) - When you need more

### Quick Links to Examples
- [Configuration Templates](./PYTHON_SNIPPETS.md#configuration)
- [Error Handling Implementation](./PYTHON_SNIPPETS.md#standardized-error-handling)
- [Test Patterns](./PYTHON_SNIPPETS.md#test-file-template)
- [Security Middleware](./PYTHON_SNIPPETS.md#security-headers-middleware)
- [Logging Setup](./PYTHON_SNIPPETS.md#logging-examples)

## That's it. Security first, errors standardized, tests comprehensive.