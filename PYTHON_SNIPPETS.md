# Python Code Snippets

Centralized code patterns and examples. Reference these snippets in your projects.

## Table of Contents

- [Configuration](#configuration)
- [Dependency Injection](#dependency-injection)
- [Database](#database)
- [Testing](#testing)
- [API Patterns](#api-patterns)
- [Security](#security)
- [Services](#services)
- [Background Tasks](#background-tasks)
- [Caching](#caching)
- [Monitoring](#monitoring)

## Configuration

### DynaConf Setup
```python
# src/project/core/config.py
from __future__ import annotations

from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="PROJECT",  # export PROJECT_DEBUG=true
    settings_files=["settings.toml", ".secrets.toml"],
    environments=True,  # Enable [development], [production] sections
    load_dotenv=True,  # Load .env files
    env_switcher="PROJECT_ENV",  # export PROJECT_ENV=production
    
    # Defaults
    DEBUG=False,
    PROJECT_NAME="Project",
)

# Required settings validation
settings.validators.register(
    settings.Validator("DATABASE_URL", must_exist=True),
    settings.Validator("SECRET_KEY", must_exist=True),
)
```

### Settings TOML
```toml
# settings.toml
[default]
project_name = "MyProject"

[development]
debug = true
database_url = "postgresql+asyncpg://dev:dev@localhost/devdb"

[production]
debug = false
# DATABASE_URL from environment
```

## Dependency Injection

### Container Setup
```python
# src/project/core/container.py
from __future__ import annotations

from dependency_injector import containers, providers
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from project.core.config import settings
from project.core.security import SecurityService
from project.services.user import UserService


class Container(containers.DeclarativeContainer):
    """Application DI container."""
    
    config = providers.Configuration()
    
    # Database
    db_engine = providers.Singleton(
        create_async_engine,
        config.DATABASE_URL,
        echo=config.DEBUG,
    )
    
    db_session = providers.Factory(
        AsyncSession,
        bind=db_engine,
        expire_on_commit=False,
    )
    
    # Services
    security_service = providers.Singleton(
        SecurityService,
        secret_key=config.SECRET_KEY,
    )
    
    user_service = providers.Factory(
        UserService,
        session=db_session,
        security=security_service,
    )


# Global container instance
container = Container()
container.config.from_dict(settings.as_dict())
```

### Wiring FastAPI
```python
# src/project/main.py
from project.core.container import container

# Wire the container
container.wire(modules=["project.api.deps", "project.api.routes"])
```

### Resource Management
```python
# src/project/core/resources.py
from dependency_injector import resources
from httpx import AsyncClient
from redis.asyncio import Redis


class HttpClientResource(resources.AsyncResource):
    """Managed HTTP client resource."""
    
    async def init(self, timeout: int = 30) -> AsyncClient:
        return AsyncClient(timeout=timeout)
    
    async def shutdown(self, client: AsyncClient) -> None:
        await client.aclose()


class RedisResource(resources.AsyncResource):
    """Managed Redis connection resource."""
    
    async def init(self, url: str) -> Redis:
        return await Redis.from_url(url)
    
    async def shutdown(self, redis: Redis) -> None:
        await redis.close()
```

## Database

### Database Connection Pooling
```python
# src/project/core/database.py
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import contextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import QueuePool
from loguru import logger

from project.core.config import settings


# Create engine with proper connection pooling
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    poolclass=QueuePool,
    pool_size=5,  # Number of persistent connections
    max_overflow=10,  # Maximum overflow connections
    pool_timeout=30,  # Timeout for getting connection
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Verify connections before use
)

Base = declarative_base()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide database session with automatic transaction handling."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@contextmanager
async def get_db_connection():
    """Get database connection with monitoring."""
    async with engine.connect() as conn:
        logger.info(f"Active connections: {engine.pool.size()}")
        logger.info(f"Overflow connections: {engine.pool.overflow()}")
        yield conn
```

### Base Model
```python
# src/project/models/user.py
from __future__ import annotations

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from project.core.database import Base


class User(Base):
    """User model."""
    
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

### Advanced Queries
```python
# Eager loading
from sqlalchemy.orm import selectinload

query = select(User).options(
    selectinload(User.posts).selectinload(Post.comments)
)

# Bulk operations
await session.execute(
    insert(User),
    [{"name": f"User {i}", "email": f"user{i}@test.com"} 
     for i in range(1000)]
)

# Raw SQL when needed
result = await session.execute(
    text("SELECT * FROM users WHERE created_at > :date"),
    {"date": cutoff_date}
)
```

## Testing

### Test Database Fixture
```python
# tests/conftest.py
from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from project.core.database import get_session, Base
from project.main import app


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide transactional database session for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        async with AsyncSession(conn) as session:
            yield session
            # Automatic rollback after test


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide test client with database override."""
    app.dependency_overrides[get_session] = lambda: db_session
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()
```

### Test Factories
```python
# tests/factories.py
from __future__ import annotations

import factory
from factory import Faker

from project.models import User


class UserFactory(factory.Factory):
    """User factory for tests."""
    
    class Meta:
        model = User
    
    id = factory.Sequence(lambda n: n)
    email = Faker("email")
    name = Faker("name")
    hashed_password = "$2b$12$test.hash"  # Pre-hashed 'password'
    is_active = True
    
    @factory.post_generation
    def groups(self, create: bool, extracted: list[str] | None) -> None:
        """Handle M2M relationships."""
        if not create or not extracted:
            return
        
        self.groups.extend(extracted)
```

### Testing with DI Override
```python
# tests/test_with_di.py
import pytest
from unittest.mock import AsyncMock

from project.core.container import Container
from project.services.user import UserService


@pytest.fixture
def mock_container():
    """Container with mocked dependencies."""
    container = Container()
    
    container.security_service.override(
        providers.Factory(
            lambda: AsyncMock(spec=SecurityService)
        )
    )
    
    yield container
    
    container.unwire()
```

## API Patterns

### Dependencies
```python
# src/project/api/deps.py
from __future__ import annotations

from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import Depends, HTTPException, status

from project.core.container import Container
from project.models import User
from project.services.user import UserService


@inject
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_service: UserService = Depends(Provide[Container.user_service]),
) -> User:
    """Get current authenticated user."""
    user = await user_service.get_by_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
SessionDep = Annotated[AsyncSession, Depends(Provide[Container.db_session])]
```

### Request/Response Models
```python
# src/project/api/schemas.py
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """User creation request."""
    email: EmailStr
    password: str
    name: str


class UserResponse(BaseModel):
    """User response."""
    id: int
    email: str
    name: str
    is_active: bool
    
    class Config:
        from_attributes = True
```

### API Routes
```python
# src/project/api/routes.py
from fastapi import APIRouter, HTTPException, status

from project.api.deps import CurrentUser, SessionDep
from project.api.schemas import UserCreate, UserResponse


router = APIRouter(prefix="/api")


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/users", response_model=UserResponse)
async def create_user(
    data: UserCreate,
    session: SessionDep,
) -> Any:
    """Create new user."""
    # Implementation
    pass
```

### Standardized Error Handling
```python
# src/project/core/errors.py
from __future__ import annotations

from datetime import datetime
from fastapi import Request
from fastapi.responses import JSONResponse


class APIError(Exception):
    """Standardized API error."""
    
    def __init__(self, status_code: int, error_code: str, detail: str):
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail


# src/project/main.py
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle API errors with standard format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.error_code,
            'detail': exc.detail,
            'timestamp': datetime.utcnow().isoformat(),
            'path': request.url.path,
            'request_id': getattr(request.state, 'request_id', 'unknown')
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            'error': 'SYSTEM_INTERNAL_ERROR',
            'detail': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat(),
            'path': request.url.path,
            'request_id': getattr(request.state, 'request_id', 'unknown')
        },
    )


# Usage in endpoints
from project.core.errors import APIError

@router.post("/api/users")
async def create_user(data: UserCreate):
    existing = await find_by_email(data.email)
    if existing:
        raise APIError(
            status_code=409,
            error_code='RESOURCE_ALREADY_EXISTS',
            detail=f'User with email {data.email} already exists'
        )
    
    # Create user...
```

## Security

### Security Headers Middleware
```python
# src/project/core/security_headers.py
from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


# In main.py - MANDATORY for all APIs
from project.core.security_headers import SecurityHeadersMiddleware

app.add_middleware(SecurityHeadersMiddleware)
```

### Security Service
```python
# src/project/core/security.py
from __future__ import annotations

from passlib.context import CryptContext


class SecurityService:
    """Security service for auth operations."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain, hashed)
    
    def create_token(self, user_id: int) -> str:
        """Create access token."""
        # Use python-jose in production
        return f"token-{user_id}"
    
    def decode_token(self, token: str) -> int | None:
        """Decode and validate token."""
        if token.startswith("token-"):
            return int(token.split("-")[1])
        return None
```

### JWT Implementation
```python
# src/project/core/auth.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel

from project.core.config import settings


class TokenData(BaseModel):
    sub: str
    exp: datetime
    scopes: list[str] = []


def create_access_token(
    subject: str | int,
    expires_delta: timedelta | None = None,
    scopes: list[str] | None = None,
) -> str:
    """Create JWT access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "scopes": scopes or [],
    }
    return jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm="HS256"
    )
```

### Enhanced Input Validation
```python
# src/project/api/schemas.py
from __future__ import annotations

import re
from pydantic import BaseModel, field_validator, EmailStr, constr


class UserCreate(BaseModel):
    """User creation with validation and sanitization."""
    
    email: EmailStr
    password: str
    name: constr(min_length=2, max_length=100)
    phone: str | None = None
    
    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        """Remove dangerous characters, normalize whitespace."""
        # Remove special characters except spaces and hyphens
        cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', '', v)
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        # Ensure length limits
        return cleaned[:100]
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Enforce strong password requirements."""
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain special character')
        return v
    
    @field_validator('phone')
    @classmethod
    def sanitize_phone(cls, v: str | None) -> str | None:
        """Clean and validate phone number."""
        if not v:
            return None
        # Remove all non-digits
        digits = re.sub(r'\D', '', v)
        # Check length (10-15 digits typical)
        if len(digits) < 10 or len(digits) > 15:
            raise ValueError('Invalid phone number length')
        return digits


class HTMLContent(BaseModel):
    """Content that may contain HTML - sanitize it."""
    
    content: str
    
    @field_validator('content')
    @classmethod
    def sanitize_html(cls, v: str) -> str:
        """Remove dangerous HTML/scripts."""
        # Basic sanitization - use bleach library in production
        # Remove script tags
        v = re.sub(r'<script[^>]*>.*?</script>', '', v, flags=re.DOTALL | re.IGNORECASE)
        # Remove event handlers
        v = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', v, flags=re.IGNORECASE)
        # Remove javascript: URLs
        v = re.sub(r'javascript:', '', v, flags=re.IGNORECASE)
        return v


# SQL Injection prevention - always use parameterized queries
# Good - SQLAlchemy handles parameterization
stmt = select(User).where(User.email == email)

# Bad - Never do string formatting
# query = f"SELECT * FROM users WHERE email = '{email}'"  # NEVER DO THIS
```

### Permission Decorator
```python
# src/project/core/permissions.py
from enum import Enum
from functools import wraps

from fastapi import HTTPException, status


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


def require_permissions(*permissions: Permission):
    """Decorator to check user permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: CurrentUser, **kwargs):
            if not all(p in user.permissions for p in permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator
```

## Services

### Base Service Pattern
```python
# src/project/services/user.py
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from project.core.security import SecurityService
from project.models import User


class UserService:
    """User service handling business logic."""
    
    def __init__(self, session: AsyncSession, security: SecurityService):
        self.session = session
        self.security = security
    
    async def get_by_token(self, token: str) -> User | None:
        """Get user by auth token."""
        user_id = self.security.decode_token(token)
        if not user_id:
            return None
        
        return await self.session.get(User, user_id)
    
    async def create_user(self, email: str, password: str, name: str) -> User:
        """Create new user."""
        hashed_password = self.security.hash_password(password)
        
        user = User(
            email=email,
            name=name,
            hashed_password=hashed_password,
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        
        return user
    
    async def find_by_email(self, email: str) -> User | None:
        """Find user by email."""
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
```

## Background Tasks

### Dramatiq Setup
```python
# src/project/core/tasks.py
import dramatiq
from dramatiq.brokers.redis import RedisBroker

from project.core.config import settings

redis_broker = RedisBroker(url=settings.REDIS_URL)
dramatiq.set_broker(redis_broker)


@dramatiq.actor
def send_email(email: str, subject: str, body: str):
    """Background email task."""
    # Email sending logic
    pass


@dramatiq.actor(max_retries=3)
def process_upload(file_id: str):
    """Process uploaded file."""
    # Processing logic
    pass
```

### Task Usage
```python
# In API endpoint
from project.core.tasks import send_email

@router.post("/api/register")
async def register(data: UserCreate):
    user = await create_user(data)
    
    # Queue background task
    send_email.send(
        user.email, 
        "Welcome!", 
        "Thanks for signing up"
    )
    
    return user
```

## Caching

### Redis Cache Decorator
```python
# src/project/core/cache.py
import json
from typing import Any
from functools import wraps

import redis.asyncio as redis

from project.core.config import settings

redis_client = redis.from_url(settings.REDIS_URL)


async def cache_get(key: str) -> Any | None:
    """Get value from cache."""
    value = await redis_client.get(key)
    return json.loads(value) if value else None


async def cache_set(
    key: str, 
    value: Any, 
    expire: int = 3600
) -> None:
    """Set value in cache."""
    await redis_client.setex(
        key, 
        expire, 
        json.dumps(value)
    )


def cached(expire: int = 3600):
    """Cache decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try cache first
            result = await cache_get(cache_key)
            if result is not None:
                return result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_set(cache_key, result, expire)
            
            return result
        return wrapper
    return decorator
```

### Cache Usage
```python
@cached(expire=300)  # 5 minutes
async def get_user_stats(user_id: int) -> dict:
    """Get cached user statistics."""
    # Expensive calculation
    return calculate_stats(user_id)
```

## Request Tracking

### Request ID Middleware
```python
# src/project/core/request_id.py
from __future__ import annotations

import uuid
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable to store request ID
request_id_var: ContextVar[str] = ContextVar('request_id', default='unknown')


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to track request IDs for correlation."""
    
    async def dispatch(self, request: Request, call_next):
        # Get or generate request ID
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        
        # Set in context var for logging
        request_id_var.set(request_id)
        
        # Store in request state for access in handlers
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = request_id
        
        return response


# In main.py - Add AFTER security headers
from project.core.request_id import RequestIDMiddleware, request_id_var

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

# Usage in logging
from project.core.request_id import request_id_var

logger.bind(request_id=request_id_var.get()).info("Processing request")

# Usage in error handlers
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.error_code,
            'detail': exc.detail,
            'timestamp': datetime.utcnow().isoformat(),
            'path': request.url.path,
            'request_id': request.state.request_id  # Available here
        }
    )
```

## Monitoring

### Structured Logging
```python
# src/project/core/logging.py
from __future__ import annotations

import structlog
from structlog.contextvars import merge_contextvars

from project.core.config import settings
from project.core.request_id import request_id_var


def configure_logging() -> None:
    """Configure structured logging with context."""
    structlog.configure(
        processors=[
            # Add context vars (includes request_id)
            merge_contextvars,
            # Standard processors
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Conditional formatting
            structlog.processors.JSONRenderer()
            if not settings.DEBUG
            else structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Call in main.py startup
configure_logging()

# Get logger instance
logger = structlog.get_logger()

# Usage with automatic request ID
logger.info("user_login", user_id=user.id, ip=request.client.host)
# Output includes: {"request_id": "uuid", "user_id": 123, ...}

# Usage with context binding
log = logger.bind(user_id=user.id, role=user.role)
log.info("Starting operation")
log.info("Operation completed")  # Both logs have user_id and role

# Usage in services
class UserService:
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    async def create_user(self, data: UserCreate) -> User:
        self.logger.info("Creating user", email=data.email)
        try:
            user = await self._create(data)
            self.logger.info("User created", user_id=user.id)
            return user
        except Exception as e:
            self.logger.error("User creation failed", email=data.email, error=str(e))
            raise
```

### Request Metrics
```python
# src/project/core/metrics.py
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
request_count = Counter(
    "app_requests_total", 
    "Total requests", 
    ["method", "endpoint", "status"]
)

request_duration = Histogram(
    "app_request_duration_seconds",
    "Request duration",
    ["method", "endpoint"]
)

# Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

### Health Checks
```python
# src/project/api/health.py
@router.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe."""
    try:
        # Check database
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        
        # Check cache
        await redis_client.ping()
        
        return {"status": "ready"}
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready"}
        )
```

## Usage Guidelines

1. **Copy snippets** - Don't modify in this file
2. **Adapt to your needs** - Snippets are starting points
3. **Follow imports** - Ensure all imports are included
4. **Check dependencies** - Verify packages in PYTHON_STACK.md
5. **Test thoroughly** - Snippets are examples, not production code

## Contributing

When adding new snippets:
1. Keep them focused and single-purpose
2. Include all necessary imports
3. Add usage examples
4. Document any prerequisites
5. Link to relevant sections in other docs