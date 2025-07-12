# Python Later Additions

Advanced patterns to add when needed. Build on PYTHON_MUST.md foundation.

## Testing Optimization

### Coverage Configuration

```toml
# pyproject.toml addition
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/migrations/*"]

[tool.coverage.report]
precision = 2
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "@abstract",
]
```

### Performance Testing

```python
# tests/performance/test_load.py
import pytest
from locust import HttpUser, task, between


class APIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def health_check(self):
        self.client.get("/api/health")
    
    @task(3)
    def create_item(self):
        self.client.post("/api/items", json={
            "name": "Load Test Item",
            "price": 99.99
        })


# Benchmark test
@pytest.mark.benchmark
def test_parse_performance(benchmark):
    result = benchmark(expensive_function, data)
    assert result.is_valid
```

### E2E Testing

```python
# tests/e2e/test_user_flow.py
async def test_complete_user_journey(client: AsyncClient):
    """Test full user registration to purchase flow."""
    # 1. Register
    register_response = await client.post("/api/auth/register", json={
        "email": "test@example.com",
        "password": "secure123"
    })
    assert register_response.status_code == 201
    
    # 2. Login
    token = register_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Create item
    item_response = await client.post(
        "/api/items", 
        json={"name": "Test", "price": 10},
        headers=headers
    )
    assert item_response.status_code == 201
```

## Advanced Dependency Injection

### Resource Management

```python
# src/project/core/resources.py
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dependency_injector import resources
from httpx import AsyncClient
from redis.asyncio import Redis

from project.core.config import settings


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


# In container.py
class Container(containers.DeclarativeContainer):
    # ... existing providers ...
    
    http_client = providers.Resource(
        HttpClientResource,
        timeout=config.HTTP_TIMEOUT,
    )
    
    redis = providers.Resource(
        RedisResource,
        url=config.REDIS_URL,
    )
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
    
    # Mock specific services
    container.security_service.override(
        providers.Factory(
            lambda: AsyncMock(spec=SecurityService)
        )
    )
    
    yield container
    
    container.unwire()


async def test_user_service(mock_container):
    """Test with mocked dependencies."""
    user_service = mock_container.user_service()
    
    # Mock security service is automatically injected
    user_service.security.decode_token.return_value = 123
    
    result = await user_service.get_by_token("fake-token")
    assert result is not None
```

### Factory Pattern with DI

```python
# src/project/services/factory.py
from dependency_injector import providers

from project.services.email import EmailService
from project.services.sms import SMSService
from project.services.push import PushService


class NotificationFactory:
    """Factory for notification services."""
    
    def __init__(self):
        self._services = {}
    
    def register(self, name: str, service: type):
        self._services[name] = service
    
    def create(self, name: str) -> NotificationService:
        service_class = self._services.get(name)
        if not service_class:
            raise ValueError(f"Unknown service: {name}")
        return service_class()


# In container
notification_factory = providers.Singleton(
    NotificationFactory,
)

# Wire up services
@containers.copy(Container)
class ApplicationContainer(Container):
    @providers.inject
    def _configure_notifications(
        factory: NotificationFactory = providers.Provide[Container.notification_factory],
    ):
        factory.register("email", EmailService)
        factory.register("sms", SMSService)
        factory.register("push", PushService)
```

### Scoped Dependencies

```python
# src/project/core/scopes.py
from contextvars import ContextVar
from dependency_injector import providers

# Request-scoped context
request_id_var: ContextVar[str] = ContextVar("request_id")


class RequestScope:
    """Request-scoped provider."""
    
    def __init__(self):
        self.request_id = request_id_var.get()
    
    @property
    def correlation_id(self) -> str:
        return f"req-{self.request_id}"


# In container
request_scope = providers.Factory(
    RequestScope,
    scope=providers.Scope.REQUEST,
)

# Middleware to set scope
@app.middleware("http")
async def request_scope_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response
```

## API Scaling

### JWT Authentication

```python
# src/project/core/auth.py
from datetime import datetime, timedelta
from typing import Any

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


def decode_access_token(token: str) -> TokenData | None:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=["HS256"]
        )
        return TokenData(**payload)
    except JWTError:
        return None
```

### Permission System

```python
# src/project/core/permissions.py
from enum import Enum
from functools import wraps

from fastapi import HTTPException, status

from project.api.deps import CurrentUser


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


# Usage
@router.delete("/items/{item_id}")
@require_permissions(Permission.DELETE)
async def delete_item(item_id: int, user: CurrentUser):
    ...
```

### Rate Limiting

```python
# src/project/core/ratelimit.py
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# In main.py
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

# Usage
@router.post("/api/items")
@limiter.limit("5/minute")
async def create_item(request: Request, ...):
    ...
```

### WebSocket Support

```python
# src/project/api/websocket.py
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client {client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### Background Tasks

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


# Usage in API
from project.core.tasks import send_email

@router.post("/api/register")
async def register(data: UserCreate):
    user = await create_user(data)
    send_email.send(user.email, "Welcome!", "Thanks for signing up")
    return user
```

### API Versioning

```python
# src/project/api/v1/routes.py
from fastapi import APIRouter

v1_router = APIRouter(prefix="/api/v1")

# src/project/api/v2/routes.py  
v2_router = APIRouter(prefix="/api/v2")

# In main.py
app.include_router(v1_router)
app.include_router(v2_router)
```

## Security Hardening

### Secret Rotation

```python
# src/project/core/secrets.py
import os
from datetime import datetime, timedelta

import hvac


class SecretManager:
    def __init__(self):
        self.client = hvac.Client(
            url=os.getenv("VAULT_URL"),
            token=os.getenv("VAULT_TOKEN")
        )
    
    def get_secret(self, path: str) -> dict:
        """Get secret from Vault."""
        response = self.client.secrets.kv.v2.read_secret_version(
            path=path
        )
        return response["data"]["data"]
    
    def rotate_database_password(self):
        """Rotate database credentials."""
        new_password = generate_password()
        # Update in Vault
        self.client.secrets.kv.v2.create_or_update_secret(
            path="database/creds",
            secret={"password": new_password}
        )
        # Update database
        update_db_password(new_password)
```

### CORS Configuration

```python
# src/project/core/middleware.py
from fastapi.middleware.cors import CORSMiddleware

from project.core.config import settings

# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
)
```

### Security Headers

```python
# src/project/core/security_headers.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response


# In main.py
app.add_middleware(SecurityHeadersMiddleware)
```

## Database Advanced

### Alembic Migrations

```bash
# Initial setup
alembic init -t async migrations

# alembic.ini
[alembic]
script_location = migrations
prepend_sys_path = .
sqlalchemy.url = postgresql+asyncpg://user:pass@localhost/db

# Create migration
alembic revision --autogenerate -m "Add user table"

# Run migrations
alembic upgrade head
```

### Connection Pooling

```python
# src/project/core/database.py
from sqlalchemy.pool import NullPool, QueuePool

engine = create_async_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)
```

### Read Replicas

```python
# src/project/core/database.py
class RoutingSession(AsyncSession):
    def get_bind(self, mapper=None, clause=None):
        if self._flushing or isinstance(clause, (Update, Delete, Insert)):
            return engines["master"]
        return engines["replica"]


engines = {
    "master": create_async_engine(settings.MASTER_DB_URL),
    "replica": create_async_engine(settings.REPLICA_DB_URL),
}
```

### Query Optimization

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

## Performance

### Profiling

```python
# src/project/core/profiling.py
import cProfile
import pstats
from functools import wraps

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        stats = pstats.Stats(pr)
        stats.sort_stats("cumulative")
        stats.print_stats(10)
        
        return result
    return wrapper
```

### Redis Caching

```python
# src/project/core/cache.py
import json
from typing import Any

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

## Operations

### Dockerfile

```dockerfile
# Multi-stage build
FROM python:3.12-slim as builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv venv && \
    uv pip install --no-cache-dir .

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "project.main:app", "--host", "0.0.0.0"]
```

### Health Checks

```python
# src/project/api/health.py
from fastapi import status

from project.core.database import engine


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

### Graceful Shutdown

```python
# src/project/main.py
import signal
import sys

shutdown_event = asyncio.Event()


def signal_handler(sig, frame):
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await shutdown_event.wait()
    await close_db_connections()
    await redis_client.close()
```

## Monitoring

### Structured Logging

```python
# src/project/core/logging.py
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info("user_login", user_id=user.id, ip=request.client.host)
```

### OpenTelemetry

```python
# src/project/core/telemetry.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add exporter
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument
FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument(engine=engine)
```

### Custom Metrics

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

# Metrics endpoint
@router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Migration Guide

### From MUST to LATER

1. **Testing**: Add coverage, benchmarks, E2E tests
2. **API**: Add JWT, permissions, rate limiting incrementally
3. **Security**: Layer on CORS, headers, secret rotation
4. **Database**: Add migrations, then pooling, then replicas
5. **Performance**: Profile first, then cache, then optimize
6. **Operations**: Local Docker → K8s → Multi-region
7. **Monitoring**: Logs → Metrics → Traces

Each addition builds on the MUST foundation without breaking changes.