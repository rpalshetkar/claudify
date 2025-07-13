# XSETTINGS.md - Settings Feature Specification

## Overview

XSettings is a configuration management system built on top of DynaConf and XObjPrototype, providing flexible, environment-aware settings management with strong validation and type safety.

## Core Design Decisions

### 1. Base Class: XObjPrototype

- XSettings inherits from XObjPrototype for Pydantic-based validation
- Provides type safety and automatic validation
- Integrates with the project's base model architecture

### 2. Loading Order: Configurable (Option C)

- Default order: `['defaults', 'toml', 'env', 'secrets']`
- Configurable at initialization or via config file
- Override capability for testing scenarios

### 3. Validation Strategy: Configurable (Option C)

- Three modes: `eager`, `lazy`, `hybrid` (default)
- Hybrid mode validates required fields immediately, optional on access
- Configurable per instance for different use cases

### 4. Environment Support: Hybrid (Option C)

- Priority: parameter > env var > class default > 'development'
- Support for multi-environment loading (import/export scenarios)
- Lazy loading of additional environments

### 5. Default Values: Hybrid Storage (Option C)

- `.env.default` for simple values and secret templates
- `settings.toml [default]` for complex structures
- Clear precedence rules

### 6. Access Pattern: Hybrid with Nested Structure (Option C - Modified)

- Default singleton for convenience
- Explicit DI support for testing
- Nested Pydantic models for logical grouping (database, api, cache, etc.)

## Architecture

### Class Structure

```python
from __future__ import annotations

from typing import Literal, Any
from pydantic import Field, field_validator, BaseModel
from dynaconf import Dynaconf
import os

from core.base import XObjPrototype


class DatabaseSettings(BaseModel):
    """Database configuration settings."""
    url: str = Field(description='Database connection URL')
    pool_size: int = Field(default=10, description='Connection pool size')
    max_connections: int = Field(default=100, description='Maximum connections')
    timeout: int = Field(default=30, description='Connection timeout in seconds')
    retry_attempts: int = Field(default=3, description='Number of retry attempts')
    retry_delay: int = Field(default=1, description='Delay between retries in seconds')


class APISettings(BaseModel):
    """External API configuration settings."""
    key: str | None = Field(default=None, description='API key')
    secret: str | None = Field(default=None, description='API secret')
    timeout: int = Field(default=30, description='Request timeout in seconds')
    base_url: str | None = Field(default=None, description='Base URL for API')
    retry_count: int = Field(default=3, description='Number of retries on failure')
    rate_limit: int = Field(default=100, description='Requests per minute')


class CacheSettings(BaseModel):
    """Cache configuration settings."""
    enabled: bool = Field(default=True, description='Enable caching')
    ttl: int = Field(default=3600, description='Default TTL in seconds')
    backend: str = Field(default='redis', description='Cache backend type')
    redis_url: str | None = Field(default=None, description='Redis connection URL')
    max_entries: int = Field(default=10000, description='Maximum cache entries')
    eviction_policy: str = Field(default='lru', description='Cache eviction policy')


class LoggingSettings(BaseModel):
    """Logging configuration settings."""
    level: str = Field(default='info', description='Log level')
    format: str = Field(default='json', description='Log format')
    file: str | None = Field(default=None, description='Log file path')
    max_size: int = Field(default=10485760, description='Max log file size in bytes')
    backup_count: int = Field(default=5, description='Number of backup files')
    console: bool = Field(default=True, description='Enable console logging')


class SecuritySettings(BaseModel):
    """Security configuration settings."""
    secret_key: str = Field(description='Application secret key')
    jwt_secret: str | None = Field(default=None, description='JWT signing secret')
    jwt_algorithm: str = Field(default='HS256', description='JWT algorithm')
    jwt_expiry: int = Field(default=3600, description='JWT expiry in seconds')
    cors_origins: list[str] = Field(default_factory=list, description='Allowed CORS origins')
    allowed_hosts: list[str] = Field(default_factory=lambda: ['*'], description='Allowed hosts')


class ServerSettings(BaseModel):
    """Server configuration settings."""
    host: str = Field(default='0.0.0.0', description='Server host')
    port: int = Field(default=8000, description='Server port')
    workers: int = Field(default=1, description='Number of workers')
    reload: bool = Field(default=False, description='Auto-reload on changes')
    debug: bool = Field(default=False, description='Debug mode')


class XSettings(XObjPrototype):
    """
    Settings management with DynaConf integration.

    Provides environment-aware configuration with validation,
    configurable loading order, and multi-environment support.
    """

    # Nested structure for settings categories
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings
    server: ServerSettings = Field(default_factory=ServerSettings)

    # Application-level settings
    app_name: str = Field(default='MyApp', description='Application name')
    environment: str = Field(default='development', description='Current environment')
    version: str = Field(default='1.0.0', description='Application version')
    debug: bool = Field(default=False, description='Global debug mode')

    # Class-level attributes
    _default_instance: XSettings | None = None
    _default_env: str | None = None
    _dynaconf: Dynaconf | None = None

    def __init__(
        self,
        env: str | None = None,
        loading_order: list[str] | None = None,
        validation_mode: Literal['eager', 'lazy', 'hybrid'] = 'hybrid',
        load_envs: list[str] | None = None,
        **kwargs
    ):
        """
        Initialize settings with configurable options.

        Args:
            env: Primary environment to load
            loading_order: Custom loading order for settings sources
            validation_mode: When to validate settings
            load_envs: Additional environments to load (for import/export)
            **kwargs: Additional settings overrides (supports nested via dots)
        """
        # Determine environment
        self.environment = self._determine_environment(env)
        self.validation_mode = validation_mode
        self.loaded_envs = {}

        # Initialize DynaConf with loading order
        self._init_dynaconf(loading_order)

        # Load primary environment
        self._load_environment(self.environment)

        # Load additional environments if specified
        if load_envs:
            for env_name in load_envs:
                self.loaded_envs[env_name] = self._load_environment_data(env_name)

        # Apply any kwargs overrides (supports nested paths)
        self._apply_overrides(kwargs)

        # Initialize parent
        super().__init__()

        # Perform validation based on mode
        self._validate_based_on_mode()

    @classmethod
    def get_default(cls) -> XSettings:
        """Get or create default settings instance."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def set_default(cls, settings: XSettings) -> None:
        """Set default settings instance."""
        cls._default_instance = settings
```

### Loading Order Configuration

```python
def _init_dynaconf(self, loading_order: list[str] | None) -> None:
    """Initialize DynaConf with specified loading order."""
    default_order = ['defaults', 'toml', 'env', 'secrets']

    # Check if loading order is in config file
    config_order = self._get_config_loading_order()

    # Priority: parameter > config > default
    order = loading_order or config_order or default_order

    # Map order to DynaConf settings
    settings_files = []
    loaders = []

    for source in order:
        if source == 'defaults':
            settings_files.extend(['.env.default', 'settings.toml'])
        elif source == 'toml':
            settings_files.append('settings.toml')
        elif source == 'env':
            loaders.append('dynaconf.loaders.env_loader')
        elif source == 'secrets':
            settings_files.append('.secrets.toml')

    self._dynaconf = Dynaconf(
        environments=True,
        envvar_prefix='PROJECT',
        settings_files=settings_files,
        loaders=loaders,
        merge_enabled=True,
    )

def _load_environment(self, env: str) -> None:
    """Load settings from DynaConf for specified environment."""
    self._dynaconf.setenv(env)

    # Load nested settings from DynaConf
    self._load_nested_settings()

def _load_nested_settings(self) -> None:
    """Load settings from DynaConf into nested Pydantic models."""
    # Database settings
    if hasattr(self._dynaconf, 'database'):
        self.database = DatabaseSettings(**self._dynaconf.database.to_dict())

    # API settings
    if hasattr(self._dynaconf, 'api'):
        self.api = APISettings(**self._dynaconf.api.to_dict())

    # Cache settings
    if hasattr(self._dynaconf, 'cache'):
        self.cache = CacheSettings(**self._dynaconf.cache.to_dict())

    # Logging settings
    if hasattr(self._dynaconf, 'logging'):
        self.logging = LoggingSettings(**self._dynaconf.logging.to_dict())

    # Security settings (required)
    if hasattr(self._dynaconf, 'security'):
        self.security = SecuritySettings(**self._dynaconf.security.to_dict())
    else:
        # Security is required, so we need to fail gracefully
        raise ValueError("Security settings are required but not found in configuration")

    # Server settings
    if hasattr(self._dynaconf, 'server'):
        self.server = ServerSettings(**self._dynaconf.server.to_dict())

    # Top-level settings
    if hasattr(self._dynaconf, 'app_name'):
        self.app_name = self._dynaconf.app_name
    if hasattr(self._dynaconf, 'version'):
        self.version = self._dynaconf.version
    if hasattr(self._dynaconf, 'debug'):
        self.debug = self._dynaconf.debug

def _apply_overrides(self, overrides: dict[str, Any]) -> None:
    """Apply overrides to settings, supporting nested paths."""
    for key, value in overrides.items():
        if '.' in key:
            # Handle nested paths like 'database.url'
            parts = key.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            # Direct attribute
            setattr(self, key, value)
```

### Environment Management

```python
def switch_to(self, env: str) -> None:
    """Switch to a different environment."""
    self.environment = env
    self._load_environment(env)
    self._validate_based_on_mode()

def get_env_value(self, key: str, env: str | None = None) -> Any:
    """Get value from specific environment."""
    if env and env in self.loaded_envs:
        return self.loaded_envs[env].get(key)
    return getattr(self, key)

@property
def envs(self):
    """Lazy access to all environments."""
    class EnvironmentAccessor:
        def __init__(self, settings):
            self._settings = settings

        def __getattr__(self, env_name: str) -> dict[str, Any]:
            if env_name not in self._settings.loaded_envs:
                self._settings.loaded_envs[env_name] = \
                    self._settings._load_environment_data(env_name)
            return self._settings.loaded_envs[env_name]

    return EnvironmentAccessor(self)
```

### Validation Modes

```python
def _validate_based_on_mode(self) -> None:
    """Validate settings based on configured mode."""
    if self.validation_mode == 'eager':
        self._validate_all()
    elif self.validation_mode == 'hybrid':
        self._validate_required()
    # lazy mode: no upfront validation

def _validate_required(self) -> None:
    """Validate only required fields."""
    for field_name, field_info in self.model_fields.items():
        if field_info.is_required:
            # Trigger validation by accessing
            getattr(self, field_name)

def _validate_all(self) -> None:
    """Validate all fields."""
    for field_name in self.model_fields:
        getattr(self, field_name)
```

## File Structure

```
server/
├── .env.default              # Simple defaults and secret templates
├── .env                      # Local overrides (gitignored)
├── settings.toml             # Complex configuration with environments
├── .secrets.toml            # Secrets (gitignored)
├── src/
│   └── server/
│       ├── core/
│       │   ├── settings.py   # XSettings implementation
│       │   └── base.py       # XObjPrototype
│       └── config.py         # Settings instance
```

### .env.default Example

```bash
# Database
PROJECT_DATABASE__URL=mongodb://localhost:27017/myapp
PROJECT_DATABASE__POOL_SIZE=10
PROJECT_DATABASE__TIMEOUT=30

# API Configuration
PROJECT_API__KEY=your-api-key-here
PROJECT_API__TIMEOUT=30
PROJECT_API__RATE_LIMIT=100

# Security
PROJECT_SECURITY__SECRET_KEY=change-this-in-production
PROJECT_SECURITY__JWT_SECRET=change-this-in-production

# Logging
PROJECT_LOGGING__LEVEL=info
PROJECT_LOGGING__FORMAT=json

# Server
PROJECT_SERVER__HOST=0.0.0.0
PROJECT_SERVER__PORT=8000

# Application
PROJECT_DEBUG=false
PROJECT_APP_NAME=MyApp
```

### settings.toml Example

```toml
[default]
app_name = "MyApp"
version = "1.0.0"
debug = false

[default.database]
url = "mongodb://localhost:27017/myapp"
pool_size = 10
max_connections = 100
timeout = 30
retry_attempts = 3
retry_delay = 1

[default.api]
timeout = 30
retry_count = 3
rate_limit = 100

[default.cache]
enabled = true
ttl = 3600
backend = "redis"
redis_url = "redis://localhost:6379/0"
max_entries = 10000
eviction_policy = "lru"

[default.logging]
level = "info"
format = "json"
console = true
max_size = 10485760
backup_count = 5

[default.security]
secret_key = "change-this-in-production"
jwt_algorithm = "HS256"
jwt_expiry = 3600
cors_origins = ["http://localhost:3000"]
allowed_hosts = ["*"]

[default.server]
host = "0.0.0.0"
port = 8000
workers = 1
reload = false
debug = false

# Development environment overrides
[development]
debug = true

[development.database]
url = "mongodb://localhost:27017/myapp_dev"

[development.logging]
level = "debug"
format = "text"

[development.server]
reload = true
debug = true

# Production environment overrides
[production]
debug = false

[production.database]
pool_size = 50
max_connections = 500

[production.cache]
ttl = 7200

[production.logging]
level = "warning"
file = "/var/log/myapp/app.log"

[production.server]
workers = 4

# Test environment overrides
[test]
[test.database]
url = "mongodb://localhost:27017/myapp_test"

[test.cache]
enabled = false

[test.logging]
level = "debug"
```

## Usage Examples

### Basic Usage

```python
from core.settings import XSettings

# Using default singleton
settings = XSettings.get_default()

# Access nested settings
print(settings.database.url)
print(settings.database.pool_size)
print(settings.api.timeout)
print(settings.cache.ttl)
print(settings.logging.level)

# Access top-level settings
print(settings.app_name)
print(settings.debug)

# Creating specific instance
dev_settings = XSettings(env='development')
prod_settings = XSettings(env='production')

# Compare settings across environments
print(f"Dev DB: {dev_settings.database.url}")
print(f"Prod DB: {prod_settings.database.url}")
```

### Testing Usage

```python
import pytest
from core.settings import XSettings

def test_with_custom_settings():
    # Create test-specific settings with nested overrides
    test_settings = XSettings(
        env='test',
        **{
            'database.url': 'mongodb://testdb:27017/test',
            'cache.enabled': False,
            'logging.level': 'debug'
        }
    )

    # Use in test
    assert test_settings.cache.enabled is False
    assert 'test' in test_settings.database.url
    assert test_settings.logging.level == 'debug'

@pytest.fixture
def settings():
    """Fixture for test settings."""
    return XSettings(env='test', validation_mode='eager')

def test_nested_validation():
    # Test that nested models validate properly
    with pytest.raises(ValidationError):
        XSettings(
            env='test',
            **{
                'database.pool_size': 'invalid',  # Should be int
                'api.timeout': -1  # Should be positive
            }
        )
```

### Dependency Injection Usage

```python
from dependency_injector import containers, providers
from core.settings import XSettings

class Container(containers.DeclarativeContainer):
    config = providers.Singleton(XSettings)

    database = providers.Singleton(
        MongoDB,
        url=config.provided.database.url,
        pool_size=config.provided.database.pool_size,
        timeout=config.provided.database.timeout,
    )

    cache = providers.Singleton(
        RedisCache,
        url=config.provided.cache.redis_url,
        ttl=config.provided.cache.ttl,
        enabled=config.provided.cache.enabled,
    )

# In services
class UserService:
    def __init__(self, settings: XSettings):
        self.settings = settings
        self.api_timeout = settings.api.timeout
        self.rate_limit = settings.api.rate_limit

    def configure_client(self):
        return APIClient(
            base_url=self.settings.api.base_url,
            key=self.settings.api.key,
            timeout=self.settings.api.timeout
        )
```

### Multi-Environment Usage (Import/Export)

```python
# Load multiple environments
settings = XSettings(
    env='production',
    load_envs=['test', 'staging']
)

# Access different environments
def migrate_data():
    # Get test database URL
    test_db_url = settings.get_env_value('database.url', env='test')
    test_db = MongoDB(test_db_url)

    # Production is primary environment
    prod_db = MongoDB(settings.database.url)

    # Migrate data
    data = test_db.export_all()
    prod_db.import_data(data)

# Using lazy environment access
def compare_settings():
    test_cache = settings.envs.test['cache']['ttl']
    prod_cache = settings.envs.production['cache']['ttl']

    print(f"Test cache TTL: {test_cache}")
    print(f"Prod cache TTL: {prod_cache}")

# Compare nested settings across environments
def audit_database_configs():
    for env_name in ['development', 'staging', 'production']:
        env_data = settings.envs[env_name]
        db_config = env_data.get('database', {})
        print(f"\n{env_name.upper()} Database Config:")
        print(f"  URL: {db_config.get('url')}")
        print(f"  Pool Size: {db_config.get('pool_size')}")
        print(f"  Timeout: {db_config.get('timeout')}")
```

### Custom Loading Order

```python
# For testing with only env vars
test_settings = XSettings(
    loading_order=['env', 'defaults']
)

# For production with secrets priority
prod_settings = XSettings(
    loading_order=['defaults', 'toml', 'secrets', 'env']
)
```

### Environment Switching

```python
settings = XSettings()

# Start in development
print(f"Dev DB: {settings.database_url}")

# Switch to production
settings.switch_to('production')
print(f"Prod DB: {settings.database_url}")

# Temporary environment context (future enhancement)
# with settings.environment('staging'):
#     staging_data = fetch_data(settings.api_url)
```

## Testing Strategy

### Unit Tests

```python
class TestXSettings:
    def test_default_loading(self):
        settings = XSettings()
        assert settings.environment == 'development'
        assert settings.debug is True
        # Test nested access
        assert isinstance(settings.database.url, str)
        assert settings.cache.enabled is True

    def test_environment_override(self):
        settings = XSettings(env='production')
        assert settings.environment == 'production'
        assert settings.debug is False
        # Test nested overrides
        assert settings.database.pool_size == 50  # Production override
        assert settings.cache.ttl == 7200  # Production override

    def test_nested_validation(self):
        # Test nested model validation
        with pytest.raises(ValidationError):
            XSettings(
                validation_mode='eager',
                **{
                    'database.pool_size': -1,  # Invalid: negative
                    'api.timeout': 'invalid',  # Invalid: not an int
                }
            )

    def test_nested_overrides(self):
        settings = XSettings(
            env='development',
            **{
                'database.url': 'mongodb://custom:27017/db',
                'cache.ttl': 1800,
                'logging.level': 'warning'
            }
        )

        assert settings.database.url == 'mongodb://custom:27017/db'
        assert settings.cache.ttl == 1800
        assert settings.logging.level == 'warning'

    def test_loading_order_with_nested(self):
        # Test that env vars can override nested settings
        os.environ['PROJECT_DATABASE__URL'] = 'env-db-url'
        os.environ['PROJECT_CACHE__TTL'] = '9999'

        settings = XSettings(loading_order=['defaults', 'env'])
        assert settings.database.url == 'env-db-url'
        assert settings.cache.ttl == 9999

    def test_multi_environment_loading(self):
        settings = XSettings(
            env='development',
            load_envs=['test', 'production']
        )

        # Test primary environment
        assert 'dev' in settings.database.url

        # Test loaded environments
        test_db = settings.get_env_value('database.url', 'test')
        prod_db = settings.get_env_value('database.url', 'production')

        assert 'test' in test_db
        assert test_db != prod_db

        # Test nested environment access
        test_cache = settings.envs.test['cache']['enabled']
        assert test_cache is False  # Test disables cache

    def test_security_required(self):
        # Security settings are required
        with pytest.raises(ValueError, match="Security settings are required"):
            # Create a mock scenario where security is missing
            settings = XSettings()
            settings._dynaconf = type('MockDynaconf', (), {})()
            settings._load_nested_settings()
```

### Integration Tests

```python
def test_with_real_files(tmp_path):
    # Create temporary config files
    env_default = tmp_path / '.env.default'
    env_default.write_text('DATABASE_URL=default-db\nAPI_KEY=default-key')

    settings_toml = tmp_path / 'settings.toml'
    settings_toml.write_text('''
        [default]
        cache_ttl = 3600

        [production]
        cache_ttl = 7200
    ''')

    # Test loading
    settings = XSettings(env='production')
    assert settings.cache_ttl == 7200
```

## Security Considerations

1. **Secrets Management**

   - Never commit `.secrets.toml` or `.env` files
   - Use placeholders in `.env.default`
   - Validate secret format/strength

2. **Environment Isolation**

   - Ensure production secrets are not accessible in development
   - Use separate secret stores per environment
   - Audit environment access

3. **Validation**
   - Validate all external inputs
   - Sanitize database URLs
   - Check API keys format

## Migration Guide

### From Simple Config

```python
# Before
config = {
    'database_url': os.getenv('DATABASE_URL', 'mongodb://localhost'),
    'debug': os.getenv('DEBUG', 'false').lower() == 'true'
}

# After
settings = XSettings.get_default()
# Use settings.database_url, settings.debug
```

### From Other Config Libraries

```python
# Before (python-decouple)
from decouple import config
DATABASE_URL = config('DATABASE_URL')

# After
from core.settings import XSettings
settings = XSettings.get_default()
DATABASE_URL = settings.database_url
```

## Future Enhancements

1. **Namespace Integration**

   - When ns feature is ready, integrate for resource access
   - `settings.ns.mongodb` for ns-aware settings

2. **Dynamic Reloading**

   - Watch config files for changes
   - Reload without restart

3. **Encryption**

   - Encrypt sensitive values at rest
   - Decrypt on load

4. **Remote Configuration**

   - Support loading from config servers
   - Consul, etcd integration

5. **Audit Trail**
   - Log all configuration changes
   - Track who changed what and when

## Summary

XSettings provides a robust, flexible configuration management system that:

- Integrates with DynaConf for powerful configuration loading
- Uses XObjPrototype for validation and type safety
- Employs nested Pydantic models for logical grouping of settings
- Supports multiple environments with easy switching
- Provides flexible validation strategies
- Offers both singleton and DI patterns
- Enables deep nested access (e.g., `settings.database.url`)
- Supports dotted-path overrides (e.g., `database.url='...'`)
- Enables multi-environment loading for import/export scenarios
- Validates nested structures with full Pydantic support

The design uses nested models to organize settings into logical categories (database, api, cache, etc.) while maintaining the flexibility and power of DynaConf for configuration loading. This approach provides:

- Better organization and discoverability of settings
- Type safety at all nesting levels
- Clear separation of concerns
- Easy validation of complex configurations
- Intuitive access patterns that match the configuration structure
