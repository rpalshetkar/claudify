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

### 6. Access Pattern: Hybrid with Flat Structure (Option C)
- Default singleton for convenience
- Explicit DI support for testing
- Flat field structure (no nested categories)

## Architecture

### Class Structure

```python
from __future__ import annotations

from typing import Literal, Any
from pydantic import Field, field_validator
from dynaconf import Dynaconf
import os

from core.base import XObjPrototype

class XSettings(XObjPrototype):
    """
    Settings management with DynaConf integration.
    
    Provides environment-aware configuration with validation,
    configurable loading order, and multi-environment support.
    """
    
    # Flat structure for all settings
    database_url: str = Field(description='Database connection URL')
    database_pool_size: int = Field(default=10, description='Database connection pool size')
    api_key: str | None = Field(default=None, description='External API key')
    api_timeout: int = Field(default=30, description='API request timeout in seconds')
    cache_enabled: bool = Field(default=True, description='Enable caching')
    cache_ttl: int = Field(default=3600, description='Cache TTL in seconds')
    log_level: str = Field(default='info', description='Logging level')
    debug: bool = Field(default=False, description='Debug mode')
    
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
            **kwargs: Additional settings overrides
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
        
        # Apply any kwargs overrides
        for key, value in kwargs.items():
            setattr(self, key, value)
        
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
DATABASE_URL=mongodb://localhost:27017/myapp
DATABASE_POOL_SIZE=10

# API Configuration  
API_KEY=your-api-key-here
API_TIMEOUT=30

# Security
SECRET_KEY=change-this-in-production
JWT_SECRET=change-this-in-production

# Basic Settings
LOG_LEVEL=info
DEBUG=false
```

### settings.toml Example

```toml
[default]
# Complex structures and app behavior
cache_enabled = true
cache_ttl = 3600

# Feature flags
available_exporters = ["json", "csv", "xml"]
supported_languages = ["en", "es", "fr"]

# Rate limiting
rate_limit_requests = 100
rate_limit_window = 60

[development]
debug = true
log_level = "debug"
database_url = "mongodb://localhost:27017/myapp_dev"

[production]
debug = false
log_level = "warning"
cache_ttl = 7200

[test]
database_url = "mongodb://localhost:27017/myapp_test"
cache_enabled = false
```

## Usage Examples

### Basic Usage

```python
from core.settings import XSettings

# Using default singleton
settings = XSettings.get_default()
print(settings.database_url)
print(settings.api_timeout)

# Creating specific instance
dev_settings = XSettings(env='development')
prod_settings = XSettings(env='production')
```

### Testing Usage

```python
import pytest
from core.settings import XSettings

def test_with_custom_settings():
    # Create test-specific settings
    test_settings = XSettings(
        env='test',
        database_url='mongodb://testdb:27017/test',
        cache_enabled=False
    )
    
    # Use in test
    assert test_settings.cache_enabled is False
    assert 'test' in test_settings.database_url

@pytest.fixture
def settings():
    """Fixture for test settings."""
    return XSettings(env='test', validation_mode='eager')
```

### Dependency Injection Usage

```python
from dependency_injector import containers, providers
from core.settings import XSettings

class Container(containers.DeclarativeContainer):
    config = providers.Singleton(XSettings)
    
    database = providers.Singleton(
        MongoDB,
        url=config.provided.database_url,
        pool_size=config.provided.database_pool_size,
    )

# In services
class UserService:
    def __init__(self, settings: XSettings):
        self.settings = settings
        self.timeout = settings.api_timeout
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
    test_db_url = settings.get_env_value('database_url', env='test')
    test_db = MongoDB(test_db_url)
    
    # Production is primary environment
    prod_db = MongoDB(settings.database_url)
    
    # Migrate data
    data = test_db.export_all()
    prod_db.import_data(data)

# Using lazy environment access
def compare_settings():
    test_cache = settings.envs.test['cache_ttl']
    prod_cache = settings.envs.production['cache_ttl']
    
    print(f"Test cache TTL: {test_cache}")
    print(f"Prod cache TTL: {prod_cache}")
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
    
    def test_environment_override(self):
        settings = XSettings(env='production')
        assert settings.environment == 'production'
        assert settings.debug is False
    
    def test_validation_modes(self):
        # Eager validation
        with pytest.raises(ValidationError):
            XSettings(
                validation_mode='eager',
                database_url=None  # Required field
            )
        
        # Lazy validation - no error on init
        settings = XSettings(
            validation_mode='lazy',
            database_url=None
        )
        # Error on access
        with pytest.raises(ValidationError):
            _ = settings.database_url
    
    def test_loading_order(self):
        # Only load from env
        os.environ['PROJECT_DATABASE_URL'] = 'env-db-url'
        settings = XSettings(loading_order=['env'])
        assert settings.database_url == 'env-db-url'
    
    def test_multi_environment_loading(self):
        settings = XSettings(
            env='development',
            load_envs=['test', 'production']
        )
        
        dev_db = settings.database_url
        test_db = settings.get_env_value('database_url', 'test')
        prod_db = settings.get_env_value('database_url', 'production')
        
        assert 'dev' in dev_db
        assert 'test' in test_db
        assert test_db != prod_db
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
   - When namespace feature is ready, integrate for resource access
   - `settings.ns.mongodb` for namespace-aware settings

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
- Supports multiple environments with easy switching
- Provides flexible validation strategies
- Offers both singleton and DI patterns
- Maintains flat structure for simplicity
- Enables multi-environment loading for import/export scenarios

The design balances simplicity for common cases with flexibility for complex requirements, following all project standards and architectural principles.