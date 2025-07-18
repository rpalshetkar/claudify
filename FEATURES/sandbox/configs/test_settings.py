"""Test settings configuration."""

from typing import Any

# Test database configuration
TEST_DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "xarch_test",
    "user": "test_user",
    "password": "test_password",
    "min_size": 1,
    "max_size": 5,
}

# Test Redis configuration
TEST_REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 15,  # Use a separate db for tests
    "password": None,
}

# Test API configuration
TEST_API_CONFIG = {
    "base_url": "http://localhost:8000",
    "timeout": 30,
    "max_retries": 3,
    "api_key": "test_api_key",
}

# Test file paths
TEST_FILE_PATHS = {
    "csv_file": "test_data.csv",
    "json_file": "test_data.json",
    "excel_file": "test_data.xlsx",
    "temp_dir": "/tmp/xarch_test",  # noqa: S108
}

# Test logging configuration
TEST_LOGGING_CONFIG = {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
        },
    },
}

# Test environment variables
TEST_ENV_VARS = {
    "ENVIRONMENT": "test",
    "DEBUG": "true",
    "TESTING": "true",
    "DATABASE_URL": "postgresql://test_user:test_password@localhost:5432/xarch_test",
    "REDIS_URL": "redis://localhost:6379/15",
}

# Test data samples
TEST_SAMPLE_DATA = {
    "users": [
        {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
            "active": True,
            "metadata": {"role": "admin"},
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "email": "jane@example.com",
            "active": False,
            "metadata": {"role": "user"},
        },
    ],
    "products": [
        {
            "id": 1,
            "name": "Product A",
            "price": 99.99,
            "category": "electronics",
            "in_stock": True,
        },
        {
            "id": 2,
            "name": "Product B",
            "price": 149.99,
            "category": "books",
            "in_stock": False,
        },
    ],
}

# Test schema definitions
TEST_SCHEMAS = {
    "user_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "active": {"type": "boolean"},
            "metadata": {"type": "object"},
        },
        "required": ["id", "name", "email"],
    },
    "product_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "price": {"type": "number"},
            "category": {"type": "string"},
            "in_stock": {"type": "boolean"},
        },
        "required": ["id", "name", "price"],
    },
}

# Test connection strings
TEST_CONNECTION_STRINGS = {
    "postgresql": "postgresql://test_user:test_password@localhost:5432/xarch_test",
    "mysql": "mysql://test_user:test_password@localhost:3306/xarch_test",
    "sqlite": "sqlite:///test_xarch.db",
    "mongodb": "mongodb://test_user:test_password@localhost:27017/xarch_test",
    "redis": "redis://localhost:6379/15",
}

# Test namespaces
TEST_NAMESPACES = {
    "models": "ns.test.models",
    "repos": "ns.test.repos",
    "resources": "ns.test.resources",
    "cache": "ns.test.cache",
    "fuzzy": "ns.test.fuzzy",
}

# Test permissions
TEST_PERMISSIONS = {
    "admin": ["create", "read", "update", "delete"],
    "user": ["read", "update"],
    "guest": ["read"],
}

# Test metadata templates
TEST_METADATA_TEMPLATES = {
    "basic": {
        "created_at": "2024-01-01T00:00:00Z",
        "created_by": "test_user",
        "version": 1,
    },
    "advanced": {
        "created_at": "2024-01-01T00:00:00Z",
        "created_by": "test_user",
        "version": 1,
        "tags": ["test", "sample"],
        "category": "test_data",
        "description": "Test metadata for unit tests",
    },
}


def get_test_config(config_type: str) -> dict[str, Any]:
    """Get test configuration by type."""
    configs: dict[str, dict[str, Any]] = {
        "database": TEST_DATABASE_CONFIG,
        "redis": TEST_REDIS_CONFIG,
        "api": TEST_API_CONFIG,
        "logging": TEST_LOGGING_CONFIG,
    }
    return configs.get(config_type, {})


def get_test_data(data_type: str) -> Any:
    """Get test data by type."""
    return TEST_SAMPLE_DATA.get(data_type, [])


def get_test_schema(schema_name: str) -> dict[str, Any]:
    """Get test schema by name."""
    return TEST_SCHEMAS.get(schema_name, {})


def get_connection_string(db_type: str) -> str:
    """Get test connection string by database type."""
    return TEST_CONNECTION_STRINGS.get(db_type, "")


def get_test_namespace(namespace_type: str) -> str:
    """Get test namespace by type."""
    return TEST_NAMESPACES.get(namespace_type, "ns.test.default")
