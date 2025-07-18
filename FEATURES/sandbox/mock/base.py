"""Base mock factories for XArchitecture components."""

from datetime import datetime
from typing import Any, TypeVar
from unittest.mock import AsyncMock, Mock

import factory
from faker import Faker
from pydantic import BaseModel
from rich.console import Console

fake = Faker()
console = Console()
T = TypeVar("T", bound=BaseModel)


class BaseModelFactory(factory.Factory):
    """Base factory for Pydantic models."""

    class Meta:
        abstract = True

    id = factory.Sequence(lambda n: n)
    created_at = factory.LazyFunction(datetime.now)
    updated_at = factory.LazyFunction(datetime.now)

    @classmethod
    def _create(cls, model_class: type[T], *args: Any, **kwargs: Any) -> T:
        """Create model instance using Pydantic."""
        return model_class(**kwargs)


class MockConnectionFactory:
    """Factory for creating mock database connections."""

    @staticmethod
    def create_mock_connection() -> Mock:
        """Create a mock database connection."""
        mock = Mock()
        mock.execute = AsyncMock(return_value=None)
        mock.fetch = AsyncMock(return_value=[])
        mock.fetchrow = AsyncMock(return_value=None)
        mock.fetchval = AsyncMock(return_value=None)
        mock.close = AsyncMock()
        mock.transaction = AsyncMock()
        return mock

    @staticmethod
    def create_mock_pool() -> Mock:
        """Create a mock connection pool."""
        mock = Mock()
        mock.acquire = AsyncMock(
            return_value=MockConnectionFactory.create_mock_connection()
        )
        mock.release = AsyncMock()
        mock.close = AsyncMock()
        return mock


class MockResourceFactory:
    """Factory for creating mock resources."""

    @staticmethod
    def create_file_resource(file_type: str = "csv") -> Mock:
        """Create a mock file resource."""
        mock = Mock()
        mock.resource_type = "file"
        mock.file_type = file_type
        mock.connection_params = {
            "file_path": fake.file_path(extension=file_type),
            "encoding": "utf-8",
        }
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.read = AsyncMock(return_value=[])
        mock.write = AsyncMock()
        return mock

    @staticmethod
    def create_database_resource(db_type: str = "postgresql") -> Mock:
        """Create a mock database resource."""
        mock = Mock()
        mock.resource_type = "database"
        mock.db_type = db_type
        mock.connection_params = {
            "host": fake.ipv4(),
            "port": fake.port_number(),
            "database": fake.word(),
            "user": fake.user_name(),
            "password": fake.password(),
        }
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.execute = AsyncMock()
        mock.fetch = AsyncMock(return_value=[])
        mock.pool = MockConnectionFactory.create_mock_pool()
        return mock

    @staticmethod
    def create_api_resource() -> Mock:
        """Create a mock API resource."""
        mock = Mock()
        mock.resource_type = "api"
        mock.connection_params = {
            "base_url": fake.url(),
            "api_key": fake.uuid4(),
            "timeout": 30,
        }
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.get = AsyncMock(return_value={})
        mock.post = AsyncMock(return_value={})
        mock.put = AsyncMock(return_value={})
        mock.delete = AsyncMock(return_value={})
        return mock


class MockInspectorFactory:
    """Factory for creating mock inspectors."""

    @staticmethod
    def create_inspector() -> Mock:
        """Create a mock inspector."""
        mock = Mock()
        mock.inspect_schema = AsyncMock(
            return_value={
                "columns": [
                    {"name": "id", "type": "integer", "nullable": False},
                    {"name": "name", "type": "string", "nullable": False},
                    {"name": "active", "type": "boolean", "nullable": True},
                ],
                "primary_key": ["id"],
                "indexes": [],
            }
        )
        mock.profile_data = AsyncMock(
            return_value={
                "row_count": fake.random_int(100, 10000),
                "column_stats": {},
                "data_types": {},
            }
        )
        mock.generate_model = Mock(
            return_value=type(
                "TestModel",
                (BaseModel,),
                {
                    "id": int,
                    "name": str,
                    "active": bool,
                },
            )
        )
        mock.get_preview = AsyncMock(return_value=[])
        return mock


class MockRepositoryFactory:
    """Factory for creating mock repositories."""

    @staticmethod
    def create_connected_repo() -> Mock:
        """Create a mock connected repository."""
        mock = Mock()
        mock.create = AsyncMock(return_value={})
        mock.read = AsyncMock(return_value={})
        mock.update = AsyncMock(return_value={})
        mock.delete = AsyncMock(return_value=True)
        mock.query = AsyncMock(return_value=[])
        mock.count = AsyncMock(return_value=0)
        mock.exists = AsyncMock(return_value=False)
        mock.transaction = AsyncMock()
        return mock

    @staticmethod
    def create_materialized_repo() -> Mock:
        """Create a mock materialized repository."""
        mock = Mock()
        mock.data = []
        mock.create = AsyncMock(return_value={})
        mock.read = AsyncMock(return_value={})
        mock.update = AsyncMock(return_value={})
        mock.delete = AsyncMock(return_value=True)
        mock.query = AsyncMock(return_value=[])
        mock.refresh = AsyncMock()
        mock.sync = AsyncMock()
        return mock


class MockRegistryFactory:
    """Factory for creating mock registry components."""

    @staticmethod
    def create_registry() -> Mock:
        """Create a mock registry."""
        mock = Mock()
        mock.registry = {}
        mock.register_model = Mock(return_value=True)
        mock.get_model = Mock(return_value=None)
        mock.list_models = Mock(return_value=[])
        mock.deregister_model = Mock(return_value=True)
        mock.get_widget = Mock(return_value="text")
        mock.check_permission = Mock(return_value=True)
        return mock


class MockCacheFactory:
    """Factory for creating mock cache components."""

    @staticmethod
    def create_cache() -> Mock:
        """Create a mock cache."""
        mock = Mock()
        mock.storage = {}
        mock.register_ns = Mock(return_value=True)
        mock.get_ns = Mock(return_value=None)
        mock.list_ns = Mock(return_value=[])
        mock.fuzzy_search = Mock(return_value=[])
        mock.clear = Mock()
        return mock


class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_sample_data(count: int = 10) -> list[dict[str, Any]]:
        """Create sample data for testing."""
        return [
            {
                "id": i,
                "name": fake.name(),
                "email": fake.email(),
                "active": fake.boolean(),
                "created_at": fake.date_time(),
                "metadata": {
                    "tags": [fake.word() for _ in range(fake.random_int(1, 3))],
                    "category": fake.word(),
                },
            }
            for i in range(1, count + 1)
        ]

    @staticmethod
    def create_csv_data() -> str:
        """Create CSV data for testing."""
        header = "id,name,email,active\n"
        rows = []
        for i in range(1, 11):
            rows.append(f"{i},{fake.name()},{fake.email()},{fake.boolean()}")
        return header + "\n".join(rows)

    @staticmethod
    def create_json_data() -> list[dict[str, Any]]:
        """Create JSON data for testing."""
        return TestDataFactory.create_sample_data()


def create_async_mock(*args: Any, **kwargs: Any) -> AsyncMock:
    """Create an async mock with proper return values."""
    mock = AsyncMock(*args, **kwargs)

    # Common async methods
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.execute = AsyncMock()
    mock.fetch = AsyncMock(return_value=[])
    mock.create = AsyncMock(return_value={})
    mock.read = AsyncMock(return_value={})
    mock.update = AsyncMock(return_value={})
    mock.delete = AsyncMock(return_value=True)

    return mock
