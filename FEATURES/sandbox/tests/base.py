"""Base test classes and utilities for XArchitecture testing."""

import asyncio
from abc import abstractmethod
from typing import Any, Protocol, TypeVar
from unittest.mock import MagicMock, Mock

import pytest
from faker import Faker
from pydantic import BaseModel
from rich.console import Console

fake = Faker()
console = Console()
T = TypeVar("T", bound=BaseModel)


class FactoryProtocol(Protocol):
    """Protocol for factory classes."""

    @classmethod
    def create(cls, **kwargs: Any) -> Any:
        """Create an instance."""
        ...

    @classmethod
    def create_batch(cls, size: int, **kwargs: Any) -> list[Any]:
        """Create multiple instances."""
        ...


class BaseTestCase:
    """Base test case with common utilities."""

    def setup_method(self) -> None:
        """Set up test method."""
        self.faker = Faker()
        self.mocks: list[Mock] = []

    def teardown_method(self) -> None:
        """Clean up after test method."""
        for mock in self.mocks:
            mock.reset_mock()

    def create_mock(self, spec: type | None = None) -> Mock:
        """Create a mock and track it for cleanup."""
        mock = Mock(spec=spec)
        self.mocks.append(mock)
        return mock

    def create_magic_mock(self, spec: type | None = None) -> MagicMock:
        """Create a magic mock and track it for cleanup."""
        mock = MagicMock(spec=spec)
        self.mocks.append(mock)
        return mock


class AsyncTestCase:
    """Base test case for async operations."""

    def __init__(self) -> None:
        """Initialize async test case."""
        self.faker = Faker()
        self.mocks: list[Mock] = []

    async def setup_method(self) -> None:
        """Set up async test method."""
        self.faker = Faker()
        self.mocks = []
        self.loop = asyncio.get_event_loop()

    async def teardown_method(self) -> None:
        """Clean up after async test method."""
        for mock in self.mocks:
            mock.reset_mock()
        # Cancel any pending tasks except the current one
        current_task = asyncio.current_task()
        pending = asyncio.all_tasks(self.loop)
        tasks_to_cancel = [task for task in pending if task != current_task]
        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    def create_mock(self, spec: type | None = None) -> Mock:
        """Create a mock and track it for cleanup."""
        mock = Mock(spec=spec)
        self.mocks.append(mock)
        return mock

    def create_magic_mock(self, spec: type | None = None) -> MagicMock:
        """Create a magic mock and track it for cleanup."""
        mock = MagicMock(spec=spec)
        self.mocks.append(mock)
        return mock


class ModelTestCase(BaseTestCase):
    """Base test case for model testing."""

    @abstractmethod
    def get_model_class(self) -> type[BaseModel]:
        """Return the model class to test."""
        pass

    @abstractmethod
    def get_valid_data(self) -> dict[str, Any]:
        """Return valid data for model creation."""
        pass

    @abstractmethod
    def get_invalid_data(self) -> dict[str, Any]:
        """Return invalid data for model creation."""
        pass

    def create_model(self, **kwargs: Any) -> BaseModel:
        """Create a model instance with valid data."""
        data = self.get_valid_data()
        data.update(kwargs)
        return self.get_model_class()(**data)

    def test_model_creation_valid(self) -> None:
        """Test model creation with valid data."""
        model = self.create_model()
        assert isinstance(model, self.get_model_class())

    def test_model_creation_invalid(self) -> None:
        """Test model creation with invalid data."""
        with pytest.raises(ValueError):
            self.get_model_class()(**self.get_invalid_data())

    def test_model_serialization(self) -> None:
        """Test model serialization."""
        model = self.create_model()
        data = model.model_dump()
        assert isinstance(data, dict)

        # Test deserialization
        recreated = self.get_model_class()(**data)
        assert recreated == model


class FactoryTestCase(BaseTestCase):
    """Base test case for factory testing."""

    @abstractmethod
    def get_factory_class(self) -> type[FactoryProtocol]:
        """Return the factory class to test."""
        pass

    def test_factory_creation(self) -> None:
        """Test factory instance creation."""
        factory_class = self.get_factory_class()
        instance = factory_class.create()
        assert instance is not None

    def test_factory_batch_creation(self) -> None:
        """Test factory batch creation."""
        factory_class = self.get_factory_class()
        instances = factory_class.create_batch(5)
        assert len(instances) == 5
        assert all(instance is not None for instance in instances)


class DatabaseTestCase(AsyncTestCase):
    """Base test case for database operations."""

    async def setup_method(self) -> None:
        """Set up database test method."""
        await super().setup_method()
        # Setup test database connection
        self.db_connection = self.create_mock()
        self.db_connection.execute = self.create_mock()
        self.db_connection.fetch = self.create_mock()
        self.db_connection.fetchrow = self.create_mock()

    async def teardown_method(self) -> None:
        """Clean up database test method."""
        if hasattr(self, "db_connection") and hasattr(self.db_connection, "close"):
            await self.db_connection.close()
        await super().teardown_method()


class APITestCase(AsyncTestCase):
    """Base test case for API testing."""

    async def setup_method(self) -> None:
        """Set up API test method."""
        await super().setup_method()
        self.client = self.create_mock()
        self.client.get = self.create_mock()
        self.client.post = self.create_mock()
        self.client.put = self.create_mock()
        self.client.delete = self.create_mock()


def assert_model_equal(model1: BaseModel, model2: BaseModel) -> None:
    """Assert two models are equal."""
    assert type(model1) is type(model2)
    assert model1.model_dump() == model2.model_dump()


def assert_model_fields_equal(
    model1: BaseModel, model2: BaseModel, fields: list[str]
) -> None:
    """Assert specific fields of two models are equal."""
    data1 = model1.model_dump()
    data2 = model2.model_dump()

    for field in fields:
        assert data1.get(field) == data2.get(field), f"Field '{field}' differs"


def create_test_data[T: BaseModel](
    model_class: type[T], count: int = 1, **kwargs: Any
) -> list[T]:
    """Create test data for a model class."""
    instances = []
    for i in range(count):
        data = {
            "id": i + 1,
            "name": fake.name(),
            **kwargs,
        }
        instances.append(model_class(**data))
    return instances
