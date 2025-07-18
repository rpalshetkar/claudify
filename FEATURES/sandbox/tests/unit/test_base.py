"""Test base utilities and fixtures."""

import asyncio
from datetime import datetime
from typing import Any, cast

import pytest
from pydantic import BaseModel
from sandbox.tests.base import (
    APITestCase,
    AsyncTestCase,
    BaseTestCase,
    DatabaseTestCase,
    ModelTestCase,
    assert_model_equal,
    create_test_data,
)


class SampleModel(BaseModel):
    """Sample model for testing."""

    id: int
    name: str
    active: bool = True
    metadata: dict[str, Any] = {}


class TestBaseTestCase:
    """Test BaseTestCase functionality."""

    def test_setup_method(self) -> None:
        """Test setup method creates required attributes."""
        test_case = BaseTestCase()
        test_case.setup_method()

        assert hasattr(test_case, "faker")
        assert hasattr(test_case, "mocks")
        assert isinstance(test_case.mocks, list)

    def test_create_mock(self) -> None:
        """Test mock creation and tracking."""
        test_case = BaseTestCase()
        test_case.setup_method()

        mock = test_case.create_mock()
        assert mock is not None
        assert mock in test_case.mocks

    def test_teardown_method(self) -> None:
        """Test teardown method resets mocks."""
        test_case = BaseTestCase()
        test_case.setup_method()

        mock = test_case.create_mock()
        mock.some_method = lambda: "test"

        test_case.teardown_method()
        # Mock should be reset but still tracked
        assert mock in test_case.mocks


class TestAsyncTestCase:
    """Test AsyncTestCase functionality."""

    @pytest.mark.asyncio
    async def test_setup_method(self) -> None:
        """Test async setup method."""
        test_case = AsyncTestCase()
        await test_case.setup_method()

        assert hasattr(test_case, "loop")
        assert isinstance(test_case.loop, asyncio.AbstractEventLoop)

    @pytest.mark.asyncio
    async def test_teardown_method(self) -> None:
        """Test async teardown method."""
        test_case = AsyncTestCase()
        await test_case.setup_method()

        # Create a task that will be cancelled
        async def dummy_task() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(dummy_task())

        await test_case.teardown_method()

        # Task should be cancelled
        assert task.cancelled()


class TestSampleModelTestCase(ModelTestCase):
    """Test ModelTestCase with sample model."""

    def get_model_class(self) -> type[BaseModel]:
        """Return sample model class."""
        return SampleModel

    def get_valid_data(self) -> dict[str, Any]:
        """Return valid sample data."""
        return {
            "id": 1,
            "name": "Test Sample",
            "active": True,
            "metadata": {"test": True},
        }

    def get_invalid_data(self) -> dict[str, Any]:
        """Return invalid sample data."""
        return {
            "id": "invalid",  # Should be int
            "name": None,  # Should be str
        }

    def test_create_model_with_kwargs(self) -> None:
        """Test model creation with custom kwargs."""
        model = self.create_model(name="Custom Name")
        sample_model = cast(SampleModel, model)
        assert sample_model.name == "Custom Name"
        assert sample_model.id == 1  # From valid data


class TestDatabaseTestCase:
    """Test DatabaseTestCase functionality."""

    @pytest.mark.asyncio
    async def test_setup_creates_db_connection(self) -> None:
        """Test database test case setup."""
        test_case = DatabaseTestCase()
        await test_case.setup_method()

        assert hasattr(test_case, "db_connection")
        assert hasattr(test_case.db_connection, "execute")
        assert hasattr(test_case.db_connection, "fetch")
        assert hasattr(test_case.db_connection, "fetchrow")


class TestAPITestCase:
    """Test APITestCase functionality."""

    @pytest.mark.asyncio
    async def test_setup_creates_client(self) -> None:
        """Test API test case setup."""
        test_case = APITestCase()
        await test_case.setup_method()

        assert hasattr(test_case, "client")
        assert hasattr(test_case.client, "get")
        assert hasattr(test_case.client, "post")
        assert hasattr(test_case.client, "put")
        assert hasattr(test_case.client, "delete")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_assert_model_equal(self) -> None:
        """Test model equality assertion."""
        model1 = SampleModel(id=1, name="Test")
        model2 = SampleModel(id=1, name="Test")

        # Should not raise
        assert_model_equal(model1, model2)

        # Should raise for different models
        model3 = SampleModel(id=2, name="Different")
        with pytest.raises(AssertionError):
            assert_model_equal(model1, model3)

    def test_create_test_data(self) -> None:
        """Test test data creation."""
        data = create_test_data(SampleModel, count=3)

        assert len(data) == 3
        assert all(isinstance(item, SampleModel) for item in data)
        assert all(item.active for item in data)  # Default value

    def test_create_test_data_with_kwargs(self) -> None:
        """Test test data creation with custom kwargs."""
        data = create_test_data(SampleModel, count=2, active=False)

        assert len(data) == 2
        assert all(not item.active for item in data)


class TestFixtures:
    """Test global fixtures."""

    def test_sample_metadata_fixture(self, sample_metadata: dict[str, Any]) -> None:
        """Test sample metadata fixture."""
        assert "created_at" in sample_metadata
        assert "created_by" in sample_metadata
        assert "tags" in sample_metadata
        assert "version" in sample_metadata
        assert "description" in sample_metadata

        assert isinstance(sample_metadata["created_at"], datetime)
        assert isinstance(sample_metadata["created_by"], str)
        assert isinstance(sample_metadata["tags"], list)
        assert isinstance(sample_metadata["version"], int)
        assert isinstance(sample_metadata["description"], str)

    def test_sample_namespace_fixture(self, sample_namespace: str) -> None:
        """Test sample namespace fixture."""
        assert sample_namespace.startswith("ns.test.")
        assert len(sample_namespace.split(".")) == 4

    def test_test_model_fixture(self, test_model: Any) -> None:
        """Test test model fixture."""
        assert hasattr(test_model, "id")
        assert hasattr(test_model, "name")
        assert hasattr(test_model, "active")
        assert hasattr(test_model, "metadata")

    def test_test_models_fixture(self, test_models: list[Any]) -> None:
        """Test test models fixture."""
        assert len(test_models) == 5
        assert all(hasattr(model, "id") for model in test_models)
        assert all(hasattr(model, "name") for model in test_models)
