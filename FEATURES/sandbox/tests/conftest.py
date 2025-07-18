"""Global test configuration and fixtures."""

import asyncio
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from faker import Faker
from icecream import ic
from pydantic import BaseModel
from rich.console import Console

# Configure icecream
ic.configureOutput(prefix="🍦 |> ")

fake = Faker()
console = Console()


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Sample metadata for testing."""
    return {
        "created_at": fake.date_time(),
        "created_by": fake.user_name(),
        "tags": [fake.word() for _ in range(3)],
        "version": fake.random_int(1, 10),
        "description": fake.text(max_nb_chars=100),
    }


@pytest.fixture
def sample_namespace() -> str:
    """Sample namespace for testing."""
    return f"ns.test.{fake.word()}.{fake.word()}"


class TestModel(BaseModel):
    """Simple test model for base testing."""

    id: int
    name: str
    active: bool = True
    metadata: dict[str, Any] = {}


@pytest.fixture
def test_model() -> TestModel:
    """Create a test model instance."""
    return TestModel(
        id=fake.random_int(1, 1000),
        name=fake.name(),
        active=fake.boolean(),
        metadata={
            "created_at": fake.date_time().isoformat(),
            "tags": [fake.word() for _ in range(2)],
        },
    )


@pytest.fixture
def test_models(test_model: TestModel) -> list[TestModel]:
    """Create multiple test model instances."""
    models = [test_model]
    for _ in range(4):
        models.append(
            TestModel(
                id=fake.random_int(1, 1000),
                name=fake.name(),
                active=fake.boolean(),
            )
        )
    return models
