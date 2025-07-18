"""Example test showing Rich console usage."""

from typing import Any

import pytest
from icecream import ic
from pydantic import BaseModel, ValidationError
from sandbox.utils.output import console, debug, error, info, success, table


class User(BaseModel):
    """Sample user model."""

    id: int
    name: str
    email: str
    active: bool = True


class TestRichOutput:
    """Demonstrate Rich output in tests."""

    def test_model_creation_with_output(self) -> None:
        """Test model creation with Rich output."""
        info("Creating user model...")

        user_data: dict[str, Any] = {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
        }

        # Debug the input data
        debug(user_data, "Input data")

        user = User(**user_data)

        # Use icecream to inspect the model
        ic(user)
        ic(user.model_dump())

        success("User model created successfully")

        assert user.id == 1
        assert user.active is True

    def test_validation_error_display(self) -> None:
        """Test validation error with Rich display."""
        info("Testing validation error handling...")

        invalid_data: dict[str, Any] = {
            "id": "not_a_number",  # Invalid type
            "name": "Jane",
            # Missing required email
        }

        with pytest.raises(ValidationError) as exc_info:
            User(**invalid_data)

        error("Validation failed as expected")

        # Display validation errors in a nice format
        errors = exc_info.value.errors()
        error_table = [
            {
                "field": err["loc"][0],
                "type": err["type"],
                "message": err["msg"],
            }
            for err in errors
        ]

        table(error_table, title="Validation Errors")

    def test_batch_operations(self) -> None:
        """Test batch operations with progress display."""
        info("Running batch user creation...")

        users = []
        for i in range(5):
            user = User(
                id=i + 1,
                name=f"User {i + 1}",
                email=f"user{i + 1}@example.com",
                active=i % 2 == 0,
            )
            users.append(user)
            console.print(f"  → Created {user.name}", style="dim")

        success(f"Created {len(users)} users")

        # Display summary
        active_count = sum(1 for u in users if u.active)
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Total users: {len(users)}")
        console.print(f"  Active: [green]{active_count}[/green]")
        console.print(f"  Inactive: [red]{len(users) - active_count}[/red]")

        assert len(users) == 5
        assert active_count == 3

    @pytest.mark.parametrize(
        "user_id,expected_status",
        [(1, "active"), (2, "inactive"), (3, "active")],
    )
    def test_parametrized_with_output(self, user_id: int, expected_status: str) -> None:
        """Test parametrized test with output."""
        console.print(
            f"\n[cyan]Testing user {user_id} - expecting {expected_status}[/cyan]"
        )

        user = User(
            id=user_id,
            name=f"Test User {user_id}",
            email=f"test{user_id}@example.com",
            active=(expected_status == "active"),
        )

        # Debug the created user
        ic(user.model_dump())

        if user.active:
            success(f"User {user_id} is active✅")
        else:
            console.print(f"User {user_id} is inactive ⏸", style="yellow")

        assert user.active == (expected_status == "active")
