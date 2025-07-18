"""Mock factories for XObjPrototype models."""

from typing import Any

import factory
from faker import Faker

from mock.base import BaseModelFactory
from xobj_prototype import UserModel, ProductModel, ConfigModel, OrderModel

fake = Faker()


class XObjPrototypeFactory(BaseModelFactory):
    """Factory for creating mock XObjPrototype instances."""

    class Meta:
        abstract = True

    # Metadata factory
    _metadata = factory.Dict({})

    @classmethod
    def add_metadata(cls, **kwargs: Any) -> dict[str, Any]:
        """Helper to add metadata to factory."""
        return {"_metadata": kwargs}


class UserModelFactory(XObjPrototypeFactory):
    """Factory for creating User model instances."""

    id = factory.LazyFunction(lambda: f"user-{fake.uuid4()}")
    name = factory.Faker("name")
    email = factory.Faker("email")
    age = factory.Faker("random_int", min=18, max=80)
    role = factory.Faker("random_element", elements=["user", "admin", "moderator"])

    class Meta:
        model = UserModel


class ProductModelFactory(XObjPrototypeFactory):
    """Factory for creating Product model instances."""

    id = factory.LazyFunction(lambda: f"prod-{fake.uuid4()}")
    name = factory.Faker("word")
    sku = factory.LazyFunction(lambda: f"SKU-{fake.random_int(1000, 9999)}")
    price = factory.Faker("pydecimal", left_digits=4, right_digits=2, positive=True)
    category = factory.Faker("random_element", elements=["electronics", "clothing", "food", "books"])
    active = factory.Faker("boolean", chance_of_getting_true=75)

    class Meta:
        model = ProductModel


class ConfigModelFactory(XObjPrototypeFactory):
    """Factory for creating Configuration model instances."""

    id = factory.LazyFunction(lambda: f"config-{fake.uuid4()}")
    key = factory.Faker("word")
    value = factory.Faker("sentence")
    environment = factory.Faker("random_element", elements=["dev", "staging", "prod"])
    enabled = factory.Faker("boolean")

    class Meta:
        model = ConfigModel


class OrderModelFactory(XObjPrototypeFactory):
    """Factory for creating Order model instances."""

    id = factory.LazyFunction(lambda: f"order-{fake.uuid4()}")
    user_id = factory.LazyFunction(lambda: f"user-{fake.uuid4()}")
    customer_name = factory.Faker("name")
    customer_email = factory.Faker("email")
    total = factory.Faker("pydecimal", left_digits=4, right_digits=2, positive=True)
    status = factory.Faker("random_element", elements=["pending", "processing", "shipped", "delivered", "cancelled"])
    items = factory.LazyFunction(lambda: [])  # Empty list by default

    class Meta:
        model = OrderModel


class ValidationTestFactory:
    """Factory for creating test data for validation scenarios."""

    @staticmethod
    def valid_user_data() -> dict[str, Any]:
        """Create valid user data."""
        return {
            "id": f"user-{fake.uuid4()}",
            "name": fake.name(),
            "email": fake.email(),
            "age": fake.random_int(18, 80),
            "role": "user"
        }

    @staticmethod
    def invalid_user_data() -> list[dict[str, Any]]:
        """Create various invalid user data scenarios."""
        return [
            # Missing required field
            {"name": fake.name(), "email": fake.email()},
            # Invalid email
            {"id": "123", "name": fake.name(), "email": "not-an-email", "age": 25},
            # Invalid age
            {"id": "123", "name": fake.name(), "email": fake.email(), "age": -5},
            # Wrong type
            {"id": "123", "name": fake.name(), "email": fake.email(), "age": "not-a-number"},
        ]

    @staticmethod
    def edge_case_data() -> list[dict[str, Any]]:
        """Create edge case test data."""
        return [
            # Empty strings
            {"id": "", "name": "", "email": "test@example.com", "age": 25},
            # Very long strings
            {"id": "x" * 1000, "name": "y" * 1000, "email": "test@example.com", "age": 25},
            # Null values for optional fields
            {"id": "123", "name": "Test", "email": "test@example.com", "age": 25, "role": None},
            # Max/min values
            {"id": "123", "name": "Test", "email": "test@example.com", "age": 0},
            {"id": "123", "name": "Test", "email": "test@example.com", "age": 999},
        ]


class NamespaceTestFactory:
    """Factory for creating namespace test scenarios."""

    @staticmethod
    def create_namespace_hierarchy() -> dict[str, Any]:
        """Create a nested namespace structure for testing."""
        return {
            "users": {
                "admin": ["user-001", "user-002"],
                "regular": ["user-003", "user-004", "user-005"],
                "guest": ["user-006"]
            },
            "products": {
                "electronics": ["prod-001", "prod-002"],
                "clothing": ["prod-003", "prod-004"],
                "food": ["prod-005", "prod-006", "prod-007"]
            },
            "config": {
                "system": ["config-001", "config-002"],
                "user": ["config-003", "config-004"]
            }
        }

    @staticmethod
    def create_collision_scenarios() -> list[dict[str, Any]]:
        """Create namespace collision test scenarios."""
        return [
            {"namespace": "users.admin", "id": "user-001", "conflicts": True},
            {"namespace": "users.admin", "id": "user-999", "conflicts": False},
            {"namespace": "products.electronics", "id": "prod-001", "conflicts": True},
            {"namespace": "config.new", "id": "config-001", "conflicts": False},
        ]


class MetadataTestFactory:
    """Factory for creating metadata test scenarios."""

    @staticmethod
    def create_basic_metadata() -> dict[str, Any]:
        """Create basic metadata."""
        return {
            "created_by": fake.user_name(),
            "created_at": fake.date_time().isoformat(),
            "tags": [fake.word() for _ in range(3)],
            "version": fake.random_int(1, 10)
        }

    @staticmethod
    def create_complex_metadata() -> dict[str, Any]:
        """Create complex nested metadata."""
        return {
            "audit": {
                "created_by": fake.user_name(),
                "created_at": fake.date_time().isoformat(),
                "modified_by": fake.user_name(),
                "modified_at": fake.date_time().isoformat(),
            },
            "permissions": {
                "read": ["user", "admin"],
                "write": ["admin"],
                "delete": ["admin"]
            },
            "cache": {
                "ttl": fake.random_int(60, 3600),
                "key": fake.uuid4(),
                "invalidate_on": ["update", "delete"]
            },
            "custom": {
                "department": fake.company(),
                "priority": fake.random_int(1, 5),
                "notes": fake.text()
            }
        }


class FuzzySearchTestFactory:
    """Factory for creating fuzzy search test data."""

    @staticmethod
    def create_searchable_models() -> list[dict[str, Any]]:
        """Create models with fuzzy searchable fields."""
        return [
            {
                "id": "user-001",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "description": "Software engineer with 10 years experience"
            },
            {
                "id": "user-002",
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "description": "Project manager specializing in agile"
            },
            {
                "id": "user-003",
                "name": "Bob Johnson",
                "email": "bob.j@example.com",
                "description": "Senior developer and team lead"
            }
        ]

    @staticmethod
    def create_search_queries() -> list[dict[str, Any]]:
        """Create search query test cases."""
        return [
            {"query": "john", "expected_matches": ["user-001", "user-003"]},
            {"query": "doe", "expected_matches": ["user-001"]},
            {"query": "example.com", "expected_matches": ["user-001", "user-002", "user-003"]},
            {"query": "developer", "expected_matches": ["user-003"]},
            {"query": "manager", "expected_matches": ["user-002"]},
        ]


class MetadataGenerationFactory:
    """Enhanced metadata generation utilities for comprehensive testing."""

    @staticmethod
    def generate_validation_metadata() -> dict[str, Any]:
        """Generate metadata for field validation rules."""
        return {
            "validation_rules": {
                "email": {
                    "type": "email",
                    "required": True,
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "age": {
                    "type": "integer",
                    "min": 0,
                    "max": 150,
                    "required": True
                },
                "price": {
                    "type": "decimal",
                    "min": 0,
                    "precision": 2,
                    "required": True
                },
                "status": {
                    "type": "enum",
                    "values": ["pending", "processing", "shipped", "delivered", "cancelled"],
                    "default": "pending"
                }
            }
        }

    @staticmethod
    def generate_fuzzy_search_metadata() -> dict[str, Any]:
        """Generate metadata for fuzzy search configuration."""
        return {
            "fuzzy_config": {
                "enabled": True,
                "fields": ["name", "email", "description"],
                "algorithm": "levenshtein",
                "threshold": 0.8,
                "boost_fields": {
                    "name": 2.0,
                    "email": 1.5,
                    "description": 1.0
                },
                "synonyms": {
                    "developer": ["programmer", "coder", "engineer"],
                    "manager": ["lead", "supervisor", "coordinator"]
                }
            }
        }

    @staticmethod
    def generate_ui_metadata() -> dict[str, Any]:
        """Generate metadata for UI widget mapping."""
        return {
            "ui_widgets": {
                "email": "email_input",
                "age": "number_slider",
                "price": "currency_input",
                "status": "dropdown",
                "active": "toggle_switch",
                "description": "textarea",
                "created_at": "datetime_picker",
                "tags": "tag_input"
            },
            "ui_hints": {
                "email": {"placeholder": "user@example.com"},
                "age": {"min": 0, "max": 150, "step": 1},
                "price": {"currency": "USD", "decimal_places": 2},
                "status": {"color_map": {"pending": "yellow", "shipped": "blue", "delivered": "green"}}
            }
        }

    @staticmethod
    def generate_audit_metadata(user: str = None) -> dict[str, Any]:
        """Generate audit trail metadata."""
        if not user:
            user = fake.user_name()
        return {
            "audit": {
                "created_by": user,
                "created_at": fake.date_time().isoformat(),
                "modified_by": user,
                "modified_at": fake.date_time().isoformat(),
                "version": 1,
                "change_log": []
            }
        }

    @staticmethod
    def generate_permission_metadata(roles: list[str] = None) -> dict[str, Any]:
        """Generate permission metadata."""
        if not roles:
            roles = ["admin", "user"]
        return {
            "permissions": {
                "read": roles,
                "write": ["admin"],
                "delete": ["admin"],
                "update": ["admin", "owner"],
                "custom_actions": {
                    "approve": ["admin", "manager"],
                    "export": roles,
                    "audit": ["admin", "auditor"]
                }
            }
        }

    @staticmethod
    def generate_cache_metadata() -> dict[str, Any]:
        """Generate caching configuration metadata."""
        return {
            "cache": {
                "enabled": True,
                "ttl": fake.random_int(60, 3600),
                "strategy": fake.random_element(["lru", "lfu", "fifo"]),
                "invalidate_on": ["update", "delete"],
                "key_pattern": "{namespace}:{id}:{version}",
                "tags": [fake.word() for _ in range(3)]
            }
        }

    @staticmethod
    def generate_complete_metadata(model_type: str = "user") -> dict[str, Any]:
        """Generate complete metadata set for a model."""
        metadata = {}
        metadata.update(MetadataGenerationFactory.generate_validation_metadata())
        metadata.update(MetadataGenerationFactory.generate_fuzzy_search_metadata())
        metadata.update(MetadataGenerationFactory.generate_ui_metadata())
        metadata.update(MetadataGenerationFactory.generate_audit_metadata())
        metadata.update(MetadataGenerationFactory.generate_permission_metadata())
        metadata.update(MetadataGenerationFactory.generate_cache_metadata())
        
        # Add model-specific metadata
        metadata["model_info"] = {
            "type": model_type,
            "version": "1.0.0",
            "namespace": f"app.models.{model_type}",
            "description": f"Test {model_type} model with full metadata"
        }
        
        return metadata