"""Test fixtures for XObjPrototype models."""

from decimal import Decimal
from typing import Any

import pytest

from mock.xobj_factories import (
    FuzzySearchTestFactory,
    MetadataTestFactory,
    NamespaceTestFactory,
    ValidationTestFactory,
    MetadataGenerationFactory,
)


@pytest.fixture
def valid_model_data() -> dict[str, Any]:
    """Valid model instance data."""
    return ValidationTestFactory.valid_user_data()


@pytest.fixture
def invalid_model_data() -> list[dict[str, Any]]:
    """Various invalid model data scenarios."""
    return ValidationTestFactory.invalid_user_data()


@pytest.fixture
def edge_case_data() -> list[dict[str, Any]]:
    """Edge case model data."""
    return ValidationTestFactory.edge_case_data()


@pytest.fixture
def namespace_hierarchy() -> dict[str, Any]:
    """Nested namespace structure."""
    return NamespaceTestFactory.create_namespace_hierarchy()


@pytest.fixture
def namespace_collisions() -> list[dict[str, Any]]:
    """Namespace collision scenarios."""
    return NamespaceTestFactory.create_collision_scenarios()


@pytest.fixture
def basic_metadata() -> dict[str, Any]:
    """Basic metadata dictionary."""
    return MetadataTestFactory.create_basic_metadata()


@pytest.fixture
def complex_metadata() -> dict[str, Any]:
    """Complex nested metadata."""
    return MetadataTestFactory.create_complex_metadata()


@pytest.fixture
def searchable_models() -> list[dict[str, Any]]:
    """Models with fuzzy searchable fields."""
    return FuzzySearchTestFactory.create_searchable_models()


@pytest.fixture
def search_queries() -> list[dict[str, Any]]:
    """Search query test cases."""
    return FuzzySearchTestFactory.create_search_queries()


@pytest.fixture
def sample_user_data() -> dict[str, Any]:
    """Sample user model data."""
    return {
        "id": "user-123",
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "role": "admin"
    }


@pytest.fixture
def sample_product_data() -> dict[str, Any]:
    """Sample product model data."""
    return {
        "id": "prod-456",
        "name": "Laptop",
        "sku": "SKU-1234",
        "price": Decimal("999.99"),
        "category": "electronics",
        "active": True
    }


@pytest.fixture
def sample_order_data() -> dict[str, Any]:
    """Sample order model data."""
    return {
        "id": "order-789",
        "user_id": "user-123",
        "customer_name": "John Doe",
        "customer_email": "john@example.com",
        "total": Decimal("999.99"),
        "status": "pending",
        "items": []
    }


@pytest.fixture
def bulk_model_data() -> list[dict[str, Any]]:
    """Bulk model data for performance testing."""
    return [
        {
            "id": f"user-{i:04d}",
            "name": f"User {i}",
            "email": f"user{i}@example.com",
            "age": 20 + (i % 60),
            "role": ["user", "admin", "moderator"][i % 3]
        }
        for i in range(100)
    ]


@pytest.fixture
def validation_test_cases() -> list[dict[str, Any]]:
    """Comprehensive validation test cases."""
    return [
        {
            "name": "valid_all_fields",
            "data": {
                "id": "test-001",
                "name": "Test User",
                "email": "test@example.com",
                "age": 25,
                "role": "user"
            },
            "should_pass": True
        },
        {
            "name": "missing_required_id",
            "data": {
                "name": "Test User",
                "email": "test@example.com",
                "age": 25
            },
            "should_pass": False,
            "error_field": "id"
        },
        {
            "name": "invalid_email_format",
            "data": {
                "id": "test-002",
                "name": "Test User",
                "email": "not-an-email",
                "age": 25
            },
            "should_pass": False,
            "error_field": "email"
        },
        {
            "name": "negative_age",
            "data": {
                "id": "test-003",
                "name": "Test User",
                "email": "test@example.com",
                "age": -5
            },
            "should_pass": False,
            "error_field": "age"
        },
        {
            "name": "wrong_type_age",
            "data": {
                "id": "test-004",
                "name": "Test User",
                "email": "test@example.com",
                "age": "twenty-five"
            },
            "should_pass": False,
            "error_field": "age"
        }
    ]


@pytest.fixture
def serialization_test_data() -> dict[str, Any]:
    """Data for testing serialization/deserialization."""
    return {
        "model_data": {
            "id": "test-ser-001",
            "name": "Serialization Test",
            "email": "serial@test.com",
            "age": 30,
            "role": "admin"
        },
        "metadata": {
            "created_by": "test_system",
            "version": 1,
            "tags": ["test", "serialization"]
        },
        "expected_json_keys": ["id", "name", "email", "age", "role"],
        "excluded_keys": ["_metadata"]
    }


@pytest.fixture
def fuzzy_field_test_data() -> dict[str, Any]:
    """Data for testing fuzzy field functionality."""
    return {
        "fields": {
            "name": {"value": "John Doe", "fuzzy": True},
            "email": {"value": "john@example.com", "fuzzy": True},
            "internal_id": {"value": "INT-123", "fuzzy": False},
            "notes": {"value": "Internal notes", "fuzzy": False}
        },
        "expected_fuzzy_fields": ["name", "email"],
        "expected_fuzzy_text": "John Doe john@example.com"
    }


@pytest.fixture
def namespace_registration_data() -> dict[str, Any]:
    """Data for testing namespace registration."""
    return {
        "model": {
            "id": "test-ns-001",
            "name": "Namespace Test",
            "namespace": "tests"
        },
        "expected_paths": {
            "main": "ns.tests.test-ns-001",
            "fuzzy": "ns.fuzzy.tests.test-ns-001"
        },
        "custom_prefix": {
            "prefix": "custom",
            "expected": "custom.tests.test-ns-001"
        }
    }


# Enhanced fixtures for comprehensive testing
@pytest.fixture
def complete_valid_models() -> dict[str, dict[str, Any]]:
    """Complete valid model data for all model types."""
    return {
        "user": {
            "id": "user-valid-001",
            "name": "Alice Johnson",
            "email": "alice.johnson@example.com",
            "age": 35,
            "role": "admin",
            "_metadata": MetadataGenerationFactory.generate_complete_metadata("user")
        },
        "product": {
            "id": "prod-valid-001",
            "name": "Premium Laptop",
            "sku": "SKU-LAPTOP-001",
            "price": Decimal("1299.99"),
            "category": "electronics",
            "active": True,
            "_metadata": MetadataGenerationFactory.generate_complete_metadata("product")
        },
        "config": {
            "id": "config-valid-001",
            "key": "max_upload_size",
            "value": "10MB",
            "environment": "prod",
            "enabled": True,
            "_metadata": MetadataGenerationFactory.generate_complete_metadata("config")
        },
        "order": {
            "id": "order-valid-001",
            "user_id": "user-valid-001",
            "customer_name": "Alice Johnson",
            "customer_email": "alice.johnson@example.com",
            "total": Decimal("1299.99"),
            "status": "pending",
            "items": [
                {"product_id": "prod-valid-001", "quantity": 1, "price": Decimal("1299.99")}
            ],
            "_metadata": MetadataGenerationFactory.generate_complete_metadata("order")
        }
    }


@pytest.fixture
def invalid_models_comprehensive() -> dict[str, list[dict[str, Any]]]:
    """Comprehensive invalid model scenarios for each model type."""
    return {
        "user": [
            # Missing required fields
            {"name": "Test", "email": "test@example.com", "age": 25},  # missing id
            {"id": "u1", "email": "test@example.com", "age": 25},  # missing name
            {"id": "u2", "name": "Test", "age": 25},  # missing email
            {"id": "u3", "name": "Test", "email": "test@example.com"},  # missing age
            
            # Invalid field types
            {"id": 123, "name": "Test", "email": "test@example.com", "age": 25},  # id not string
            {"id": "u4", "name": 123, "email": "test@example.com", "age": 25},  # name not string
            {"id": "u5", "name": "Test", "email": "test@example.com", "age": "25"},  # age not int
            
            # Invalid field values
            {"id": "u6", "name": "Test", "email": "invalid-email", "age": 25},  # invalid email
            {"id": "u7", "name": "Test", "email": "test@example.com", "age": -1},  # negative age
            {"id": "u8", "name": "Test", "email": "test@example.com", "age": 200},  # age too high
            {"id": "u9", "name": "Test", "email": "test@example.com", "age": 25, "role": "superuser"},  # invalid role
        ],
        "product": [
            # Missing required fields
            {"name": "Product", "sku": "SKU-001", "price": Decimal("10.00"), "category": "test"},  # missing id
            {"id": "p1", "sku": "SKU-001", "price": Decimal("10.00"), "category": "test"},  # missing name
            
            # Invalid field types
            {"id": "p2", "name": "Product", "sku": "SKU-001", "price": "10.00", "category": "test", "active": True},  # price not Decimal
            {"id": "p3", "name": "Product", "sku": "SKU-001", "price": Decimal("10.00"), "category": "test", "active": "yes"},  # active not bool
            
            # Invalid field values
            {"id": "p4", "name": "Product", "sku": "SKU-001", "price": Decimal("-10.00"), "category": "test"},  # negative price
        ],
        "order": [
            # Invalid email in order
            {"id": "o1", "user_id": "u1", "customer_name": "Test", "customer_email": "bad-email", "total": Decimal("10.00")},
            
            # Invalid status
            {"id": "o2", "user_id": "u1", "customer_name": "Test", "customer_email": "test@example.com", "total": Decimal("10.00"), "status": "invalid"},
            
            # Negative total
            {"id": "o3", "user_id": "u1", "customer_name": "Test", "customer_email": "test@example.com", "total": Decimal("-10.00")},
        ]
    }


@pytest.fixture
def edge_case_models() -> dict[str, list[dict[str, Any]]]:
    """Edge case scenarios for model validation."""
    return {
        "empty_strings": [
            {"id": "", "name": "", "email": "test@example.com", "age": 25},  # empty id and name
            {"id": "u1", "name": "Test", "email": "", "age": 25},  # empty email
        ],
        "very_long_strings": [
            {"id": "u" * 1000, "name": "n" * 1000, "email": "test@example.com", "age": 25},
            {"id": "p1", "name": "Product", "sku": "S" * 1000, "price": Decimal("10.00"), "category": "test"},
        ],
        "boundary_values": [
            {"id": "u1", "name": "Test", "email": "test@example.com", "age": 0},  # min age
            {"id": "u2", "name": "Test", "email": "test@example.com", "age": 150},  # max age
            {"id": "p1", "name": "Product", "sku": "SKU-001", "price": Decimal("0.00"), "category": "test"},  # zero price
            {"id": "p2", "name": "Product", "sku": "SKU-001", "price": Decimal("999999.99"), "category": "test"},  # large price
        ],
        "special_characters": [
            {"id": "u-1", "name": "Test User!@#$%^&*()", "email": "test+tag@example.com", "age": 25},
            {"id": "p-1", "name": "Product™ with © symbols", "sku": "SKU-001™", "price": Decimal("10.00"), "category": "test"},
        ],
        "unicode_data": [
            {"id": "u1", "name": "テスト", "email": "test@example.com", "age": 25},  # Japanese
            {"id": "u2", "name": "Тест", "email": "test@example.com", "age": 25},  # Cyrillic
            {"id": "u3", "name": "🧪 Test User 🎯", "email": "test@example.com", "age": 25},  # Emojis
        ],
        "null_optional_fields": [
            {"id": "u1", "name": "Test", "email": "test@example.com", "age": 25, "role": None},
            {"id": "o1", "user_id": "u1", "customer_name": "Test", "customer_email": "test@example.com", "total": Decimal("10.00"), "items": None},
        ]
    }


@pytest.fixture
def metadata_test_scenarios() -> dict[str, dict[str, Any]]:
    """Comprehensive metadata test scenarios."""
    return {
        "validation_metadata": MetadataGenerationFactory.generate_validation_metadata(),
        "fuzzy_search_metadata": MetadataGenerationFactory.generate_fuzzy_search_metadata(),
        "ui_metadata": MetadataGenerationFactory.generate_ui_metadata(),
        "audit_metadata": MetadataGenerationFactory.generate_audit_metadata("test_user"),
        "permission_metadata": MetadataGenerationFactory.generate_permission_metadata(["admin", "user", "guest"]),
        "cache_metadata": MetadataGenerationFactory.generate_cache_metadata(),
        "complete_metadata": MetadataGenerationFactory.generate_complete_metadata("test_model")
    }


@pytest.fixture
def nested_model_data() -> dict[str, Any]:
    """Data for testing nested model scenarios."""
    return {
        "order_with_items": {
            "id": "order-nested-001",
            "user_id": "user-001",
            "customer_name": "John Doe",
            "customer_email": "john@example.com",
            "total": Decimal("299.97"),
            "status": "pending",
            "items": [
                {
                    "product_id": "prod-001",
                    "name": "Widget A",
                    "quantity": 2,
                    "price": Decimal("49.99"),
                    "subtotal": Decimal("99.98")
                },
                {
                    "product_id": "prod-002",
                    "name": "Widget B",
                    "quantity": 1,
                    "price": Decimal("199.99"),
                    "subtotal": Decimal("199.99")
                }
            ],
            "_metadata": {
                "items_count": 2,
                "discount_applied": False,
                "shipping_method": "standard"
            }
        }
    }


@pytest.fixture
def performance_test_data() -> dict[str, Any]:
    """Data for performance testing."""
    return {
        "large_dataset": [
            {
                "id": f"perf-{i:06d}",
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "age": 20 + (i % 60),
                "role": ["user", "admin", "moderator"][i % 3]
            }
            for i in range(10000)
        ],
        "complex_metadata": {
            f"level_{i}": {
                f"sublevel_{j}": {
                    "data": f"value_{i}_{j}",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "tags": [f"tag_{k}" for k in range(5)]
                }
                for j in range(10)
            }
            for i in range(5)
        }
    }


# Enhanced namespace fixtures
@pytest.fixture
def comprehensive_namespace_scenarios() -> dict[str, Any]:
    """Comprehensive namespace test scenarios."""
    return {
        "deep_hierarchy": {
            "app": {
                "models": {
                    "core": {
                        "user": ["user-001", "user-002", "user-003"],
                        "role": ["role-001", "role-002"],
                        "permission": ["perm-001", "perm-002", "perm-003"]
                    },
                    "business": {
                        "product": ["prod-001", "prod-002"],
                        "order": ["order-001", "order-002"],
                        "invoice": ["inv-001", "inv-002"]
                    },
                    "config": {
                        "system": ["sys-001", "sys-002"],
                        "user": ["usr-cfg-001", "usr-cfg-002"]
                    }
                },
                "services": {
                    "auth": ["auth-svc-001"],
                    "api": ["api-svc-001"],
                    "cache": ["cache-svc-001"]
                }
            }
        },
        "collision_scenarios": [
            # Same ID in different namespaces (should be allowed)
            {"ns": "users.admin", "id": "001", "expected": "allowed"},
            {"ns": "users.regular", "id": "001", "expected": "allowed"},
            
            # Same ID in same namespace (should conflict)
            {"ns": "users.admin", "id": "duplicate", "first": True},
            {"ns": "users.admin", "id": "duplicate", "expected": "conflict"},
            
            # Nested namespace conflicts
            {"ns": "app.models", "id": "test", "expected": "allowed"},
            {"ns": "app.models.user", "id": "test", "expected": "allowed"},
            {"ns": "app", "id": "models", "expected": "conflict"},  # conflicts with namespace path
        ],
        "dynamic_namespace_creation": [
            {"action": "create", "path": "dynamic.level1.level2", "id": "item-001"},
            {"action": "create", "path": "dynamic.level1.level3", "id": "item-002"},
            {"action": "move", "from": "dynamic.level1.level2", "to": "dynamic.level1.level3", "id": "item-001"},
            {"action": "delete", "path": "dynamic.level1.level2"},
        ],
        "special_characters_in_namespace": [
            {"ns": "user-data", "id": "001", "valid": True},
            {"ns": "user_data", "id": "002", "valid": True},
            {"ns": "user.data", "id": "003", "valid": True},
            {"ns": "user@data", "id": "004", "valid": False},
            {"ns": "user/data", "id": "005", "valid": False},
            {"ns": "user\\data", "id": "006", "valid": False},
        ],
        "namespace_inheritance": {
            "parent": {
                "namespace": "app.models",
                "metadata": {"access": "restricted", "version": "1.0"},
                "children": {
                    "user": {
                        "inherits": True,
                        "overrides": {"access": "public"}
                    },
                    "admin": {
                        "inherits": True,
                        "additional": {"role": "superuser"}
                    }
                }
            }
        }
    }


@pytest.fixture
def namespace_traversal_data() -> dict[str, Any]:
    """Data for testing namespace traversal operations."""
    return {
        "search_patterns": [
            {"pattern": "app.*", "expected_count": 10},
            {"pattern": "*.user", "expected_count": 3},
            {"pattern": "app.*.config", "expected_count": 1},
            {"pattern": "**", "expected_count": 25},  # all items
        ],
        "path_resolution": [
            {"input": "app.models.user", "resolved": ["app", "models", "user"]},
            {"input": "app..models", "resolved": ["app", "models"]},  # double dot handling
            {"input": ".app.models.", "resolved": ["app", "models"]},  # leading/trailing dots
        ],
        "breadcrumb_generation": [
            {
                "path": "app.models.user.profile",
                "breadcrumbs": [
                    {"name": "app", "path": "app"},
                    {"name": "models", "path": "app.models"},
                    {"name": "user", "path": "app.models.user"},
                    {"name": "profile", "path": "app.models.user.profile"}
                ]
            }
        ]
    }


@pytest.fixture
def namespace_permission_data() -> dict[str, Any]:
    """Data for testing namespace-based permissions."""
    return {
        "permission_hierarchy": {
            "app": {"read": ["*"], "write": ["admin"]},
            "app.models": {"read": ["user", "admin"], "write": ["admin"]},
            "app.models.user": {"read": ["user", "admin"], "write": ["user", "admin"]},
            "app.models.admin": {"read": ["admin"], "write": ["admin"]},
        },
        "access_tests": [
            {"user": "guest", "namespace": "app", "action": "read", "expected": True},
            {"user": "guest", "namespace": "app.models", "action": "read", "expected": False},
            {"user": "user", "namespace": "app.models.user", "action": "write", "expected": True},
            {"user": "user", "namespace": "app.models.admin", "action": "read", "expected": False},
        ]
    }