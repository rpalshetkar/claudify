# XObjPrototype Feature Documentation

## Overview

XObjPrototype is a foundational abstraction that extends Pydantic's BaseModel to provide enhanced validation, namespace support, and metadata management capabilities. It serves as an **abstract base class** for all data models in the system, ensuring consistent validation patterns and integration with the namespace architecture.

**Important**: XObjPrototype cannot be instantiated directly - it must be subclassed. The implementation uses strict ABC enforcement with `__init_subclass__` validation to ensure all required methods are implemented.

## Core Features

### 1. Abstract Base Class

- Cannot be instantiated directly
- Must be subclassed to create concrete models
- Enforces implementation of required methods in subclasses
- Provides common functionality to all derived models

### 2. Schema Validation

- Built on Pydantic v2.10+ for robust data validation
- Automatic type coercion and validation
- Custom validators for domain-specific rules
- Nested model validation support

### 3. Namespace Integration

- Models are namespace-aware with get_ns() method
- Provides registration data for external NamespaceRegistrar
- Automatic fuzzy search text generation
- Cross-namespace model references
- Namespace-scoped validation rules

### 4. Metadata Management

- Flat dictionary metadata storage (Dict[str, Any])
- Helper methods for metadata manipulation
- Runtime introspection capabilities
- Excluded from serialization by default
- Fluent interface for metadata operations

### 5. Serialization/Deserialization

- JSON, YAML, TOML support out of the box
- Custom serializers for complex types
- Partial serialization support
- Schema export capabilities

## Usage Examples

### Basic Model Definition

```python
from server.core.base import XObjPrototype, FuzzyField
from pydantic import Field

# This will raise an error - cannot instantiate directly
# obj = XObjPrototype()  # InstantiationError: Cannot instantiate XObjPrototype directly

# Correct usage - must subclass
class User(XObjPrototype):
    id: str = Field(description="Unique identifier")
    name: str = FuzzyField(description="User's full name")
    email: str = FuzzyField(description="Email address")
    age: int = Field(description="User's age")
    internal_notes: str = Field(default="", description="Not searchable")
    
    def get_ns(self) -> str:
        return "users"
    
    def get_collection(self) -> str:
        return "user_profiles"
    
    def get_indexes(self) -> List[List[str]]:
        return [
            ["email"],  # Single field index
            ["name", "age"],  # Composite index
        ]

# Metadata usage
user = User(id="123", name="John Doe", email="john@example.com", age=30)
user.add_metadata(created_by="admin", department="sales", tags=["vip"])
print(user.get_metadata("created_by"))  # "admin"
```

### Complete Model with All Features

```python
from decimal import Decimal
from datetime import datetime
from typing import List

class Order(XObjPrototype):
    id: str = Field(description="Order ID")
    user_id: str = Field(description="User reference")
    customer_name: str = FuzzyField(description="Customer name for search")
    customer_email: str = FuzzyField(description="Customer email")
    items: List[OrderItem] = Field(default_factory=list)
    total: Decimal = Field(decimal_places=2)
    status: str = Field(default="pending")
    internal_notes: str = Field(default="", description="Not searchable")
    
    def get_ns(self) -> str:
        return "orders"
    
    def get_collection(self) -> str:
        return "order_history"
    
    def get_indexes(self) -> List[List[str]]:
        return [
            ["id"],
            ["user_id", "created_at"],
            ["status", "created_at"]
        ]

# Usage with namespace registration
order = Order(
    id="ORD-123",
    user_id="USER-456",
    customer_name="John Doe",
    customer_email="john@example.com",
    total=Decimal("99.99")
)

# Add metadata
order.add_metadata(
    created_by="sales_system",
    channel="web",
    promotion_code="SAVE10"
)

# Register in namespace
registrar = NamespaceRegistrar(cache_manager)
order_path = registrar.register_model(order)
# Registers: "ns.orders.ORD-123" -> order instance
# Also registers: "ns.fuzzy.orders.ORD-123" -> "John Doe john@example.com"
```

### Error Handling Example

```python
class APIConfiguration(XObjPrototype):
    endpoint: str = Field(description="API endpoint URL")
    api_key: str = Field(description="API key", repr=False)  # Hidden in repr
    timeout: int = Field(default=30, ge=1, le=300)
    
    def get_ns(self) -> str:
        return "config.api"
    
    def validate_endpoint(self) -> None:
        """Custom validation with specific exceptions"""
        if not self.endpoint.startswith(('http://', 'https://')):
            raise ValidationError(
                "Endpoint must start with http:// or https://",
                details={"endpoint": self.endpoint}
            )
    
    def set_api_key(self, key: str) -> None:
        """Set API key with validation"""
        if len(key) < 32:
            raise ValidationError(
                "API key must be at least 32 characters",
                details={"key_length": len(key)}
            )
        self.api_key = key
        self.add_metadata(api_key_updated=datetime.now())

# Usage with error handling
try:
    config = APIConfiguration(
        endpoint="https://api.example.com",
        api_key="short_key"  # This will pass initial validation
    )
    config.validate_endpoint()  # Custom validation
    config.set_api_key("short_key")  # This will raise ValidationError
except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Details: {e.details}")
```

### Abstract Base Implementation

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PrivateAttr
from typing import Any, Dict, List, Optional
import inspect

class XObjPrototype(BaseModel, ABC):
    """Abstract base class with strict enforcement"""
    
    # Metadata storage - flat dictionary
    _metadata: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    
    def __init__(self, **data: Any) -> None:
        if type(self) is XObjPrototype:
            raise InstantiationError(
                "Cannot instantiate XObjPrototype directly. Create a subclass."
            )
        super().__init__(**data)
    
    def __init_subclass__(cls, **kwargs):
        """Validate subclass implementation at class definition time"""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            # Check required abstract method implementation
            if cls.get_ns is XObjPrototype.get_ns:
                raise TypeError(f"{cls.__name__} must implement get_ns()")
    
    # Required abstract method
    @abstractmethod
    def get_ns(self) -> str:
        """Return the namespace for this model"""
        pass
    
    # Optional methods with sensible defaults
    def get_collection(self) -> str:
        """Return the collection/table name"""
        return f"{self.get_ns()}_{self.__class__.__name__.lower()}"
    
    def get_indexes(self) -> List[List[str]]:
        """Return index definitions"""
        return []  # No indexes by default
    
    # Metadata helper methods
    def add_metadata(self, **kwargs) -> "XObjPrototype":
        """Add multiple metadata entries"""
        self._metadata.update(kwargs)
        return self
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with optional default"""
        return self._metadata.get(key, default)
    
    # Fuzzy search support
    def get_fuzzy_fields(self) -> List[str]:
        """Return list of fuzzy searchable field names"""
        return [
            name for name, field in self.model_fields.items()
            if field.json_schema_extra and 
            field.json_schema_extra.get("fuzzy_searchable")
        ]
    
    def get_fuzzy_text(self) -> str:
        """Get concatenated fuzzy searchable text"""
        fuzzy_fields = self.get_fuzzy_fields()
        if not fuzzy_fields:
            return ""
        values = [str(getattr(self, field, "")) for field in fuzzy_fields]
        return " ".join(filter(None, values))
    
    # Namespace registration support
    def get_namespace_path(self, prefix: str = "ns") -> str:
        """Get the namespace path for this instance"""
        instance_id = getattr(self, 'id', 'unnamed')
        return f"{prefix}.{self.get_ns()}.{instance_id}"
    
    def get_registration_data(self) -> Dict[str, Any]:
        """Get data needed for namespace registration"""
        instance_id = getattr(self, 'id', '')
        return {
            "path": self.get_namespace_path(),
            "fuzzy_path": f"ns.fuzzy.{self.get_ns()}.{instance_id}",
            "fuzzy_text": self.get_fuzzy_text(),
            "instance": self
        }
```

## Exception Hierarchy

```python
# Base exception with details support
class XObjPrototypeError(Exception):
    """Base exception for all XObjPrototype errors"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

# Specific exception types
class InstantiationError(XObjPrototypeError):
    """Cannot instantiate abstract class"""
    pass

class ValidationError(XObjPrototypeError):
    """Validation failed"""
    pass

class MetadataError(XObjPrototypeError):
    """Metadata operation failed"""
    pass

class NamespaceError(XObjPrototypeError):
    """Namespace resolution failed"""
    pass

class SchemaError(XObjPrototypeError):
    """Schema-related error"""
    pass

class FuzzySearchError(XObjPrototypeError):
    """Fuzzy search configuration error"""
    pass
```

## Helper Functions

```python
from pydantic import Field

def FuzzyField(**kwargs):
    """Create a field that's included in fuzzy search"""
    kwargs.setdefault('json_schema_extra', {})['fuzzy_searchable'] = True
    return Field(**kwargs)
```

## Namespace Registration

```python
class NamespaceRegistrar:
    """Handles model registration in namespace"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    def register_model(self, model: XObjPrototype) -> str:
        """Register a model instance in namespace"""
        data = model.get_registration_data()
        
        # Register main instance
        self.cache.register_ns(data["path"], data["instance"])
        
        # Register fuzzy search if applicable
        if data["fuzzy_text"]:
            self.cache.register_ns(data["fuzzy_path"], data["fuzzy_text"])
        
        return data["path"]
    
    def register_models(self, models: List[XObjPrototype]) -> List[str]:
        """Bulk register multiple models"""
        return [self.register_model(model) for model in models]
```

## Implementation Plan

### Phase 1: Core Foundation (Week 1)

1. **Abstract Base Class Structure**

   - Create XObjPrototype with strict ABC pattern
   - Implement __init_subclass__ validation
   - Define required and optional methods
   - Add metadata storage with helper methods
   - Implement fuzzy search field detection

2. **Validation Framework**

   - Custom validator decorators
   - Field-level validation rules
   - Model-level validation methods
   - Abstract validation methods for subclasses

3. **Serialization Support**
   - JSON serialization with custom encoders
   - YAML/TOML support via extensions
   - Schema export functionality
   - Redaction of sensitive fields

### Phase 2: Namespace Integration (Week 2)

1. **Namespace Support**

   - Implement get_namespace_path() method
   - Create get_registration_data() for external registration
   - Build NamespaceRegistrar utility class
   - Support fuzzy search path generation

2. **Exception Implementation**

   - Create comprehensive exception hierarchy
   - Add error details support
   - Implement specific error contexts
   - Domain-specific error handling

3. **Settings Bridge**
   - DynaConf integration
   - Environment-aware loading
   - Namespace-based configuration
   - Sensitive data handling

### Phase 3: Advanced Features (Week 3)

1. **Fuzzy Search Integration**

   - Implement FuzzyField helper function
   - Add get_fuzzy_fields() method
   - Create get_fuzzy_text() concatenation
   - Test fuzzy search registration

2. **Metadata Management**

   - Implement add_metadata() fluent interface
   - Add get_metadata() with defaults
   - Create metadata validation
   - Document metadata patterns

3. **Performance Optimization**
   - Validation caching
   - Lazy loading support
   - Bulk validation methods
   - Index optimization hints

## Technical Architecture

### Class Hierarchy

```
pydantic.BaseModel
    └── XObjPrototype (ABC)
        ├── Meta configuration
        ├── Namespace integration
        ├── Validation extensions
        ├── Serialization helpers
        └── Abstract method definitions
            └── Concrete Model Classes (User, Order, etc.)
```

### Key Methods

**Required Abstract Methods:**
- `get_ns()` - Must return the namespace for this model

**Optional Override Methods:**
- `get_collection()` - Returns collection/table name (default: ns_classname)
- `get_indexes()` - Returns index definitions (default: empty list)

**Core Methods:**
- `__init__()` - Prevents direct instantiation with InstantiationError
- `__init_subclass__()` - Validates subclass implementation

**Metadata Methods:**
- `add_metadata(**kwargs)` - Fluent interface for adding metadata
- `get_metadata(key, default)` - Retrieve metadata with optional default

**Fuzzy Search Methods:**
- `get_fuzzy_fields()` - Returns list of fuzzy searchable field names
- `get_fuzzy_text()` - Returns concatenated fuzzy search text

**Namespace Integration:**
- `get_namespace_path(prefix)` - Generate namespace path for instance
- `get_registration_data()` - Prepare data for namespace registration

### Integration Points

1. **DynaConf Settings**

   - Models can load from settings
   - Environment-aware configuration
   - Namespace-based settings resolution
   - Sensitive field handling

2. **Repository Pattern**

   - Models work seamlessly with repositories
   - Automatic collection mapping
   - Query builder integration
   - Index hint propagation

3. **Resource/Connector**
   - Models define resource requirements
   - Automatic connector resolution
   - Resource lifecycle management
   - Connection pooling awareness

## Testing Strategy

### Unit Tests

```python
def test_cannot_instantiate_directly():
    """Test XObjPrototype cannot be instantiated"""
    with pytest.raises(InstantiationError, match="Cannot instantiate XObjPrototype directly"):
        XObjPrototype()

def test_missing_get_ns_implementation():
    """Test error when get_ns() not implemented"""
    with pytest.raises(TypeError, match="must implement get_ns"):
        class BadModel(XObjPrototype):
            name: str
            # Missing get_ns() implementation

def test_metadata_operations():
    """Test metadata helper methods"""
    class TestModel(XObjPrototype):
        name: str
        def get_ns(self) -> str:
            return "test"
    
    model = TestModel(name="test")
    model.add_metadata(created_by="admin", tags=["test", "unit"])
    
    assert model.get_metadata("created_by") == "admin"
    assert model.get_metadata("missing", "default") == "default"
    assert model._metadata["tags"] == ["test", "unit"]

def test_fuzzy_search_fields():
    """Test fuzzy field detection and text generation"""
    class Product(XObjPrototype):
        name: str = FuzzyField()
        sku: str = FuzzyField()
        internal_id: str = Field()
        
        def get_ns(self) -> str:
            return "products"
    
    product = Product(name="Blue Widget", sku="BW-123", internal_id="INT-456")
    
    assert set(product.get_fuzzy_fields()) == {"name", "sku"}
    assert product.get_fuzzy_text() == "Blue Widget BW-123"

def test_namespace_registration_data():
    """Test namespace registration data generation"""
    class User(XObjPrototype):
        id: str
        name: str = FuzzyField()
        
        def get_ns(self) -> str:
            return "users"
    
    user = User(id="123", name="John Doe")
    data = user.get_registration_data()
    
    assert data["path"] == "ns.users.123"
    assert data["fuzzy_path"] == "ns.fuzzy.users.123"
    assert data["fuzzy_text"] == "John Doe"
    assert data["instance"] is user
```

### Integration Tests

```python
def test_dynaconf_integration():
    """Test loading from DynaConf settings"""

def test_repository_integration():
    """Test model works with repository pattern"""

def test_serialization_roundtrip():
    """Test serialization/deserialization"""

def test_sensitive_field_redaction():
    """Test sensitive fields are redacted in logs"""
```

## Security Considerations

1. **Input Validation**

   - All inputs validated through Pydantic
   - Custom validators for sensitive data
   - SQL injection prevention
   - Abstract validation requirements

2. **Data Sanitization**

   - Automatic HTML escaping
   - Path traversal prevention
   - Command injection protection
   - Sensitive field redaction

3. **Access Control**
   - Namespace-based permissions
   - Field-level access control
   - Audit logging support
   - Abstract class protection

## Performance Considerations

1. **Validation Caching**

   - Cache validation results for unchanged data
   - Lazy validation for large datasets
   - Batch validation support
   - Index-aware queries

2. **Memory Management**

   - Efficient model instantiation
   - Garbage collection optimization
   - Large dataset handling
   - Subclass registry efficiency

3. **Serialization Performance**
   - Fast JSON serialization
   - Streaming support for large models
   - Partial serialization capabilities
   - Composite key optimization

## Migration Guide

### From Standard Pydantic

```python
# Before - Standard Pydantic
from pydantic import BaseModel

class User(BaseModel):
    id: str
    name: str
    email: str
    role: str

# After - XObjPrototype with all features
from server.core.base import XObjPrototype, FuzzyField
from pydantic import Field

class User(XObjPrototype):
    id: str = Field(description="Unique user ID")
    name: str = FuzzyField(description="Full name for search")
    email: str = FuzzyField(description="Email for search")
    role: str = Field(default="user", description="User role")
    
    def get_ns(self) -> str:
        return "users"
    
    def get_collection(self) -> str:
        return "user_profiles"
    
    def get_indexes(self) -> List[List[str]]:
        return [
            ["email"],  # Unique email index
            ["name", "role"],  # Composite index for queries
        ]

# Startup registration
async def startup(cache_manager: CacheManager):
    # Create registrar
    registrar = NamespaceRegistrar(cache_manager)
    
    # Create and register user
    user = User(
        id="user-123",
        name="John Doe",
        email="john@example.com",
        role="admin"
    )
    
    # Add metadata
    user.add_metadata(
        created_at=datetime.now(),
        created_by="system",
        source="migration"
    )
    
    # Register in namespace
    user_path = registrar.register_model(user)
    print(f"User registered at: {user_path}")
    # Output: User registered at: ns.users.user-123
    
    # Fuzzy search also registered automatically
    # ns.fuzzy.users.user-123 -> "John Doe john@example.com"
```

## Future Considerations

### 1. GraphQL Support

- Automatic GraphQL schema generation
- Resolver integration
- Subscription support
- Field-level permissions

### 2. Event Sourcing

- Model change tracking
- Event stream generation
- Replay capabilities
- Audit trail

### 3. Multi-tenancy

- Tenant-aware models
- Data isolation
- Cross-tenant queries
- Tenant-specific indexes

### 4. Architectural Decisions Summary

Based on our architecture review, the following decisions have been made:

- **Abstract Methods**: Only `get_ns()` is required; others have sensible defaults
- **Instantiation**: Strict ABC enforcement with `__init_subclass__` validation
- **Metadata**: Flat dictionary with helper methods for fluent interface
- **Fuzzy Search**: Field-level metadata using `FuzzyField` helper
- **Exceptions**: Comprehensive domain-specific exception hierarchy
- **Namespace**: External registration via `NamespaceRegistrar` pattern
- **Registration**: Manual registration order during application startup

### 5. Integration with Other Components

- **XInspector**: Will generate dynamic models that inherit from XObjPrototype
- **XModels**: Will register XObjPrototype instances for runtime management
- **XRepo**: Will use XObjPrototype metadata for audit and permissions
- **XResource**: Connection information can be stored in XObjPrototype metadata

### 6. Performance Optimizations

- Fuzzy field detection is cached at class level
- Metadata operations are O(1) dictionary operations
- Namespace paths are generated on-demand
- Index definitions are static class-level data

### 7. Security Considerations Enhanced

- Sensitive fields can be marked in metadata for redaction
- Field-level permissions can be stored in model metadata
- Audit trail integration via metadata timestamps
- Validation errors include safe details only

## Dependencies

- pydantic >= 2.10
- dynaconf >= 3.2
- typing-extensions >= 4.0
- python >= 3.12
