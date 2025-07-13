# XObjPrototype Feature Documentation

## Overview

XObjPrototype is a foundational abstraction that extends Pydantic's BaseModel to provide enhanced validation, namespace support, and metadata management capabilities. It serves as an **abstract base class** for all data models in the system, ensuring consistent validation patterns and integration with the namespace architecture. 

**Important**: XObjPrototype cannot be instantiated directly - it must be subclassed.

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
- Models are namespace-aware
- Automatic namespace resolution from model context
- Cross-namespace model references
- Namespace-scoped validation rules

### 4. Metadata Management
- Rich metadata support for models
- Runtime introspection capabilities
- Schema evolution tracking
- Model versioning support

### 5. Serialization/Deserialization
- JSON, YAML, TOML support out of the box
- Custom serializers for complex types
- Partial serialization support
- Schema export capabilities

## Usage Examples

### Basic Model Definition
```python
from server.core.base import XObjPrototype
from abc import ABC

# This will raise an error - cannot instantiate directly
# obj = XObjPrototype()  # TypeError: Cannot instantiate abstract class

# Correct usage - must subclass
class User(XObjPrototype):
    id: str
    name: str
    email: str
    age: int
    
    class Meta:
        namespace = "users"
        collection = "user_profiles"
        indexes = [
            ["email"],  # Single field index
            ["name", "age"],  # Composite index
        ]
        unique_constraints = [
            ["email"],
            ["id", "namespace"]
        ]
```

### Namespace-Aware Model
```python
class Order(XObjPrototype):
    order_id: str
    user_id: str
    items: List[OrderItem]
    total: Decimal
    
    class Meta:
        namespace = "orders"
        collection = "order_history"
        indexes = [
            ["order_id"],
            ["user_id", "created_at"],
            ["status", "created_at"]
        ]
        
    def get_user(self) -> User:
        # Automatically resolves user from users namespace
        return self.resolve_reference("users", self.user_id)
```

### Settings Integration
```python
class DatabaseSettings(XObjPrototype):
    host: str
    port: int
    database: str
    username: str
    password: str
    
    class Meta:
        namespace = "settings.database"
        source = "dynaconf"
        sensitive_fields = ["password"]  # Fields to redact in logs
        
    @classmethod
    def from_namespace(cls, ns: str) -> "DatabaseSettings":
        # Load settings from namespace-specific config
        return cls.load_from_dynaconf(ns)
```

### Abstract Base Implementation
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict, Type

class XObjPrototype(BaseModel, ABC):
    """Abstract base class that cannot be instantiated directly"""
    
    def __init__(self, **data: Any) -> None:
        if self.__class__ is XObjPrototype:
            raise TypeError(
                "XObjPrototype is an abstract class and cannot be instantiated directly. "
                "Please create a subclass."
            )
        super().__init__(**data)
    
    @abstractmethod
    def get_namespace(self) -> str:
        """Each model must define its namespace"""
        pass
    
    class Meta:
        abstract = True
```

## Implementation Plan

### Phase 1: Core Foundation (Week 1)
1. **Abstract Base Class Structure**
   - Create XObjPrototype as abstract class extending Pydantic BaseModel
   - Implement instantiation prevention mechanism
   - Define abstract methods that subclasses must implement
   - Implement Meta class configuration pattern
   - Add namespace property and resolution logic

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
1. **Namespace Awareness**
   - Namespace registry integration
   - Cross-namespace reference resolution
   - Namespace-scoped validation
   - Hierarchical namespace support

2. **Metadata System**
   - Meta class processing
   - Runtime metadata access
   - Model introspection API
   - Composite index support

3. **Settings Bridge**
   - DynaConf integration
   - Environment-aware loading
   - Namespace-based configuration
   - Sensitive data handling

### Phase 3: Advanced Features (Week 3)
1. **Model Registry**
   - Automatic model registration
   - Model discovery by namespace
   - Dynamic model loading
   - Subclass tracking

2. **Schema Evolution**
   - Version tracking
   - Migration support
   - Backward compatibility
   - Schema diff generation

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
- `__init__()` - Prevents direct instantiation
- `get_namespace()` - Abstract method for namespace definition
- `validate_namespace()` - Ensure model belongs to correct namespace
- `resolve_reference()` - Cross-namespace reference resolution
- `to_dict()` - Enhanced serialization with metadata
- `from_namespace()` - Load model from namespace config
- `get_schema()` - Export model schema
- `validate_against_schema()` - Runtime schema validation
- `get_indexes()` - Return composite index definitions
- `get_constraints()` - Return unique constraint definitions

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
    with pytest.raises(TypeError, match="abstract class"):
        XObjPrototype()

def test_subclass_instantiation():
    """Test subclasses can be instantiated"""
    class TestModel(XObjPrototype):
        name: str
        def get_namespace(self) -> str:
            return "test"
    
    model = TestModel(name="test")
    assert model.name == "test"
    
def test_composite_indexes():
    """Test composite index definitions"""
    
def test_namespace_resolution():
    """Test namespace is correctly resolved"""
    
def test_meta_configuration():
    """Test Meta class configuration"""
    
def test_cross_namespace_reference():
    """Test reference resolution across namespaces"""
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
# Before
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str

# After
from server.core.base import XObjPrototype

class User(XObjPrototype):
    name: str
    email: str
    
    def get_namespace(self) -> str:
        return "users"
    
    class Meta:
        namespace = "users"
        collection = "user_profiles"
        indexes = [
            ["email"],
            ["name", "created_at"]
        ]
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

### 4. Abstract Method Requirements
- What methods should be abstract beyond get_namespace()?
- Should we enforce validation methods in subclasses?
- How to handle optional abstract methods?

### 5. Composite Index Behavior
- Should indexes be automatically created in the database?
- How to handle index naming conventions?
- Support for partial indexes?

### 6. Namespace Structure
- Should namespaces be hierarchical (e.g., "app.users.profiles")?
- How deep should namespace nesting be allowed?
- Should models inherit parent namespace properties?

### 7. Validation Behavior
- Should validation be synchronous or support async validators?
- How should circular references be handled?
- What's the strategy for external API validation?

### 8. Settings Integration
- Should all models support loading from settings?
- How to handle environment-specific validation rules?
- What's the precedence order for configuration sources?

### 9. Performance Requirements
- What's the expected model instantiation rate?
- How large can nested models be?
- Should we implement model pooling?

### 10. Schema Evolution
- How to handle breaking changes in model schemas?
- Should we support automatic migration generation?
- What's the versioning strategy?

## Dependencies

- pydantic >= 2.10
- dynaconf >= 3.2
- typing-extensions >= 4.0
- python >= 3.12