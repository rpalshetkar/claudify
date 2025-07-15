# XRegistry - Unified Registry Architecture

## Overview

XRegistry is a comprehensive registry system that manages functions, models, types, and their metadata within the XObjPrototype ecosystem. It serves as the central registration and management hub, maintaining discipline over what code can access and use.

## Core Responsibilities

1. **Function Registry** - Register and manage Python functions with metadata and permissions
2. **Model Registry** - Manage both static (imported) and dynamic (Inspector-generated) models
3. **Type Registry** - Register custom type classes and validators
4. **Import Control** - Enforce discipline via models folder __init__.py
5. **Resource Enrichment** - Add live statistics when Resources are connected
6. **Namespace Management** - Organize all registered items in namespace paths

## Architecture Principles

- **Registration, Not Discovery** - XRegistry manages registrations; XInspector handles discovery
- **Lazy Loading** - Items are loaded only when accessed
- **Permission-Aware** - All registered items have associated permissions
- **Namespace-Organized** - Everything accessible via dot notation paths
- **Audit Everything** - Track all registration and access operations

## Function Registry

### Architecture

```python
from typing import Callable, Dict, Any, Optional, Set, Type
from pydantic import BaseModel, Field
from src.server.core.xobj_prototype import XObjPrototype
from enum import Enum
import inspect
from datetime import datetime

class FunctionPermission(Enum):
    """Function access permissions"""
    EXECUTE = "execute"
    VIEW = "view"
    MODIFY = "modify"

class FunctionMetadata(BaseModel):
    """Metadata for registered functions"""
    name: str
    module: str
    signature: str
    docstring: Optional[str]
    parameters: Dict[str, Any]
    return_type: Optional[str]
    tags: Set[str] = Field(default_factory=set)
    permissions: Dict[str, Set[FunctionPermission]] = Field(default_factory=dict)
    ns_path: str  # e.g., "ns.functions.utils.calculate_total"
    registered_at: datetime = Field(default_factory=datetime.now)

class RegisteredFunction(XObjPrototype):
    """A registered function with metadata"""
    metadata: FunctionMetadata
    func: Callable
    call_count: int = 0
    last_called: Optional[datetime] = None
    ns: str = "registry.functions"

class FunctionRegistry:
    """Registry for Python functions"""
    
    def __init__(self):
        self.functions: Dict[str, RegisteredFunction] = {}
        self.namespace_map: Dict[str, str] = {}  # ns_path -> function_name
        
    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        permissions: Optional[Dict[str, Set[FunctionPermission]]] = None
    ) -> RegisteredFunction:
        """Register a function with metadata"""
        func_name = name or func.__name__
        module = func.__module__
        
        # Build namespace path
        if namespace:
            ns_path = f"ns.functions.{namespace}.{func_name}"
        else:
            # Auto-generate from module
            module_parts = module.split('.')
            ns_path = f"ns.functions.{'.'.join(module_parts[2:])}.{func_name}"
        
        # Extract function metadata
        sig = inspect.signature(func)
        params = {
            param.name: {
                "type": str(param.annotation) if param.annotation != param.empty else "Any",
                "default": param.default if param.default != param.empty else None
            }
            for param in sig.parameters.values()
        }
        
        # Create metadata
        metadata = FunctionMetadata(
            name=func_name,
            module=module,
            signature=str(sig),
            docstring=inspect.getdoc(func),
            parameters=params,
            return_type=str(sig.return_annotation) if sig.return_annotation != sig.empty else None,
            tags=tags or set(),
            permissions=permissions or {"*": {FunctionPermission.EXECUTE}},
            ns_path=ns_path
        )
        
        # Create registered function
        registered = RegisteredFunction(
            metadata=metadata,
            func=func
        )
        
        # Store in registry
        self.functions[func_name] = registered
        self.namespace_map[ns_path] = func_name
        
        return registered
```

### Function Decorator

```python
from functools import wraps

# Global registry instance
_function_registry = FunctionRegistry()

def register_function(
    name: Optional[str] = None,
    namespace: Optional[str] = None, 
    tags: Optional[Set[str]] = None,
    permissions: Optional[Dict[str, Set[FunctionPermission]]] = None
):
    """Decorator to register functions automatically"""
    def decorator(func: Callable) -> Callable:
        # Register the function
        _function_registry.register(
            func=func,
            name=name,
            namespace=namespace,
            tags=tags,
            permissions=permissions
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Track execution
            registered = _function_registry.functions[name or func.__name__]
            registered.call_count += 1
            registered.last_called = datetime.now()
            
            # Check permissions (simplified)
            # In production, would check against current user/role
            
            # Execute function
            return func(*args, **kwargs)
            
        return wrapper
    return decorator
```

### Usage Examples

```python
# Simple registration
@register_function()
def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax for given amount and rate"""
    return amount * rate

# With namespace and tags
@register_function(
    namespace="finance.calculations",
    tags={"finance", "calculation", "tax"}
)
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: int,
    compound_frequency: int = 12
) -> float:
    """Calculate compound interest"""
    return principal * (1 + rate/compound_frequency) ** (compound_frequency * time)

# With permissions
@register_function(
    namespace="admin.operations",
    permissions={
        "admin": {FunctionPermission.EXECUTE, FunctionPermission.MODIFY},
        "user": {FunctionPermission.VIEW}
    }
)
def reset_user_password(user_id: str, new_password: str) -> bool:
    """Reset user password - admin only"""
    # Implementation
    pass

# Access via namespace
func = _function_registry.get_by_namespace("ns.functions.finance.calculations.calculate_compound_interest")
result = func(1000, 0.05, 2)
```

### Function Discovery

```python
class FunctionRegistry:
    """Extended with discovery methods"""
    
    def find_by_tag(self, tag: str) -> List[RegisteredFunction]:
        """Find all functions with a specific tag"""
        return [
            func for func in self.functions.values()
            if tag in func.metadata.tags
        ]
    
    def find_by_namespace(self, pattern: str) -> List[RegisteredFunction]:
        """Find functions matching namespace pattern"""
        import fnmatch
        matching_paths = [
            path for path in self.namespace_map.keys()
            if fnmatch.fnmatch(path, pattern)
        ]
        return [
            self.functions[self.namespace_map[path]]
            for path in matching_paths
        ]
    
    def get_by_namespace(self, ns_path: str) -> Optional[RegisteredFunction]:
        """Get function by exact namespace path"""
        func_name = self.namespace_map.get(ns_path)
        return self.functions.get(func_name) if func_name else None
```

## Model Registry

### Architecture

```python
from typing import Type, Dict, Optional, List, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ModelSource(Enum):
    """Source of model registration"""
    STATIC = "static"      # Imported from code
    DYNAMIC = "dynamic"    # Generated by Inspector
    EXTERNAL = "external"  # From external schemas

class ModelPermission(Enum):
    """Model access permissions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"

class ModelMetadata(BaseModel):
    """Metadata for registered models"""
    name: str
    source: ModelSource
    module: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    schema: Dict[str, Any]  # JSON schema representation
    ui_schema: Optional[Dict[str, Any]] = None
    permissions: Dict[str, Set[ModelPermission]] = Field(default_factory=dict)
    tags: Set[str] = Field(default_factory=set)
    ns_path: str  # e.g., "ns.models.user.Profile"
    description: Optional[str] = None
    
class RegisteredModel(XObjPrototype):
    """A registered model with metadata"""
    metadata: ModelMetadata
    model_class: Type[BaseModel]
    instances: List[str] = Field(default_factory=list)  # Track instance IDs
    ns: str = "registry.models"

class ModelRegistry:
    """Registry for Pydantic models"""
    
    def __init__(self):
        self.models: Dict[str, RegisteredModel] = {}
        self.namespace_map: Dict[str, str] = {}  # ns_path -> model_name
        self.version_history: Dict[str, List[ModelMetadata]] = {}  # Track versions
        
    def register(
        self,
        model_class: Type[BaseModel],
        name: Optional[str] = None,
        source: ModelSource = ModelSource.STATIC,
        namespace: Optional[str] = None,
        permissions: Optional[Dict[str, Set[ModelPermission]]] = None,
        tags: Optional[Set[str]] = None
    ) -> RegisteredModel:
        """Register a model with metadata"""
        model_name = name or model_class.__name__
        
        # Build namespace path
        if namespace:
            ns_path = f"ns.models.{namespace}.{model_name}"
        else:
            # Auto-generate from module if static
            if source == ModelSource.STATIC and hasattr(model_class, '__module__'):
                module_parts = model_class.__module__.split('.')
                ns_path = f"ns.models.{'.'.join(module_parts[2:])}.{model_name}"
            else:
                ns_path = f"ns.models.dynamic.{model_name}"
        
        # Generate JSON schema
        schema = model_class.model_json_schema()
        
        # Generate UI schema
        ui_schema = self._generate_ui_schema(model_class)
        
        # Create metadata
        metadata = ModelMetadata(
            name=model_name,
            source=source,
            module=model_class.__module__ if hasattr(model_class, '__module__') else None,
            schema=schema,
            ui_schema=ui_schema,
            permissions=permissions or {"*": {ModelPermission.READ, ModelPermission.LIST}},
            tags=tags or set(),
            ns_path=ns_path,
            description=model_class.__doc__
        )
        
        # Create registered model
        registered = RegisteredModel(
            metadata=metadata,
            model_class=model_class
        )
        
        # Store in registry
        self.models[model_name] = registered
        self.namespace_map[ns_path] = model_name
        
        # Track version history
        if model_name not in self.version_history:
            self.version_history[model_name] = []
        self.version_history[model_name].append(metadata)
        
        return registered
    
    def _generate_ui_schema(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
        """Generate UI schema from model"""
        # Similar to UISchemaGenerator from XMODELS
        ui_schema = {"fields": {}}
        
        for field_name, field_info in model_class.model_fields.items():
            widget_type = self._detect_widget(field_name, field_info)
            ui_schema["fields"][field_name] = {
                "widget": widget_type,
                "label": field_name.replace("_", " ").title(),
                "required": field_info.is_required(),
                "description": field_info.description
            }
        
        return ui_schema
    
    def _detect_widget(self, name: str, field_info) -> str:
        """Detect appropriate widget based on field"""
        name_lower = name.lower()
        
        # Name-based detection
        if "password" in name_lower:
            return "password"
        if "email" in name_lower:
            return "email"
        if "url" in name_lower or "link" in name_lower:
            return "url"
        if "description" in name_lower or "notes" in name_lower:
            return "textarea"
        
        # Type-based detection
        if field_info.annotation == bool:
            return "toggle"
        if field_info.annotation in (int, float):
            return "number"
        if field_info.annotation == datetime:
            return "datetime"
            
        return "text"
```

### Dynamic Model Registration

```python
class ModelRegistry:
    """Extended with dynamic model support"""
    
    async def register_from_inspector(
        self,
        resource: 'XResource',
        collection_name: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> RegisteredModel:
        """Register a model generated by Inspector"""
        from src.server.core.inspector import XInspector
        
        # Use Inspector to discover and generate model
        inspector = XInspector(resource)
        model_class = await inspector.generate_model(collection_name=collection_name)
        
        # Register as dynamic model
        return self.register(
            model_class=model_class,
            name=collection_name or model_class.__name__,
            source=ModelSource.DYNAMIC,
            namespace=namespace or f"dynamic.{resource.metadata.name}",
            tags={"dynamic", "inspector-generated", resource.get_type()}
        )
    
    def register_static_model(
        self,
        model_class: Type[BaseModel],
        **kwargs
    ) -> RegisteredModel:
        """Register a static model from code"""
        return self.register(
            model_class=model_class,
            source=ModelSource.STATIC,
            **kwargs
        )
```

### Model Versioning

```python
class ModelEvolution(BaseModel):
    """Track model schema changes"""
    model_name: str
    from_version: str
    to_version: str
    changes: List[Dict[str, Any]]
    migration_script: Optional[str] = None
    applied_at: Optional[datetime] = None

class ModelRegistry:
    """Extended with versioning support"""
    
    def update_model(
        self,
        model_name: str,
        new_model_class: Type[BaseModel],
        version: str,
        migration_script: Optional[str] = None
    ) -> RegisteredModel:
        """Update an existing model with new version"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        old_model = self.models[model_name]
        old_schema = old_model.metadata.schema
        new_schema = new_model_class.model_json_schema()
        
        # Detect changes
        changes = self._detect_schema_changes(old_schema, new_schema)
        
        # Create evolution record
        evolution = ModelEvolution(
            model_name=model_name,
            from_version=old_model.metadata.version,
            to_version=version,
            changes=changes,
            migration_script=migration_script
        )
        
        # Update model
        old_model.metadata.version = version
        old_model.metadata.schema = new_schema
        old_model.metadata.updated_at = datetime.now()
        old_model.model_class = new_model_class
        
        # Store in version history
        self.version_history[model_name].append(old_model.metadata.model_copy())
        
        return old_model
    
    def _detect_schema_changes(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect changes between schemas"""
        changes = []
        
        old_props = old_schema.get("properties", {})
        new_props = new_schema.get("properties", {})
        
        # Added fields
        for field in set(new_props) - set(old_props):
            changes.append({
                "type": "field_added",
                "field": field,
                "schema": new_props[field]
            })
        
        # Removed fields
        for field in set(old_props) - set(new_props):
            changes.append({
                "type": "field_removed",
                "field": field
            })
        
        # Modified fields
        for field in set(old_props) & set(new_props):
            if old_props[field] != new_props[field]:
                changes.append({
                    "type": "field_modified",
                    "field": field,
                    "old": old_props[field],
                    "new": new_props[field]
                })
        
        return changes
```

### Model Usage Examples

```python
# Static model registration
from pydantic import BaseModel, Field

class User(BaseModel):
    """User model"""
    id: str
    name: str = Field(min_length=1, max_length=100)
    email: str
    is_active: bool = True

# Register static model
model_registry = ModelRegistry()
model_registry.register_static_model(
    User,
    namespace="auth",
    tags={"user", "authentication"},
    permissions={
        "admin": {ModelPermission.CREATE, ModelPermission.READ, 
                 ModelPermission.UPDATE, ModelPermission.DELETE},
        "user": {ModelPermission.READ}
    }
)

# Dynamic model registration from database
mongo_resource = ResourceFactory.create("mongodb", ...)
await mongo_resource.connect()

# Register model from database collection
registered = await model_registry.register_from_inspector(
    resource=mongo_resource,
    collection_name="products",
    namespace="shop"
)

# Access model class
ProductModel = registered.model_class
product = ProductModel(name="Laptop", price=999.99)

# Find models by tag
auth_models = model_registry.find_by_tag("authentication")

# Get model by namespace
user_model = model_registry.get_by_namespace("ns.models.auth.User")
```

## Type Registry

### Architecture

```python
from typing import Type, Dict, Any, Callable, Optional, List
from pydantic import BaseModel, Field, validator
from pydantic.types import ConstrainedStr, ConstrainedInt
from datetime import datetime

class TypeCategory(Enum):
    """Categories of custom types"""
    CONSTRAINT = "constraint"    # Constrained types (e.g., PositiveInt)
    COMPOSITE = "composite"      # Composite types (e.g., Address)
    VALIDATOR = "validator"      # Custom validators
    TRANSFORMER = "transformer"  # Type transformers

class TypeMetadata(BaseModel):
    """Metadata for registered types"""
    name: str
    category: TypeCategory
    base_type: str  # Python base type
    constraints: Dict[str, Any] = Field(default_factory=dict)
    validator_func: Optional[Callable] = None
    transformer_func: Optional[Callable] = None
    description: Optional[str] = None
    examples: List[Any] = Field(default_factory=list)
    ns_path: str  # e.g., "ns.types.constraints.PositiveInt"
    tags: Set[str] = Field(default_factory=set)
    registered_at: datetime = Field(default_factory=datetime.now)

class RegisteredType(XObjPrototype):
    """A registered custom type"""
    metadata: TypeMetadata
    type_class: Type
    usage_count: int = 0
    ns: str = "registry.types"

class TypeRegistry:
    """Registry for custom types and validators"""
    
    def __init__(self):
        self.types: Dict[str, RegisteredType] = {}
        self.namespace_map: Dict[str, str] = {}
        self.validators: Dict[str, List[Callable]] = {}  # type_name -> validators
        
    def register_type(
        self,
        type_class: Type,
        name: Optional[str] = None,
        category: TypeCategory = TypeCategory.CONSTRAINT,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        examples: Optional[List[Any]] = None,
        tags: Optional[Set[str]] = None
    ) -> RegisteredType:
        """Register a custom type"""
        type_name = name or type_class.__name__
        
        # Build namespace path
        if namespace:
            ns_path = f"ns.types.{namespace}.{type_name}"
        else:
            ns_path = f"ns.types.{category.value}.{type_name}"
        
        # Extract type information
        base_type = self._get_base_type(type_class)
        constraints = self._extract_constraints(type_class)
        
        # Create metadata
        metadata = TypeMetadata(
            name=type_name,
            category=category,
            base_type=base_type,
            constraints=constraints,
            description=description or type_class.__doc__,
            examples=examples or [],
            ns_path=ns_path,
            tags=tags or set()
        )
        
        # Create registered type
        registered = RegisteredType(
            metadata=metadata,
            type_class=type_class
        )
        
        # Store in registry
        self.types[type_name] = registered
        self.namespace_map[ns_path] = type_name
        
        return registered
    
    def _get_base_type(self, type_class: Type) -> str:
        """Extract base Python type"""
        if hasattr(type_class, '__origin__'):
            return str(type_class.__origin__)
        elif hasattr(type_class, '__bases__'):
            for base in type_class.__bases__:
                if base.__module__ == 'builtins':
                    return base.__name__
        return str(type(type_class))
    
    def _extract_constraints(self, type_class: Type) -> Dict[str, Any]:
        """Extract constraints from type"""
        constraints = {}
        
        # For Pydantic constrained types
        if hasattr(type_class, '__fields__'):
            for attr in ['gt', 'ge', 'lt', 'le', 'min_length', 'max_length', 'regex']:
                if hasattr(type_class, attr):
                    value = getattr(type_class, attr)
                    if value is not None:
                        constraints[attr] = value
        
        return constraints
```

### Custom Type Examples

```python
# Constraint Types
class PositiveInt(ConstrainedInt):
    """Integer that must be positive"""
    gt = 0

class Email(ConstrainedStr):
    """Valid email address"""
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

class PhoneNumber(ConstrainedStr):
    """Valid phone number"""
    regex = r'^\+?1?\d{10,14}$'
    min_length = 10
    max_length = 15

# Register constraint types
type_registry = TypeRegistry()

type_registry.register_type(
    PositiveInt,
    category=TypeCategory.CONSTRAINT,
    description="Integer greater than zero",
    examples=[1, 100, 999],
    tags={"numeric", "validation"}
)

type_registry.register_type(
    Email,
    category=TypeCategory.CONSTRAINT,
    namespace="contact",
    examples=["user@example.com", "admin@company.org"],
    tags={"contact", "validation"}
)
```

### Validator Registry

```python
class TypeRegistry:
    """Extended with validator support"""
    
    def register_validator(
        self,
        type_name: str,
        validator_func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """Register a validator for a type"""
        if type_name not in self.validators:
            self.validators[type_name] = []
        
        # Wrap validator with metadata
        validator_func._name = name or validator_func.__name__
        validator_func._description = description
        
        self.validators[type_name].append(validator_func)
    
    def create_validated_type(
        self,
        base_type: Type,
        validators: List[Callable],
        name: str,
        namespace: Optional[str] = None
    ) -> Type:
        """Create a new type with validators"""
        # Dynamic type creation with validators
        class_dict = {
            '__annotations__': {'value': base_type},
            '__module__': 'registry.generated'
        }
        
        # Add validators
        for i, validator_func in enumerate(validators):
            class_dict[f'validate_{i}'] = validator(f'value', allow_reuse=True)(validator_func)
        
        # Create new type
        ValidatedType = type(name, (BaseModel,), class_dict)
        
        # Register it
        self.register_type(
            ValidatedType,
            name=name,
            category=TypeCategory.VALIDATOR,
            namespace=namespace
        )
        
        return ValidatedType
```

### Validator Examples

```python
# Custom validators
def validate_credit_card(value: str) -> str:
    """Validate credit card number using Luhn algorithm"""
    # Remove spaces and dashes
    value = value.replace(' ', '').replace('-', '')
    
    if not value.isdigit():
        raise ValueError("Credit card must contain only digits")
    
    # Luhn algorithm
    total = 0
    for i, digit in enumerate(reversed(value)):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    
    if total % 10 != 0:
        raise ValueError("Invalid credit card number")
    
    return value

def validate_password_strength(value: str) -> str:
    """Validate password meets security requirements"""
    if len(value) < 8:
        raise ValueError("Password must be at least 8 characters")
    if not any(c.isupper() for c in value):
        raise ValueError("Password must contain uppercase letter")
    if not any(c.islower() for c in value):
        raise ValueError("Password must contain lowercase letter")
    if not any(c.isdigit() for c in value):
        raise ValueError("Password must contain digit")
    if not any(c in "!@#$%^&*" for c in value):
        raise ValueError("Password must contain special character")
    return value

# Register validators
type_registry.register_validator(
    "str",
    validate_credit_card,
    name="credit_card_validator",
    description="Validates credit card using Luhn algorithm"
)

type_registry.register_validator(
    "str",
    validate_password_strength,
    name="strong_password",
    description="Ensures password meets security requirements"
)

# Create validated types
CreditCard = type_registry.create_validated_type(
    base_type=str,
    validators=[validate_credit_card],
    name="CreditCard",
    namespace="payment"
)

StrongPassword = type_registry.create_validated_type(
    base_type=str,
    validators=[validate_password_strength],
    name="StrongPassword",
    namespace="security"
)
```

### Composite Types

```python
class Address(BaseModel):
    """Composite address type"""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"
    
    @validator('zip_code')
    def validate_zip(cls, v):
        if not re.match(r'^\d{5}(-\d{4})?$', v):
            raise ValueError('Invalid ZIP code format')
        return v

class Money(BaseModel):
    """Composite money type with currency"""
    amount: Decimal
    currency: str = "USD"
    
    @validator('amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount cannot be negative')
        return v
    
    @validator('currency')
    def validate_currency(cls, v):
        valid_currencies = ["USD", "EUR", "GBP", "JPY", "CNY"]
        if v not in valid_currencies:
            raise ValueError(f'Currency must be one of {valid_currencies}')
        return v
    
    def to_string(self) -> str:
        """Format money as string"""
        return f"{self.currency} {self.amount:,.2f}"

# Register composite types
type_registry.register_type(
    Address,
    category=TypeCategory.COMPOSITE,
    namespace="common",
    examples=[
        {"street": "123 Main St", "city": "Boston", "state": "MA", "zip_code": "02101"}
    ],
    tags={"address", "location", "composite"}
)

type_registry.register_type(
    Money,
    category=TypeCategory.COMPOSITE,
    namespace="financial",
    examples=[
        {"amount": 100.50, "currency": "USD"},
        {"amount": 85.00, "currency": "EUR"}
    ],
    tags={"money", "currency", "financial"}
)
```

### Type Transformers

```python
class TypeRegistry:
    """Extended with transformer support"""
    
    def register_transformer(
        self,
        from_type: Type,
        to_type: Type,
        transformer_func: Callable,
        name: Optional[str] = None
    ) -> None:
        """Register a type transformer"""
        transformer_name = name or f"{from_type.__name__}_to_{to_type.__name__}"
        
        # Create transformer type
        transformer_type = type(
            transformer_name,
            (BaseModel,),
            {
                'transform': staticmethod(transformer_func),
                '__module__': 'registry.transformers'
            }
        )
        
        self.register_type(
            transformer_type,
            name=transformer_name,
            category=TypeCategory.TRANSFORMER,
            namespace="transformers"
        )

# Example transformers
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9/5) + 32

def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

# Register transformers
type_registry.register_transformer(
    float, float,
    celsius_to_fahrenheit,
    "CelsiusToFahrenheit"
)

type_registry.register_transformer(
    str, str,
    snake_to_camel,
    "SnakeToCamel"
)
```

## Import Control via models/__init__.py

### Overview

The models folder serves as the gatekeeper for what models and types are available to the application. By controlling exports through `__init__.py`, we maintain discipline over what can be accessed and prevent unauthorized model usage.

### Architecture

```python
# src/models/__init__.py
"""
Models module - Central control point for all models and types.

This module controls what models are available to the rest of the application.
Only models explicitly exported here can be imported and used.
"""

from typing import Dict, Type, List, Any
from pydantic import BaseModel

# Import registries
from src.core.registry import (
    model_registry,
    type_registry,
    function_registry
)

# Static model imports
from .user import User, UserProfile, UserSettings
from .product import Product, ProductCategory, Inventory
from .order import Order, OrderItem, OrderStatus
from .common import Address, Money, PhoneNumber

# Register static models on import
_static_models = {
    # User models
    "User": User,
    "UserProfile": UserProfile,
    "UserSettings": UserSettings,
    
    # Product models
    "Product": Product,
    "ProductCategory": ProductCategory,
    "Inventory": Inventory,
    
    # Order models
    "Order": Order,
    "OrderItem": OrderItem,
    "OrderStatus": OrderStatus,
    
    # Common types
    "Address": Address,
    "Money": Money,
    "PhoneNumber": PhoneNumber
}

# Auto-register all static models
for name, model_class in _static_models.items():
    model_registry.register_static_model(
        model_class,
        name=name,
        namespace=model_class.__module__.split('.')[-1]
    )

# Export control - what's available to import
__all__ = [
    # Registries
    "model_registry",
    "type_registry",
    "function_registry",
    
    # User models
    "User",
    "UserProfile", 
    "UserSettings",
    
    # Product models
    "Product",
    "ProductCategory",
    "Inventory",
    
    # Order models
    "Order",
    "OrderItem",
    "OrderStatus",
    
    # Common types
    "Address",
    "Money",
    "PhoneNumber",
    
    # Helper functions
    "get_model",
    "list_models",
    "register_dynamic_model"
]

def get_model(name: str) -> Type[BaseModel]:
    """
    Get a model by name from the registry.
    
    Args:
        name: Model name
        
    Returns:
        Model class
        
    Raises:
        KeyError: If model not found
    """
    if name in _static_models:
        return _static_models[name]
    
    # Check registry for dynamic models
    registered = model_registry.models.get(name)
    if registered:
        return registered.model_class
    
    raise KeyError(f"Model '{name}' not found in registry")

def list_models(category: str = None) -> List[str]:
    """
    List all available models.
    
    Args:
        category: Filter by category (static/dynamic)
        
    Returns:
        List of model names
    """
    models = []
    
    if category in (None, "static"):
        models.extend(_static_models.keys())
    
    if category in (None, "dynamic"):
        dynamic_models = [
            name for name, reg in model_registry.models.items()
            if reg.metadata.source == ModelSource.DYNAMIC
        ]
        models.extend(dynamic_models)
    
    return sorted(models)

async def register_dynamic_model(
    resource: 'XResource',
    collection_name: str,
    auto_export: bool = False
) -> Type[BaseModel]:
    """
    Register a dynamic model from a resource.
    
    Args:
        resource: Data source resource
        collection_name: Collection/table name
        auto_export: Whether to add to __all__ exports
        
    Returns:
        Generated model class
    """
    # Register via model registry
    registered = await model_registry.register_from_inspector(
        resource=resource,
        collection_name=collection_name
    )
    
    # Optionally add to exports
    if auto_export:
        globals()[collection_name] = registered.model_class
        __all__.append(collection_name)
    
    return registered.model_class
```

### Model Organization

```python
# src/models/user.py
"""User-related models"""

from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional, List
from ..core.xobj_prototype import XObjPrototype

class User(XObjPrototype):
    """User account model"""
    id: str
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    ns: str = "models.user"

class UserProfile(BaseModel):
    """User profile information"""
    user_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = Field(None, max_length=500)
    preferences: Dict[str, Any] = Field(default_factory=dict)

# src/models/common.py
"""Common shared types"""

from pydantic import BaseModel, Field, validator
from decimal import Decimal
import re

class Address(BaseModel):
    """Standard address type"""
    street: str
    city: str
    state: str = Field(min_length=2, max_length=2)
    zip_code: str
    country: str = "USA"
    
    @validator('zip_code')
    def validate_zip(cls, v):
        if not re.match(r'^\d{5}(-\d{4})?$', v):
            raise ValueError('Invalid ZIP code')
        return v

class Money(BaseModel):
    """Money with currency"""
    amount: Decimal = Field(decimal_places=2)
    currency: str = Field(default="USD", regex="^[A-Z]{3}$")
```

### Import Discipline

```python
# Example of controlled imports in application code

# ✅ CORRECT - Import from models module
from src.models import User, Product, Order
from src.models import model_registry, get_model

# ❌ WRONG - Direct import bypassing control
from src.models.user import User  # Don't do this!

# ✅ CORRECT - Get dynamic model
CustomerModel = get_model("Customer")

# ✅ CORRECT - List available models
available_models = list_models()
print(f"Available models: {available_models}")
```

### Registry Integration

```python
# src/models/__init__.py continued

class ModelAccessControl:
    """Control access to models based on permissions"""
    
    def __init__(self):
        self.access_log: List[Dict[str, Any]] = []
    
    def check_access(
        self,
        model_name: str,
        operation: str,
        user_role: str = "anonymous"
    ) -> bool:
        """Check if user role can access model"""
        # Get model from registry
        registered = model_registry.models.get(model_name)
        if not registered:
            return False
        
        # Check permissions
        permissions = registered.metadata.permissions.get(user_role, set())
        allowed = ModelPermission[operation.upper()] in permissions
        
        # Log access attempt
        self.access_log.append({
            "timestamp": datetime.now(),
            "model": model_name,
            "operation": operation,
            "role": user_role,
            "allowed": allowed
        })
        
        return allowed
    
    def get_accessible_models(self, user_role: str) -> List[str]:
        """Get list of models accessible to user role"""
        accessible = []
        
        for name, registered in model_registry.models.items():
            permissions = registered.metadata.permissions.get(user_role, set())
            if permissions:
                accessible.append(name)
        
        return accessible

# Create global access control
_access_control = ModelAccessControl()

def check_model_access(
    model_name: str,
    operation: str,
    user_role: str = "anonymous"
) -> bool:
    """Check if operation is allowed on model"""
    return _access_control.check_access(model_name, operation, user_role)
```

### Auto-Discovery Pattern

```python
# src/models/__init__.py - Auto-discovery section

import importlib
import pkgutil
from pathlib import Path

def auto_discover_models():
    """
    Automatically discover and register models in the models package.
    This should be called during application startup.
    """
    models_path = Path(__file__).parent
    
    # Iterate through all Python files in models directory
    for module_info in pkgutil.iter_modules([str(models_path)]):
        if module_info.name.startswith('_'):
            continue  # Skip private modules
        
        # Import module
        module = importlib.import_module(f'.{module_info.name}', package='src.models')
        
        # Find all BaseModel subclasses
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Check if it's a model class
            if (isinstance(attr, type) and 
                issubclass(attr, BaseModel) and 
                attr is not BaseModel and
                hasattr(attr, '__module__') and
                attr.__module__.startswith('src.models')):
                
                # Register if not already registered
                if attr.__name__ not in model_registry.models:
                    model_registry.register_static_model(
                        attr,
                        namespace=module_info.name
                    )

# Optional: Enable auto-discovery
# auto_discover_models()
```

### Testing Import Control

```python
# tests/test_model_imports.py

import pytest
from src.models import *

def test_static_models_available():
    """Test that static models are importable"""
    assert User is not None
    assert Product is not None
    assert Order is not None

def test_get_model_function():
    """Test model retrieval by name"""
    user_model = get_model("User")
    assert user_model is User
    
    with pytest.raises(KeyError):
        get_model("NonExistentModel")

def test_model_access_control():
    """Test access control for models"""
    # Admin can do everything
    assert check_model_access("User", "create", "admin") == True
    assert check_model_access("User", "delete", "admin") == True
    
    # Regular user limited
    assert check_model_access("User", "read", "user") == True
    assert check_model_access("User", "delete", "user") == False

def test_list_models():
    """Test model listing"""
    all_models = list_models()
    assert "User" in all_models
    assert "Product" in all_models
    
    static_only = list_models("static")
    assert len(static_only) > 0
```

## Resource Enrichment and Inspector Integration

### Overview

When Resources are connected, the registry can enrich registered models with live statistics, data previews, and schema information. This creates a dynamic view of the data that updates based on the actual resource state.

### Architecture

```python
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.server.core.xresource import XResource
from src.server.core.inspector import XInspector, InspectionResult

class ModelEnrichment(BaseModel):
    """Live enrichment data for a model"""
    model_name: str
    resource_name: str
    last_updated: datetime = Field(default_factory=datetime.now)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)
    categorical_fields: Dict[str, List[Any]] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    is_live: bool = True

class EnrichedRegistry:
    """Registry with resource enrichment capabilities"""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        type_registry: TypeRegistry,
        function_registry: FunctionRegistry
    ):
        self.model_registry = model_registry
        self.type_registry = type_registry
        self.function_registry = function_registry
        self.enrichments: Dict[str, ModelEnrichment] = {}
        self.connected_resources: Dict[str, XResource] = {}
    
    async def enrich_from_resource(
        self,
        resource: XResource,
        model_name: Optional[str] = None,
        auto_refresh: bool = True
    ) -> ModelEnrichment:
        """Enrich model with live data from resource"""
        # Connect resource if needed
        if not await resource.validate_connection():
            await resource.connect()
        
        # Store connected resource
        self.connected_resources[resource.metadata.name] = resource
        
        # Create inspector
        inspector = XInspector(resource)
        
        # Get inspection result
        inspection = await inspector.inspect()
        
        # If model not specified, try to find matching model
        if not model_name:
            model_name = self._find_matching_model(inspection)
        
        # Create enrichment
        enrichment = ModelEnrichment(
            model_name=model_name or inspection.model_name,
            resource_name=resource.metadata.name,
            statistics=inspection.statistics,
            sample_data=inspection.sample_data,
            categorical_fields={
                name: info.distinct_values
                for name, info in inspection.categorical_fields.items()
            },
            data_quality=await self._assess_data_quality(inspection)
        )
        
        # Store enrichment
        self.enrichments[model_name] = enrichment
        
        # Setup auto-refresh if requested
        if auto_refresh:
            asyncio.create_task(
                self._auto_refresh_enrichment(resource, model_name)
            )
        
        return enrichment
    
    def _find_matching_model(self, inspection: InspectionResult) -> Optional[str]:
        """Find a registered model matching the inspection schema"""
        inspection_props = set(inspection.schema.get("properties", {}).keys())
        
        for name, registered in self.model_registry.models.items():
            model_props = set(registered.metadata.schema.get("properties", {}).keys())
            
            # Calculate similarity
            intersection = inspection_props & model_props
            if len(intersection) / len(model_props) > 0.8:  # 80% match
                return name
        
        return None
    
    async def _assess_data_quality(self, inspection: InspectionResult) -> Dict[str, Any]:
        """Assess data quality from inspection"""
        stats = inspection.statistics
        total_records = stats.get("total_count", 0)
        
        quality = {
            "completeness": {},
            "validity": {},
            "uniqueness": {},
            "consistency": {}
        }
        
        # Assess completeness (null counts)
        for field, field_stats in stats.get("field_statistics", {}).items():
            null_count = field_stats.get("null_count", 0)
            completeness = 1 - (null_count / total_records) if total_records > 0 else 0
            quality["completeness"][field] = completeness
        
        # Assess uniqueness
        for field, field_stats in stats.get("field_statistics", {}).items():
            unique_count = field_stats.get("unique_count", 0)
            uniqueness = unique_count / total_records if total_records > 0 else 0
            quality["uniqueness"][field] = uniqueness
        
        return quality
    
    async def _auto_refresh_enrichment(
        self,
        resource: XResource,
        model_name: str,
        interval: int = 300  # 5 minutes
    ):
        """Auto-refresh enrichment data periodically"""
        while model_name in self.enrichments:
            await asyncio.sleep(interval)
            
            try:
                # Re-enrich
                await self.enrich_from_resource(resource, model_name, False)
            except Exception as e:
                print(f"Failed to refresh enrichment for {model_name}: {e}")
```

### Resource-Aware Model Operations

```python
class EnrichedRegistry:
    """Extended with resource-aware operations"""
    
    async def create_with_validation(
        self,
        model_name: str,
        data: Dict[str, Any],
        validate_against_resource: bool = True
    ) -> Any:
        """Create model instance with resource validation"""
        # Get model
        registered = self.model_registry.models.get(model_name)
        if not registered:
            raise ValueError(f"Model {model_name} not found")
        
        # Get enrichment if available
        enrichment = self.enrichments.get(model_name)
        
        if validate_against_resource and enrichment:
            # Validate categorical fields
            for field, value in data.items():
                if field in enrichment.categorical_fields:
                    valid_values = enrichment.categorical_fields[field]
                    if value not in valid_values:
                        raise ValueError(
                            f"Invalid value for {field}. "
                            f"Must be one of: {valid_values}"
                        )
        
        # Create instance
        return registered.model_class(**data)
    
    async def suggest_values(
        self,
        model_name: str,
        field_name: str
    ) -> Optional[List[Any]]:
        """Suggest valid values for a field based on resource data"""
        enrichment = self.enrichments.get(model_name)
        if not enrichment:
            return None
        
        # Return categorical values if available
        if field_name in enrichment.categorical_fields:
            return enrichment.categorical_fields[field_name]
        
        # Return sample values from preview
        sample_values = []
        for record in enrichment.sample_data[:10]:
            if field_name in record and record[field_name] not in sample_values:
                sample_values.append(record[field_name])
        
        return sample_values if sample_values else None
    
    def get_field_statistics(
        self,
        model_name: str,
        field_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific field"""
        enrichment = self.enrichments.get(model_name)
        if not enrichment:
            return None
        
        field_stats = enrichment.statistics.get("field_statistics", {}).get(field_name)
        if field_stats:
            # Add quality metrics
            field_stats["quality"] = {
                "completeness": enrichment.data_quality["completeness"].get(field_name),
                "uniqueness": enrichment.data_quality["uniqueness"].get(field_name)
            }
        
        return field_stats
```

### Inspector Integration Patterns

```python
class EnrichedRegistry:
    """Extended with Inspector integration"""
    
    async def discover_and_register(
        self,
        resource: XResource,
        namespace: Optional[str] = None
    ) -> List[str]:
        """Discover all schemas in resource and register as models"""
        registered_models = []
        
        # Get list of collections/tables
        collections = await resource.list_collections_or_tables()
        
        for collection in collections:
            try:
                # Register model from resource
                registered = await self.model_registry.register_from_inspector(
                    resource=resource,
                    collection_name=collection,
                    namespace=namespace or f"discovered.{resource.metadata.name}"
                )
                
                # Enrich with live data
                await self.enrich_from_resource(
                    resource=resource,
                    model_name=registered.metadata.name
                )
                
                registered_models.append(registered.metadata.name)
                
            except Exception as e:
                print(f"Failed to register {collection}: {e}")
        
        return registered_models
    
    async def sync_with_resource(
        self,
        model_name: str,
        resource: XResource
    ) -> Dict[str, Any]:
        """Sync model schema with resource schema"""
        # Get current model
        registered = self.model_registry.models.get(model_name)
        if not registered:
            raise ValueError(f"Model {model_name} not found")
        
        # Inspect resource
        inspector = XInspector(resource)
        inspection = await inspector.inspect()
        
        # Compare schemas
        current_schema = registered.metadata.schema
        resource_schema = inspection.schema
        
        # Detect changes
        changes = self._detect_schema_changes(current_schema, resource_schema)
        
        if changes:
            # Generate new model version
            new_model = await inspector.generate_model()
            
            # Update registry
            self.model_registry.update_model(
                model_name=model_name,
                new_model_class=new_model,
                version=self._increment_version(registered.metadata.version)
            )
        
        return {
            "synced": bool(changes),
            "changes": changes,
            "new_version": registered.metadata.version
        }
```

### Usage Examples

```python
# Create enriched registry
enriched_registry = EnrichedRegistry(
    model_registry=model_registry,
    type_registry=type_registry,
    function_registry=function_registry
)

# Connect to MongoDB resource
mongo_resource = ResourceFactory.create(
    "mongodb",
    connection_string="mongodb://localhost:27017/shop",
    database="shop"
)

# Enrich Product model with live data
enrichment = await enriched_registry.enrich_from_resource(
    resource=mongo_resource,
    model_name="Product"
)

# View enrichment data
print(f"Total products: {enrichment.statistics['total_count']}")
print(f"Categories: {enrichment.categorical_fields['category']}")
print(f"Data quality: {enrichment.data_quality}")

# Create product with validation
new_product = await enriched_registry.create_with_validation(
    "Product",
    {
        "name": "Laptop",
        "category": "Electronics",  # Validated against live data
        "price": 999.99
    }
)

# Get field suggestions
suggestions = await enriched_registry.suggest_values("Product", "category")
print(f"Valid categories: {suggestions}")

# Discover and register all models from resource
discovered = await enriched_registry.discover_and_register(
    resource=mongo_resource,
    namespace="shop"
)
print(f"Discovered models: {discovered}")

# Sync model with resource changes
sync_result = await enriched_registry.sync_with_resource(
    "Product",
    mongo_resource
)
if sync_result["synced"]:
    print(f"Model updated to version {sync_result['new_version']}")
    print(f"Changes: {sync_result['changes']}")
```

### Resource Delegation Pattern

```python
# Extension to XResource for Inspector convenience methods

class XResource:
    """Extended with Inspector delegation methods"""
    
    async def get_schema(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """Get schema via Inspector (convenience method)"""
        from src.server.core.inspector import XInspector
        inspector = XInspector(self)
        
        if collection:
            # Temporarily set collection context
            original = self.metadata.get("collection")
            self.metadata["collection"] = collection
            
        result = await inspector.discover_schema()
        
        if collection:
            # Restore original
            if original:
                self.metadata["collection"] = original
            else:
                del self.metadata["collection"]
        
        return result
    
    async def get_preview(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get data preview via Inspector"""
        from src.server.core.inspector import XInspector
        inspector = XInspector(self)
        return await inspector.preview_data(limit)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics via Inspector"""
        from src.server.core.inspector import XInspector
        inspector = XInspector(self)
        return await inspector.profile_data()
```

## Unified Registry Usage Examples

### Complete Registry Setup

```python
from src.core.registry import (
    FunctionRegistry,
    ModelRegistry,
    TypeRegistry,
    EnrichedRegistry
)
from src.server.core.xresource import ResourceFactory

# Initialize individual registries
function_registry = FunctionRegistry()
model_registry = ModelRegistry()
type_registry = TypeRegistry()

# Create unified enriched registry
registry = EnrichedRegistry(
    model_registry=model_registry,
    type_registry=type_registry,
    function_registry=function_registry
)

# Example: Complete workflow
async def setup_application_registry():
    """Setup complete application registry"""
    
    # 1. Register functions
    @function_registry.register(
        namespace="utils.validation",
        tags={"validation", "utility"}
    )
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))
    
    # 2. Register custom types
    type_registry.register_type(
        PositiveInt,
        category=TypeCategory.CONSTRAINT,
        tags={"numeric", "validation"}
    )
    
    # 3. Import and register static models
    from src.models import User, Product, Order
    
    # 4. Connect to data sources and discover dynamic models
    mongo_resource = ResourceFactory.create(
        "mongodb",
        connection_string=settings.MONGODB_URL,
        database="myapp"
    )
    
    # Discover and register models from database
    discovered = await registry.discover_and_register(
        resource=mongo_resource,
        namespace="dynamic"
    )
    
    print(f"Registered {len(discovered)} dynamic models")
    
    # 5. Enrich models with live data
    for model_name in ["User", "Product", "Order"]:
        await registry.enrich_from_resource(
            resource=mongo_resource,
            model_name=model_name
        )
    
    return registry

# Initialize registry on startup
registry = await setup_application_registry()
```

### Practical Usage Patterns

```python
# 1. Function execution with permissions
async def execute_function_with_auth(
    func_name: str,
    args: tuple,
    kwargs: dict,
    user_role: str = "user"
) -> Any:
    """Execute function with permission check"""
    
    # Get function from registry
    func = function_registry.functions.get(func_name)
    if not func:
        raise ValueError(f"Function {func_name} not found")
    
    # Check permissions
    permissions = func.metadata.permissions.get(user_role, set())
    if FunctionPermission.EXECUTE not in permissions:
        raise PermissionError(f"Role {user_role} cannot execute {func_name}")
    
    # Execute
    return func.func(*args, **kwargs)

# 2. Dynamic model creation with validation
async def create_record_with_validation(
    model_name: str,
    data: dict,
    user_role: str = "user"
) -> Any:
    """Create a record with full validation"""
    
    # Check model permissions
    if not registry.model_registry.check_access(model_name, "create", user_role):
        raise PermissionError(f"Cannot create {model_name}")
    
    # Get enrichment for validation
    enrichment = registry.enrichments.get(model_name)
    if enrichment:
        # Validate against live data
        for field, value in data.items():
            if field in enrichment.categorical_fields:
                valid_values = enrichment.categorical_fields[field]
                if value not in valid_values:
                    suggestions = ", ".join(str(v) for v in valid_values[:5])
                    raise ValueError(
                        f"Invalid {field}. Try: {suggestions}..."
                    )
    
    # Create instance
    model_class = registry.model_registry.models[model_name].model_class
    return model_class(**data)

# 3. Type validation workflow
def validate_with_custom_type(value: Any, type_name: str) -> Any:
    """Validate value against registered type"""
    
    registered_type = type_registry.types.get(type_name)
    if not registered_type:
        raise ValueError(f"Type {type_name} not found")
    
    # For constraint types
    if registered_type.metadata.category == TypeCategory.CONSTRAINT:
        type_class = registered_type.type_class
        try:
            return type_class(value)
        except ValueError as e:
            raise ValueError(f"Validation failed for {type_name}: {e}")
    
    # For composite types
    elif registered_type.metadata.category == TypeCategory.COMPOSITE:
        type_class = registered_type.type_class
        return type_class(**value) if isinstance(value, dict) else type_class(value)
    
    return value

# Usage
email = validate_with_custom_type("user@example.com", "Email")
money = validate_with_custom_type({"amount": 99.99, "currency": "USD"}, "Money")
```

## Unified Registry Usage

### Complete Example

```python
# src/core/registry.py
"""Unified registry combining all components"""

from typing import Optional
from src.server.core.xobj_prototype import XObjPrototype

class XRegistry(XObjPrototype):
    """
    Unified registry for functions, models, types, and resources.
    This is the main entry point for all registry operations.
    """
    
    def __init__(self):
        self.functions = FunctionRegistry()
        self.models = ModelRegistry()
        self.types = TypeRegistry()
        self.enriched = None  # Lazy initialization
        self.ns = "registry"
    
    def get_enriched(self) -> EnrichedRegistry:
        """Get or create enriched registry"""
        if not self.enriched:
            self.enriched = EnrichedRegistry(
                model_registry=self.models,
                type_registry=self.types,
                function_registry=self.functions
            )
        return self.enriched
    
    # Delegate common operations
    def register_function(self, *args, **kwargs):
        return self.functions.register(*args, **kwargs)
    
    def register_model(self, *args, **kwargs):
        return self.models.register(*args, **kwargs)
    
    def register_type(self, *args, **kwargs):
        return self.types.register_type(*args, **kwargs)

# Create global registry instance
registry = XRegistry()

# Export decorators
register_function = register_function  # From function registry
```

### Application Startup

```python
# src/main.py
"""Application startup with registry initialization"""

import asyncio
from src.core.registry import registry
from src.models import auto_discover_models
from src.server.core.xresource import ResourceFactory

async def initialize_registry():
    """Initialize registry on application startup"""
    
    # 1. Auto-discover and register models
    print("Discovering models...")
    auto_discover_models()
    
    # 2. Register custom types
    print("Registering custom types...")
    from src.types import register_all_types
    register_all_types(registry.types)
    
    # 3. Connect to primary data sources
    print("Connecting to data sources...")
    
    # MongoDB for main data
    mongo = ResourceFactory.create(
        "mongodb",
        name="main_db",
        connection_string=settings.MONGODB_URL,
        database=settings.MONGODB_DATABASE
    )
    await mongo.connect()
    
    # Enrich models with live data
    enriched = registry.get_enriched()
    for model_name in ["User", "Product", "Order"]:
        try:
            await enriched.enrich_from_resource(mongo, model_name)
            print(f"✓ Enriched {model_name} model")
        except Exception as e:
            print(f"✗ Failed to enrich {model_name}: {e}")
    
    # 4. Discover dynamic models
    print("Discovering dynamic models...")
    discovered = await enriched.discover_and_register(
        resource=mongo,
        namespace="dynamic"
    )
    print(f"Discovered {len(discovered)} dynamic models")
    
    return registry

# Run initialization
if __name__ == "__main__":
    registry = asyncio.run(initialize_registry())
    print(f"Registry initialized with:")
    print(f"  - {len(registry.functions.functions)} functions")
    print(f"  - {len(registry.models.models)} models")
    print(f"  - {len(registry.types.types)} types")
```

### Using the Registry

```python
# src/services/example_service.py
"""Example service using the unified registry"""

from src.core.registry import registry, register_function
from src.models import User, Product

@register_function(
    namespace="services.user",
    tags={"user", "validation"}
)
async def validate_user_email(email: str) -> bool:
    """Validate if email is already registered"""
    # Use enriched registry for live validation
    enriched = registry.get_enriched()
    
    # Get suggestions from live data
    existing_emails = await enriched.suggest_values("User", "email")
    
    return email not in (existing_emails or [])

@register_function(
    namespace="services.product",
    tags={"product", "creation"}
)
async def create_product_with_validation(data: dict) -> Product:
    """Create product with category validation"""
    enriched = registry.get_enriched()
    
    # Validate against live data
    product = await enriched.create_with_validation(
        "Product",
        data,
        validate_against_resource=True
    )
    
    return product

# Use registered functions
async def process_user_registration(user_data: dict):
    """Process user registration with validation"""
    
    # Get function from registry
    validate_email = registry.functions.get_by_namespace(
        "ns.functions.services.user.validate_user_email"
    )
    
    # Validate email
    if not await validate_email.func(user_data["email"]):
        raise ValueError("Email already registered")
    
    # Create user using model from registry
    UserModel = registry.models.get_by_namespace("ns.models.user.User").model_class
    user = UserModel(**user_data)
    
    return user
```

## Configuration

### Registry Settings

```toml
# settings.toml
[registry]
auto_discover_models = true
auto_register_functions = true
enable_enrichment = true
enrichment_refresh_interval = 300  # 5 minutes

[registry.functions]
default_permissions = ["execute"]
track_execution = true
execution_timeout = 30  # seconds

[registry.models]
default_permissions = ["read", "list"]
enable_versioning = true
max_version_history = 10
auto_generate_ui = true

[registry.types]
enable_validators = true
cache_validated_values = true
validation_cache_ttl = 3600  # 1 hour

[registry.enrichment]
auto_refresh = true
refresh_interval = 300  # 5 minutes
sample_size = 100
enable_categorical_detection = true
categorical_threshold = 100  # max distinct values

[registry.import_control]
models_path = "src/models"
types_path = "src/types"
auto_discover_on_startup = true
strict_mode = true  # Enforce __all__ exports
```

### Environment Variables

```bash
# .env
# Registry configuration
REGISTRY_AUTO_DISCOVER=true
REGISTRY_CACHE_ENABLED=true
REGISTRY_CACHE_TTL=3600

# Model registry
MODEL_REGISTRY_VERSIONING=true
MODEL_REGISTRY_DEFAULT_NAMESPACE=app

# Function registry
FUNCTION_REGISTRY_TRACK_CALLS=true
FUNCTION_REGISTRY_TIMEOUT=30

# Type registry
TYPE_REGISTRY_STRICT_MODE=true
TYPE_REGISTRY_VALIDATION_CACHE=true

# Enrichment
ENRICHMENT_AUTO_REFRESH=true
ENRICHMENT_INTERVAL=300
ENRICHMENT_SAMPLE_SIZE=100
```

### Programmatic Configuration

```python
# src/core/registry_config.py
"""Registry configuration"""

from dynaconf import Dynaconf
from pydantic import BaseModel
from typing import Dict, Set, Optional

class RegistryConfig(BaseModel):
    """Registry configuration model"""
    
    # General settings
    auto_discover_models: bool = True
    auto_register_functions: bool = True
    enable_enrichment: bool = True
    
    # Function registry
    function_defaults: Dict[str, Any] = {
        "permissions": ["execute"],
        "track_execution": True,
        "timeout": 30
    }
    
    # Model registry
    model_defaults: Dict[str, Any] = {
        "permissions": {"*": ["read", "list"]},
        "enable_versioning": True,
        "auto_generate_ui": True
    }
    
    # Type registry
    type_defaults: Dict[str, Any] = {
        "enable_validators": True,
        "cache_validated": True,
        "cache_ttl": 3600
    }
    
    # Enrichment settings
    enrichment_config: Dict[str, Any] = {
        "auto_refresh": True,
        "refresh_interval": 300,
        "sample_size": 100,
        "categorical_threshold": 100
    }

# Load settings
settings = Dynaconf(
    envvar_prefix="REGISTRY",
    settings_files=["settings.toml", ".secrets.toml"],
    environments=True,
    load_dotenv=True
)

# Create config instance
registry_config = RegistryConfig(
    auto_discover_models=settings.get("auto_discover_models", True),
    auto_register_functions=settings.get("auto_register_functions", True),
    enable_enrichment=settings.get("enable_enrichment", True),
    function_defaults=settings.get("functions", {}),
    model_defaults=settings.get("models", {}),
    type_defaults=settings.get("types", {}),
    enrichment_config=settings.get("enrichment", {})
)

# Apply configuration to registries
def configure_registries(
    function_registry: FunctionRegistry,
    model_registry: ModelRegistry,
    type_registry: TypeRegistry
) -> None:
    """Apply configuration to registries"""
    
    # Configure function registry
    if registry_config.function_defaults:
        function_registry.default_permissions = registry_config.function_defaults.get(
            "permissions", ["execute"]
        )
        function_registry.track_execution = registry_config.function_defaults.get(
            "track_execution", True
        )
    
    # Configure model registry
    if registry_config.model_defaults:
        model_registry.enable_versioning = registry_config.model_defaults.get(
            "enable_versioning", True
        )
        model_registry.auto_generate_ui = registry_config.model_defaults.get(
            "auto_generate_ui", True
        )
    
    # Configure type registry
    if registry_config.type_defaults:
        type_registry.enable_validators = registry_config.type_defaults.get(
            "enable_validators", True
        )
        type_registry.cache_ttl = registry_config.type_defaults.get(
            "cache_ttl", 3600
        )
```
# src/core/config.py
"""Registry configuration from environment"""

import os
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="XREGISTRY",
    settings_files=["settings.toml", ".secrets.toml"],
    environments=True,
    load_dotenv=True,
)

# Registry-specific settings
REGISTRY_NAMESPACE_PREFIX = settings.get("registry.namespace_prefix", "ns")
REGISTRY_CACHE_TTL = settings.get("registry.cache_ttl", 3600)
REGISTRY_AUTO_DISCOVER = settings.get("registry.auto_discover", True)

# Function registry settings
FUNCTION_MAX_CALL_HISTORY = settings.get("registry.functions.max_call_history", 1000)
FUNCTION_TRACK_PERFORMANCE = settings.get("registry.functions.track_performance", True)

# Model registry settings
MODEL_AUTO_REGISTER = settings.get("registry.models.auto_register_on_import", True)
MODEL_TRACK_VERSIONS = settings.get("registry.models.track_versions", True)
MODEL_MAX_VERSION_HISTORY = settings.get("registry.models.max_version_history", 10)

# Type registry settings
TYPE_VALIDATE_ON_REGISTER = settings.get("registry.types.validate_on_register", True)
TYPE_ALLOW_OVERRIDE = settings.get("registry.types.allow_override", False)

# Enrichment settings
ENRICHMENT_AUTO_REFRESH = settings.get("registry.enrichment.auto_refresh", True)
ENRICHMENT_REFRESH_INTERVAL = settings.get("registry.enrichment.refresh_interval", 300)
ENRICHMENT_SAMPLE_SIZE = settings.get("registry.enrichment.sample_size", 100)
```

### Registry Initialization with Config

```python
# src/core/registry.py
"""Registry with configuration support"""

from src.core.config import settings

class ConfiguredXRegistry(XRegistry):
    """Registry with configuration support"""
    
    def __init__(self):
        super().__init__()
        self._configure()
    
    def _configure(self):
        """Apply configuration settings"""
        # Configure function registry
        self.functions.max_call_history = settings.registry.functions.max_call_history
        self.functions.track_performance = settings.registry.functions.track_performance
        
        # Configure model registry
        self.models.auto_register = settings.registry.models.auto_register_on_import
        self.models.track_versions = settings.registry.models.track_versions
        self.models.max_version_history = settings.registry.models.max_version_history
        
        # Configure type registry
        self.types.validate_on_register = settings.registry.types.validate_on_register
        self.types.allow_override = settings.registry.types.allow_override
        
        # Configure namespace prefix
        self.namespace_prefix = settings.registry.namespace_prefix
    
    def get_enriched(self) -> EnrichedRegistry:
        """Get configured enriched registry"""
        if not self.enriched:
            self.enriched = EnrichedRegistry(
                model_registry=self.models,
                type_registry=self.types,
                function_registry=self.functions
            )
            
            # Apply enrichment configuration
            self.enriched.auto_refresh = settings.registry.enrichment.auto_refresh
            self.enriched.refresh_interval = settings.registry.enrichment.refresh_interval
            self.enriched.sample_size = settings.registry.enrichment.sample_size
            
        return self.enriched

# Create configured global instance
registry = ConfiguredXRegistry()
```

## Testing

### Unit Tests

```python
# tests/test_function_registry.py
import pytest
from src.core.registry import FunctionRegistry, FunctionPermission

@pytest.fixture
def function_registry():
    return FunctionRegistry()

def test_function_registration(function_registry):
    """Test basic function registration"""
    @function_registry.register(
        namespace="test",
        tags={"test", "example"}
    )
    def sample_function(x: int) -> int:
        """Sample function"""
        return x * 2
    
    # Verify registration
    assert "sample_function" in function_registry.functions
    registered = function_registry.functions["sample_function"]
    assert registered.metadata.namespace == "test"
    assert "test" in registered.metadata.tags

def test_function_permissions(function_registry):
    """Test function permission system"""
    @function_registry.register(
        permissions={
            "admin": {FunctionPermission.EXECUTE, FunctionPermission.MODIFY},
            "user": {FunctionPermission.VIEW}
        }
    )
    def secure_function():
        """Secure function"""
        pass
    
    registered = function_registry.functions["secure_function"]
    admin_perms = registered.metadata.permissions["admin"]
    assert FunctionPermission.EXECUTE in admin_perms
    assert FunctionPermission.MODIFY in admin_perms
    
    user_perms = registered.metadata.permissions["user"]
    assert FunctionPermission.VIEW in user_perms
    assert FunctionPermission.EXECUTE not in user_perms

def test_function_namespace_discovery(function_registry):
    """Test namespace-based function discovery"""
    # Register multiple functions
    @function_registry.register(namespace="math.basic")
    def add(a, b): return a + b
    
    @function_registry.register(namespace="math.basic")
    def subtract(a, b): return a - b
    
    @function_registry.register(namespace="math.advanced")
    def power(base, exp): return base ** exp
    
    # Find by namespace pattern
    basic_funcs = function_registry.find_by_namespace("ns.functions.math.basic.*")
    assert len(basic_funcs) == 2
    
    all_math = function_registry.find_by_namespace("ns.functions.math.*")
    assert len(all_math) == 3
```

### Model Registry Tests

```python
# tests/test_model_registry.py
import pytest
from pydantic import BaseModel, Field
from src.core.registry import ModelRegistry, ModelSource, ModelPermission

@pytest.fixture
def model_registry():
    return ModelRegistry()

class TestModel(BaseModel):
    """Test model"""
    name: str = Field(min_length=1)
    value: int = Field(gt=0)

def test_static_model_registration(model_registry):
    """Test registering static models"""
    registered = model_registry.register_static_model(
        TestModel,
        namespace="test",
        tags={"test", "example"}
    )
    
    assert registered.metadata.source == ModelSource.STATIC
    assert registered.metadata.name == "TestModel"
    assert "test" in registered.metadata.tags
    assert registered.model_class == TestModel

def test_model_versioning(model_registry):
    """Test model version tracking"""
    # Register initial version
    model_registry.register_static_model(TestModel, name="VersionedModel")
    
    # Update model
    class UpdatedModel(BaseModel):
        name: str
        value: int
        description: str = ""  # New field
    
    updated = model_registry.update_model(
        "VersionedModel",
        UpdatedModel,
        version="2.0.0"
    )
    
    assert updated.metadata.version == "2.0.0"
    assert len(model_registry.version_history["VersionedModel"]) == 2

def test_model_permissions(model_registry):
    """Test model access control"""
    registered = model_registry.register_static_model(
        TestModel,
        permissions={
            "admin": {ModelPermission.CREATE, ModelPermission.READ, 
                     ModelPermission.UPDATE, ModelPermission.DELETE},
            "user": {ModelPermission.READ}
        }
    )
    
    # Check permissions
    admin_perms = registered.metadata.permissions["admin"]
    assert ModelPermission.CREATE in admin_perms
    assert ModelPermission.DELETE in admin_perms
    
    user_perms = registered.metadata.permissions["user"]
    assert ModelPermission.READ in user_perms
    assert ModelPermission.CREATE not in user_perms

@pytest.mark.asyncio
async def test_dynamic_model_registration(model_registry):
    """Test registering models from Inspector"""
    from unittest.mock import Mock, AsyncMock
    
    # Mock resource and inspector
    mock_resource = Mock()
    mock_resource.metadata = {"name": "test_db"}
    mock_resource.get_type.return_value = "mongodb"
    
    # Mock inspector generate_model
    with patch('src.core.inspector.XInspector') as MockInspector:
        mock_inspector = MockInspector.return_value
        mock_inspector.generate_model = AsyncMock(return_value=TestModel)
        
        # Register from inspector
        registered = await model_registry.register_from_inspector(
            resource=mock_resource,
            collection_name="test_collection"
        )
        
        assert registered.metadata.source == ModelSource.DYNAMIC
        assert registered.metadata.name == "test_collection"
        assert "dynamic" in registered.metadata.tags
```

### Type Registry Tests

```python
# tests/test_type_registry.py
import pytest
from pydantic.types import ConstrainedInt, ConstrainedStr
from src.core.registry import TypeRegistry, TypeCategory

@pytest.fixture
def type_registry():
    return TypeRegistry()

class PositiveInt(ConstrainedInt):
    gt = 0

def test_constraint_type_registration(type_registry):
    """Test registering constraint types"""
    registered = type_registry.register_type(
        PositiveInt,
        category=TypeCategory.CONSTRAINT,
        examples=[1, 100, 999]
    )
    
    assert registered.metadata.category == TypeCategory.CONSTRAINT
    assert registered.metadata.base_type == "int"
    assert registered.metadata.constraints["gt"] == 0
    assert 100 in registered.metadata.examples

def test_validator_registration(type_registry):
    """Test registering validators"""
    def validate_even(value: int) -> int:
        if value % 2 != 0:
            raise ValueError("Must be even")
        return value
    
    type_registry.register_validator(
        "int",
        validate_even,
        name="even_validator",
        description="Ensures value is even"
    )
    
    assert "int" in type_registry.validators
    validators = type_registry.validators["int"]
    assert len(validators) == 1
    assert validators[0]._name == "even_validator"

def test_create_validated_type(type_registry):
    """Test creating types with validators"""
    def validate_range(value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("Must be between 0 and 100")
        return value
    
    # Create validated type
    PercentageType = type_registry.create_validated_type(
        base_type=int,
        validators=[validate_range],
        name="Percentage",
        namespace="common"
    )
    
    # Test validation
    valid = PercentageType(value=50)
    assert valid.value == 50
    
    with pytest.raises(ValueError):
        PercentageType(value=150)
```

### Integration Tests

```python
# tests/test_enriched_registry.py
import pytest
from src.core.registry import EnrichedRegistry, ModelRegistry, TypeRegistry, FunctionRegistry
from src.server.core.xresource import ResourceFactory

@pytest.fixture
async def enriched_registry():
    """Create enriched registry with test data"""
    model_registry = ModelRegistry()
    type_registry = TypeRegistry()
    function_registry = FunctionRegistry()
    
    registry = EnrichedRegistry(
        model_registry=model_registry,
        type_registry=type_registry,
        function_registry=function_registry
    )
    
    return registry

@pytest.mark.asyncio
async def test_resource_enrichment(enriched_registry):
    """Test enriching models with resource data"""
    from unittest.mock import Mock, AsyncMock
    
    # Mock resource
    mock_resource = Mock()
    mock_resource.metadata = {"name": "test_db"}
    mock_resource.validate_connection = AsyncMock(return_value=True)
    mock_resource.connect = AsyncMock()
    
    # Mock inspector
    with patch('src.core.inspector.XInspector') as MockInspector:
        mock_inspector = MockInspector.return_value
        mock_inspector.inspect = AsyncMock(return_value=Mock(
            model_name="TestModel",
            statistics={"total_count": 100},
            sample_data=[{"id": 1, "name": "test"}],
            categorical_fields={"status": Mock(distinct_values=["active", "inactive"])}
        ))
        
        # Enrich model
        enrichment = await enriched_registry.enrich_from_resource(
            resource=mock_resource,
            model_name="TestModel",
            auto_refresh=False
        )
        
        assert enrichment.model_name == "TestModel"
        assert enrichment.statistics["total_count"] == 100
        assert "status" in enrichment.categorical_fields
        assert "active" in enrichment.categorical_fields["status"]

@pytest.mark.asyncio
async def test_validation_with_enrichment(enriched_registry):
    """Test creating records with enrichment validation"""
    # Setup mock enrichment
    enriched_registry.enrichments["Product"] = Mock(
        categorical_fields={
            "category": ["Electronics", "Books", "Clothing"]
        }
    )
    
    # Setup mock model
    enriched_registry.model_registry.models["Product"] = Mock(
        model_class=Mock(side_effect=lambda **kwargs: kwargs)
    )
    
    # Valid creation
    product = await enriched_registry.create_with_validation(
        "Product",
        {"name": "Laptop", "category": "Electronics"}
    )
    assert product["category"] == "Electronics"
    
    # Invalid category
    with pytest.raises(ValueError) as exc:
        await enriched_registry.create_with_validation(
            "Product",
            {"name": "Item", "category": "InvalidCategory"}
        )
    assert "Invalid value for category" in str(exc.value)

def test_import_control_integration():
    """Test models folder import control"""
    from src.models import (
        User, Product, Order,  # Should work
        get_model, list_models, check_model_access
    )
    
    # Test get_model
    user_model = get_model("User")
    assert user_model is User
    
    # Test list_models
    all_models = list_models()
    assert "User" in all_models
    assert "Product" in all_models
    
    # Test access control
    assert check_model_access("User", "read", "user") == True
    assert check_model_access("User", "delete", "user") == False
    assert check_model_access("User", "delete", "admin") == True
```

### Performance Tests

```python
# tests/test_registry_performance.py
import pytest
import time
from src.core.registry import FunctionRegistry, ModelRegistry

@pytest.mark.performance
def test_function_registry_performance():
    """Test function registry performance with many functions"""
    registry = FunctionRegistry()
    
    # Register 1000 functions
    start_time = time.time()
    for i in range(1000):
        @registry.register(namespace=f"perf.test{i % 10}")
        def test_func():
            pass
    
    registration_time = time.time() - start_time
    assert registration_time < 1.0  # Should complete in under 1 second
    
    # Test namespace discovery performance
    start_time = time.time()
    results = registry.find_by_namespace("ns.functions.perf.test5.*")
    discovery_time = time.time() - start_time
    
    assert discovery_time < 0.1  # Should be very fast
    assert len(results) == 100  # Should find all test5 functions

@pytest.mark.performance
def test_model_registry_performance():
    """Test model registry with many models"""
    registry = ModelRegistry()
    
    # Create test models
    models = []
    for i in range(100):
        class_name = f"TestModel{i}"
        model_class = type(class_name, (BaseModel,), {
            "__annotations__": {"value": int},
            "__module__": "test"
        })
        models.append(model_class)
    
    # Register all models
    start_time = time.time()
    for model in models:
        registry.register_static_model(model)
    
    registration_time = time.time() - start_time
    assert registration_time < 0.5  # Should be fast
    
    # Test model lookup performance
    start_time = time.time()
    for i in range(100):
        model = registry.models.get(f"TestModel{i}")
        assert model is not None
    
    lookup_time = time.time() - start_time
    assert lookup_time < 0.01  # Lookups should be very fast
```

## API Endpoints

### Registry API

```python
# src/api/registry_endpoints.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.core.registry import registry

router = APIRouter(prefix="/api/registry", tags=["registry"])

# Request/Response Models
class FunctionInfo(BaseModel):
    """Function information response"""
    name: str
    namespace: str
    module: str
    signature: str
    docstring: Optional[str]
    tags: List[str]
    permissions: Dict[str, List[str]]
    call_count: int

class ModelInfo(BaseModel):
    """Model information response"""
    name: str
    source: str
    namespace: str
    version: str
    schema: Dict[str, Any]
    ui_schema: Optional[Dict[str, Any]]
    permissions: Dict[str, List[str]]

class TypeInfo(BaseModel):
    """Type information response"""
    name: str
    category: str
    base_type: str
    constraints: Dict[str, Any]
    examples: List[Any]

# Function Registry Endpoints
@router.get("/functions", response_model=List[FunctionInfo])
async def list_functions(
    namespace: Optional[str] = None,
    tag: Optional[str] = None
):
    """List registered functions"""
    if namespace:
        functions = registry.functions.find_by_namespace(namespace)
    elif tag:
        functions = registry.functions.find_by_tag(tag)
    else:
        functions = list(registry.functions.functions.values())
    
    return [
        FunctionInfo(
            name=func.metadata.name,
            namespace=func.metadata.ns_path,
            module=func.metadata.module,
            signature=func.metadata.signature,
            docstring=func.metadata.docstring,
            tags=list(func.metadata.tags),
            permissions={k: [p.value for p in v] for k, v in func.metadata.permissions.items()},
            call_count=func.call_count
        )
        for func in functions
    ]

@router.get("/functions/{name}", response_model=FunctionInfo)
async def get_function(name: str):
    """Get function details"""
    func = registry.functions.functions.get(name)
    if not func:
        raise HTTPException(404, f"Function {name} not found")
    
    return FunctionInfo(
        name=func.metadata.name,
        namespace=func.metadata.ns_path,
        module=func.metadata.module,
        signature=func.metadata.signature,
        docstring=func.metadata.docstring,
        tags=list(func.metadata.tags),
        permissions={k: [p.value for p in v] for k, v in func.metadata.permissions.items()},
        call_count=func.call_count
    )

@router.post("/functions/{name}/execute")
async def execute_function(
    name: str,
    args: List[Any] = [],
    kwargs: Dict[str, Any] = {},
    user_role: str = "user"
):
    """Execute a registered function"""
    func = registry.functions.functions.get(name)
    if not func:
        raise HTTPException(404, f"Function {name} not found")
    
    # Check permissions
    permissions = func.metadata.permissions.get(user_role, set())
    if FunctionPermission.EXECUTE not in permissions:
        raise HTTPException(403, f"Role {user_role} cannot execute {name}")
    
    try:
        result = func.func(*args, **kwargs)
        return {"result": result}
    except Exception as e:
        raise HTTPException(500, f"Function execution failed: {str(e)}")

# Model Registry Endpoints
@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    source: Optional[str] = None,
    tag: Optional[str] = None
):
    """List registered models"""
    models = registry.models.models.values()
    
    if source:
        models = [m for m in models if m.metadata.source.value == source]
    if tag:
        models = [m for m in models if tag in m.metadata.tags]
    
    return [
        ModelInfo(
            name=model.metadata.name,
            source=model.metadata.source.value,
            namespace=model.metadata.ns_path,
            version=model.metadata.version,
            schema=model.metadata.schema,
            ui_schema=model.metadata.ui_schema,
            permissions={k: [p.value for p in v] for k, v in model.metadata.permissions.items()}
        )
        for model in models
    ]

@router.get("/models/{name}", response_model=ModelInfo)
async def get_model(name: str):
    """Get model details"""
    model = registry.models.models.get(name)
    if not model:
        raise HTTPException(404, f"Model {name} not found")
    
    return ModelInfo(
        name=model.metadata.name,
        source=model.metadata.source.value,
        namespace=model.metadata.ns_path,
        version=model.metadata.version,
        schema=model.metadata.schema,
        ui_schema=model.metadata.ui_schema,
        permissions={k: [p.value for p in v] for k, v in model.metadata.permissions.items()}
    )

@router.get("/models/{name}/schema")
async def get_model_schema(name: str):
    """Get JSON schema for model"""
    model = registry.models.models.get(name)
    if not model:
        raise HTTPException(404, f"Model {name} not found")
    
    return model.metadata.schema

@router.get("/models/{name}/ui-schema")
async def get_model_ui_schema(name: str):
    """Get UI schema for model"""
    model = registry.models.models.get(name)
    if not model:
        raise HTTPException(404, f"Model {name} not found")
    
    return model.metadata.ui_schema

# Type Registry Endpoints
@router.get("/types", response_model=List[TypeInfo])
async def list_types(category: Optional[str] = None):
    """List registered types"""
    types = registry.types.types.values()
    
    if category:
        types = [t for t in types if t.metadata.category.value == category]
    
    return [
        TypeInfo(
            name=type_reg.metadata.name,
            category=type_reg.metadata.category.value,
            base_type=type_reg.metadata.base_type,
            constraints=type_reg.metadata.constraints,
            examples=type_reg.metadata.examples
        )
        for type_reg in types
    ]

@router.post("/types/{name}/validate")
async def validate_type(name: str, value: Any):
    """Validate value against registered type"""
    type_reg = registry.types.types.get(name)
    if not type_reg:
        raise HTTPException(404, f"Type {name} not found")
    
    try:
        # Validate using type class
        validated = type_reg.type_class(value)
        return {"valid": True, "value": validated}
    except Exception as e:
        return {"valid": False, "error": str(e)}
```

### Enrichment API

```python
# src/api/enrichment_endpoints.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from src.core.registry import registry

router = APIRouter(prefix="/api/enrichment", tags=["enrichment"])

@router.get("/models/{name}/enrichment")
async def get_model_enrichment(name: str):
    """Get enrichment data for a model"""
    enriched = registry.get_enriched()
    enrichment = enriched.enrichments.get(name)
    
    if not enrichment:
        raise HTTPException(404, f"No enrichment data for model {name}")
    
    return {
        "model_name": enrichment.model_name,
        "resource_name": enrichment.resource_name,
        "last_updated": enrichment.last_updated.isoformat(),
        "statistics": enrichment.statistics,
        "sample_data": enrichment.sample_data[:10],  # Limit sample
        "categorical_fields": enrichment.categorical_fields,
        "data_quality": enrichment.data_quality,
        "is_live": enrichment.is_live
    }

@router.get("/models/{name}/suggestions/{field}")
async def get_field_suggestions(name: str, field: str):
    """Get value suggestions for a field"""
    enriched = registry.get_enriched()
    suggestions = await enriched.suggest_values(name, field)
    
    if suggestions is None:
        raise HTTPException(404, f"No suggestions for {name}.{field}")
    
    return {"field": field, "suggestions": suggestions}

@router.get("/models/{name}/statistics/{field}")
async def get_field_statistics(name: str, field: str):
    """Get statistics for a specific field"""
    enriched = registry.get_enriched()
    stats = enriched.get_field_statistics(name, field)
    
    if stats is None:
        raise HTTPException(404, f"No statistics for {name}.{field}")
    
    return stats

@router.post("/models/{name}/validate")
async def validate_with_enrichment(
    name: str,
    data: Dict[str, Any],
    user_role: str = "user"
):
    """Validate data against model with enrichment"""
    enriched = registry.get_enriched()
    
    try:
        instance = await enriched.create_with_validation(
            name, data, validate_against_resource=True
        )
        return {"valid": True, "instance": instance}
    except ValueError as e:
        return {"valid": False, "error": str(e)}
    except PermissionError as e:
        raise HTTPException(403, str(e))

@router.post("/refresh/{name}")
async def refresh_enrichment(name: str):
    """Manually refresh enrichment data"""
    enriched = registry.get_enriched()
    
    if name not in enriched.enrichments:
        raise HTTPException(404, f"Model {name} not enriched")
    
    # Get resource
    enrichment = enriched.enrichments[name]
    resource_name = enrichment.resource_name
    
    if resource_name not in enriched.connected_resources:
        raise HTTPException(400, f"Resource {resource_name} not connected")
    
    resource = enriched.connected_resources[resource_name]
    
    # Re-enrich
    try:
        new_enrichment = await enriched.enrich_from_resource(
            resource, name, auto_refresh=False
        )
        return {"refreshed": True, "last_updated": new_enrichment.last_updated.isoformat()}
    except Exception as e:
        raise HTTPException(500, f"Refresh failed: {str(e)}")
```

### Discovery API

```python
# src/api/discovery_endpoints.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.server.core.xresource import ResourceFactory

router = APIRouter(prefix="/api/discovery", tags=["discovery"])

@router.post("/resources/{resource_name}/discover")
async def discover_models(
    resource_name: str,
    background_tasks: BackgroundTasks,
    namespace: Optional[str] = None
):
    """Discover and register models from a resource"""
    enriched = registry.get_enriched()
    
    # Check if resource exists
    if resource_name not in enriched.connected_resources:
        raise HTTPException(404, f"Resource {resource_name} not connected")
    
    resource = enriched.connected_resources[resource_name]
    
    # Discover models in background
    async def discover_task():
        discovered = await enriched.discover_and_register(
            resource=resource,
            namespace=namespace
        )
        return discovered
    
    background_tasks.add_task(discover_task)
    
    return {
        "status": "discovery_started",
        "resource": resource_name,
        "message": "Model discovery running in background"
    }

@router.get("/namespace/{pattern}")
async def search_namespace(pattern: str):
    """Search registry by namespace pattern"""
    results = {
        "functions": [],
        "models": [],
        "types": []
    }
    
    # Search functions
    funcs = registry.functions.find_by_namespace(pattern)
    results["functions"] = [f.metadata.name for f in funcs]
    
    # Search models (simple pattern matching)
    for name, model in registry.models.models.items():
        if pattern.replace("*", "") in model.metadata.ns_path:
            results["models"].append(name)
    
    # Search types
    for name, type_reg in registry.types.types.items():
        if pattern.replace("*", "") in type_reg.metadata.ns_path:
            results["types"].append(name)
    
    return results
```

## Summary

### What is XRegistry?

XRegistry is a comprehensive registry system that unifies the management of functions, models, types, and their metadata within the XObjPrototype ecosystem. It serves as the central hub for:

1. **Function Registry** - Register Python functions with metadata, permissions, and namespace organization
2. **Model Registry** - Manage both static (code-defined) and dynamic (Inspector-generated) Pydantic models
3. **Type Registry** - Register custom types, validators, and transformers
4. **Resource Enrichment** - Connect models to live data sources for validation and suggestions
5. **Import Control** - Enforce discipline through models folder __init__.py

### Key Features

- **Namespace Organization** - Everything accessible via dot notation (e.g., `ns.models.user.Profile`)
- **Permission System** - Role-based access control for functions and models
- **Live Data Integration** - Enrich models with real-time statistics and categorical values
- **Model Versioning** - Track schema evolution with migration support
- **Auto-Discovery** - Automatically find and register models from resources
- **UI Schema Generation** - Automatic UI widget mapping for models
- **Audit Trail** - Comprehensive logging of all registry operations

### Architecture Benefits

1. **Separation of Concerns**
   - XRegistry manages registrations
   - XInspector handles schema discovery
   - XResource manages connections
   - Clear boundaries and responsibilities

2. **Ultrathin Design**
   - Minimal abstractions
   - Direct access to registered items
   - Performance-optimized lookups

3. **Extensibility**
   - Plugin-style registration
   - Custom type validators
   - Transformer pipelines
   - Background enrichment

4. **Developer Experience**
   - Decorators for easy registration
   - Type-safe operations
   - Rich metadata
   - API endpoints for all operations

### Integration Points

```python
# Complete integration example
from src.core.registry import registry
from src.server.core.xresource import ResourceFactory

# 1. Setup registry on startup
async def startup():
    # Auto-discover models
    from src.models import auto_discover_models
    auto_discover_models()
    
    # Connect to data source
    mongo = ResourceFactory.create("mongodb", ...)
    await mongo.connect()
    
    # Enrich models with live data
    enriched = registry.get_enriched()
    await enriched.enrich_from_resource(mongo, "User")

# 2. Use in services
@registry.register_function(namespace="services.user")
async def create_user(data: dict):
    # Validate with enrichment
    enriched = registry.get_enriched()
    user = await enriched.create_with_validation("User", data)
    return user

# 3. Access via API
# GET /api/registry/models
# GET /api/enrichment/models/User/suggestions/role
# POST /api/registry/functions/create_user/execute
```

### Next Steps

1. **Implementation Priority**
   - Implement namespace system first (foundation)
   - Build XObjPrototype base class
   - Create registry components in order
   - Add Inspector integration
   - Enable resource enrichment

2. **Testing Strategy**
   - Unit tests for each registry
   - Integration tests for enrichment
   - Performance benchmarks
   - API endpoint tests

3. **Deployment Considerations**
   - Configure via environment variables
   - Enable audit logging
   - Set up background refresh
   - Monitor registry performance

XRegistry provides the foundation for a disciplined, metadata-rich, and extensible architecture that scales with your application's growth while maintaining clarity and performance.