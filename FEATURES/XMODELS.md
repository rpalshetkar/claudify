# Dynamic Model Registry

## Overview

The Dynamic Model Registry provides runtime model registration and management with automatic UI generation, permissions, and audit logging. It works in conjunction with the Inspector module for schema discovery and model generation.

## Architectural Decision

**Clear Separation of Concerns**:
- **XInspector**: Analyzes data sources and generates models (creation phase)
- **XModels**: Registers and manages models at runtime (management phase)

This separation ensures that model generation logic remains centralized in Inspector while the registry focuses on runtime operations.

## Core Responsibility

The registry focuses solely on:

- Model registration and storage
- UI widget mapping
- Permission management
- Audit logging
- CRUD/CQRS operations

**Important**: Schema discovery, data profiling, and model generation are handled exclusively by the Inspector module (see [XINSPECTOR.md](./XINSPECTOR.md)). This registry only handles registration and management of already-generated models.

## Architecture

```python
from typing import Dict, Type, List, Any, Set
from pydantic import BaseModel
from src.server.core.xobj_prototype import XObjPrototype
from enum import Enum

class OperationMode(Enum):
    CRUD = "crud"
    CQRS = "cqrs"
    HYBRID = "hybrid"

class DynamicModelRegistry:
    """Central registry for dynamic models"""

    def __init__(self):
        self.models: Dict[str, Type[BaseModel]] = {}
        self.ui_schemas: Dict[str, UISchema] = {}
        self.permissions: Dict[str, ModelPermissions] = {}
        self.audit_logger: AuditLogger = AuditLogger()
        self.storage: Dict[str, List[Dict[str, Any]]] = {}

    def register_model(
        self,
        name: str,
        model_class: Type[BaseModel],
        operation_mode: OperationMode = OperationMode.CRUD
    ) -> None:
        """Register a model with the registry"""
        self.models[name] = model_class
        self.ui_schemas[name] = UISchemaGenerator.generate(model_class)
        self.permissions[name] = ModelPermissions(name)
        self.storage[name] = []
```

## UI Widget System

### Widget Detection

```python
class UIWidgetType(Enum):
    TEXT_INPUT = "text_input"
    TEXT_AREA = "text_area"
    PASSWORD = "password"
    EMAIL = "email"
    URL = "url"
    NUMBER = "number"
    SLIDER = "slider"
    DATE_PICKER = "date_picker"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    TOGGLE = "toggle"
    JSON_EDITOR = "json_editor"

class UISchemaGenerator:
    """Generate UI schemas from Pydantic models"""

    @staticmethod
    def generate(model_class: Type[BaseModel]) -> UISchema:
        """Generate UI schema from model"""
        fields = {}

        for field_name, field_info in model_class.model_fields.items():
            widget_type = UISchemaGenerator._detect_widget(
                field_name,
                field_info
            )
            fields[field_name] = UIFieldSchema(
                widget=widget_type,
                label=field_name.replace("_", " ").title(),
                required=field_info.is_required(),
                metadata=field_info.metadata
            )

        return UISchema(fields=fields)

    @staticmethod
    def _detect_widget(name: str, field_info) -> UIWidgetType:
        """Detect appropriate widget based on field name and type"""
        name_lower = name.lower()

        # Name-based detection
        if "password" in name_lower:
            return UIWidgetType.PASSWORD
        if "email" in name_lower:
            return UIWidgetType.EMAIL
        if "url" in name_lower or "link" in name_lower:
            return UIWidgetType.URL
        if "description" in name_lower or "notes" in name_lower:
            return UIWidgetType.TEXT_AREA

        # Type-based detection
        if field_info.annotation == bool:
            return UIWidgetType.TOGGLE
        if field_info.annotation in (int, float):
            if hasattr(field_info, 'ge') and hasattr(field_info, 'le'):
                return UIWidgetType.SLIDER
            return UIWidgetType.NUMBER

        return UIWidgetType.TEXT_INPUT
```

### Widget Configuration

```python
class UIFieldSchema(BaseModel):
    """UI configuration for a field"""
    widget: UIWidgetType
    label: str
    placeholder: str = ""
    help_text: str = ""
    required: bool = False
    readonly: bool = False
    hidden: bool = False
    validation_rules: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
```

## Permission System

### Model-Level Permissions

```python
class Permission(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"

class ModelPermissions:
    """Manage permissions for a model"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.role_permissions: Dict[str, Set[Permission]] = {
            "admin": {Permission.CREATE, Permission.READ,
                     Permission.UPDATE, Permission.DELETE, Permission.LIST},
            "user": {Permission.READ, Permission.LIST}
        }
        self.field_permissions: Dict[str, FieldPermission] = {}

    def can_perform(
        self,
        user_role: str,
        permission: Permission
    ) -> bool:
        """Check if role can perform action"""
        return permission in self.role_permissions.get(user_role, set())
```

### Field-Level Permissions

```python
class FieldPermission:
    """Field-level access control"""

    def __init__(self, field_name: str):
        self.field_name = field_name
        self.read_roles: Set[str] = {"admin", "user"}
        self.write_roles: Set[str] = {"admin"}
        self.mask_function: Optional[Callable] = None

    def can_read(self, user_role: str) -> bool:
        return user_role in self.read_roles

    def can_write(self, user_role: str) -> bool:
        return user_role in self.write_roles

    def apply_mask(self, value: Any, user_role: str) -> Any:
        """Apply masking based on role"""
        if self.mask_function and user_role not in {"admin"}:
            return self.mask_function(value)
        return value
```

## Audit System

### Audit Logger

```python
class AuditEntry(XObjPrototype):
    """Single audit log entry"""
    model_name: str
    record_id: str
    operation: str
    user_id: str
    timestamp: datetime
    changes: Dict[str, Dict[str, Any]]  # field -> {old, new}
    metadata: Dict[str, Any] = {}
    ns: str = "audit.entries"

class AuditLogger:
    """Compressed audit logging system"""

    def __init__(self):
        self.entries: List[AuditEntry] = []
        self.compression_enabled = True

    def log_operation(
        self,
        model_name: str,
        record_id: str,
        operation: str,
        user_id: str,
        changes: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log an operation with compression"""
        entry = AuditEntry(
            model_name=model_name,
            record_id=record_id,
            operation=operation,
            user_id=user_id,
            timestamp=datetime.now(),
            changes=changes,
            metadata=metadata or {}
        )

        if self.compression_enabled:
            self._store_compressed(entry)
        else:
            self.entries.append(entry)

    def _store_compressed(self, entry: AuditEntry) -> None:
        """Store entry with zlib compression"""
        import zlib
        import base64

        json_data = entry.model_dump_json()
        compressed = zlib.compress(json_data.encode())
        encoded = base64.b64encode(compressed).decode()

        # Store compressed entry
        self.entries.append({
            "compressed": True,
            "data": encoded,
            "timestamp": entry.timestamp
        })
```

## CRUD Operations

```python
class DynamicModelRegistry:
    """Extended with CRUD operations"""

    def create(
        self,
        model_name: str,
        data: Dict[str, Any],
        user_id: str = None,
        user_role: str = "user"
    ) -> Dict[str, Any]:
        """Create a new record"""
        # Check permissions
        if not self.permissions[model_name].can_perform(user_role, Permission.CREATE):
            raise PermissionError(f"Role {user_role} cannot create {model_name}")

        # Validate with model
        model_class = self.models[model_name]
        instance = model_class(**data)

        # Generate ID if needed
        record = instance.model_dump()
        record["id"] = str(uuid.uuid4())

        # Store
        self.storage[model_name].append(record)

        # Audit
        self.audit_logger.log_operation(
            model_name=model_name,
            record_id=record["id"],
            operation="create",
            user_id=user_id,
            changes={field: {"old": None, "new": value}
                    for field, value in record.items()}
        )

        return record

    def read(
        self,
        model_name: str,
        record_id: str,
        user_id: str = None,
        user_role: str = "user"
    ) -> Dict[str, Any]:
        """Read a record with field-level permissions"""
        # Check model permission
        if not self.permissions[model_name].can_perform(user_role, Permission.READ):
            raise PermissionError(f"Role {user_role} cannot read {model_name}")

        # Find record
        record = self._find_record(model_name, record_id)
        if not record:
            raise ValueError(f"Record {record_id} not found")

        # Apply field permissions
        filtered_record = {}
        for field, value in record.items():
            field_perm = self.permissions[model_name].field_permissions.get(field)
            if field_perm and field_perm.can_read(user_role):
                filtered_record[field] = field_perm.apply_mask(value, user_role)
            elif not field_perm:  # No specific permission, allow
                filtered_record[field] = value

        return filtered_record

    def update(
        self,
        model_name: str,
        record_id: str,
        updates: Dict[str, Any],
        user_id: str = None,
        user_role: str = "user"
    ) -> Dict[str, Any]:
        """Update a record"""
        # Check permissions
        if not self.permissions[model_name].can_perform(user_role, Permission.UPDATE):
            raise PermissionError(f"Role {user_role} cannot update {model_name}")

        # Find record
        record = self._find_record(model_name, record_id)
        if not record:
            raise ValueError(f"Record {record_id} not found")

        # Track changes
        changes = {}
        for field, new_value in updates.items():
            # Check field permission
            field_perm = self.permissions[model_name].field_permissions.get(field)
            if field_perm and not field_perm.can_write(user_role):
                raise PermissionError(f"Role {user_role} cannot update field {field}")

            if field in record:
                old_value = record[field]
                record[field] = new_value
                changes[field] = {"old": old_value, "new": new_value}

        # Validate updated record
        model_class = self.models[model_name]
        instance = model_class(**record)

        # Audit
        self.audit_logger.log_operation(
            model_name=model_name,
            record_id=record_id,
            operation="update",
            user_id=user_id,
            changes=changes
        )

        return record
```

## CQRS Support

```python
class Command(XObjPrototype):
    """Base command for CQRS"""
    command_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    ns: str = "commands"

class Query(XObjPrototype):
    """Base query for CQRS"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ns: str = "queries"

class Event(XObjPrototype):
    """Base event for CQRS"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    aggregate_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    ns: str = "events"

class DynamicModelRegistry:
    """Extended with CQRS support"""

    def execute_command(self, command: Command) -> None:
        """Execute a command in CQRS mode"""
        # Command handler logic
        # Generate events
        # Update projections
        pass

    def execute_query(self, query: Query) -> Any:
        """Execute a query in CQRS mode"""
        # Query from projections
        # Apply permissions
        # Return results
        pass
```

## Integration with Inspector

```python
class DynamicModelRegistry:
    """Integration with Inspector for model discovery"""

    async def register_from_resource(
        self,
        resource: XResource,
        model_name: str = None
    ) -> str:
        """Register a model discovered from a resource"""
        from src.server.core.inspector import XInspector

        # Use Inspector to discover schema and generate model
        inspector = XInspector(resource)
        model_class = await inspector.generate_model(collection_name=model_name)

        # Register the generated model
        name = model_name or model_class.__name__
        self.register_model(name, model_class)

        return name
```

## Usage Examples

### Basic Registration

```python
from pydantic import BaseModel, Field

# Define model
class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0)
    description: str = ""
    is_active: bool = True

# Create registry
registry = DynamicModelRegistry()

# Register model
registry.register_model("Product", Product)

# Create product
product = registry.create("Product", {
    "name": "Laptop",
    "price": 999.99,
    "description": "High-performance laptop"
}, user_id="admin", user_role="admin")

# Read with permissions
product_view = registry.read(
    "Product",
    product["id"],
    user_role="user"
)
```

### With Field Permissions

```python
# Configure field permissions
registry.permissions["Product"].field_permissions["price"] = FieldPermission("price")
registry.permissions["Product"].field_permissions["price"].read_roles = {"admin", "manager"}
registry.permissions["Product"].field_permissions["price"].mask_function = lambda x: "***"

# User sees masked price
product_view = registry.read("Product", product["id"], user_role="user")
print(product_view["price"])  # "***"
```

### UI Schema Generation

```python
# Get UI schema
ui_schema = registry.ui_schemas["Product"]

# Access widget configuration
for field_name, field_schema in ui_schema.fields.items():
    print(f"{field_name}: {field_schema.widget.value}")
    # name: text_input
    # price: number
    # description: text_area
    # is_active: toggle
```

## Configuration

```toml
# settings.toml
[models]
audit_compression = true
audit_retention_days = 90
default_page_size = 50

[models.permissions]
default_user_permissions = ["read", "list"]
default_admin_permissions = ["create", "read", "update", "delete", "list"]

[models.ui]
date_format = "%Y-%m-%d"
number_format = ",.2f"
```

## Testing

```python
import pytest
from src.server.models import DynamicModelRegistry

@pytest.fixture
def registry():
    return DynamicModelRegistry()

def test_model_registration(registry):
    class TestModel(BaseModel):
        name: str
        value: int

    registry.register_model("TestModel", TestModel)
    assert "TestModel" in registry.models

def test_crud_operations(registry):
    class TestModel(BaseModel):
        name: str

    registry.register_model("TestModel", TestModel)

    # Create
    record = registry.create("TestModel", {"name": "test"})
    assert record["name"] == "test"

    # Read
    fetched = registry.read("TestModel", record["id"])
    assert fetched["name"] == "test"

def test_ui_widget_detection():
    class UserModel(BaseModel):
        email: str
        password: str
        description: str

    schema = UISchemaGenerator.generate(UserModel)
    assert schema.fields["email"].widget == UIWidgetType.EMAIL
    assert schema.fields["password"].widget == UIWidgetType.PASSWORD
    assert schema.fields["description"].widget == UIWidgetType.TEXT_AREA
```
