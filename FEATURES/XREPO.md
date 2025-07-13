# Repo Abstraction - Technical Specification

## Overview

The Repo abstraction provides a unified interface for data access operations across different storage backends. It builds upon the XResource layer to provide high-level data operations including CRUD, schema inspection, access control, auditing, statistics, and dynamic views. The Repo pattern abstracts the complexities of different data sources while providing consistent APIs for data manipulation and analysis.

**Key Innovation**: Ultrathink schema-driven factory with auto-detection that unifies all repo patterns into just two implementations: `ConnectedRepo` and `MaterializedRepo`.

## Design Principles

- **Schema-Driven**: Schema inspection determines repo behavior, not complex detection logic
- **Ultrathink**: Only 2 repo types needed - Connected (live) vs Materialized (in-memory)
- **Auto-Detection**: Simple `XRepoFactory.create(**inputs)` API with intelligent pattern detection
- **Multi-Source**: Native support for joining MongoDB collections, Redis, CSV, and in-memory data
- **Type-Safe**: Leverages Pydantic models and Python type hints throughout
- **Async-First**: All operations are asynchronous for optimal performance
- **Resource-Agnostic**: Works with any XResource implementation (MongoDB, PostgreSQL, CSV, etc.)
- **Observable**: Built-in hooks for auditing, metrics, and change tracking
- **Secure**: Fine-grained access control at row and column levels

## Architecture

### Repo Hierarchy

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
from datetime import datetime
from pydantic import BaseModel, Field
from src.server.core.xobj_prototype import XObjPrototype
from src.server.core.xresource import XResource
from enum import Enum

T = TypeVar('T', bound=XObjPrototype)

class Permission(str, Enum):
    """Repo permissions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    ADMIN = "admin"

class ACLRule(BaseModel):
    """Access Control List rule"""
    role: str
    permissions: List[Permission]
    conditions: Optional[Dict[str, Any]] = None  # Field-level or row-level conditions

class AuditEntry(BaseModel):
    """Audit log entry"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    operation: str
    entity_id: Optional[Any] = None
    changes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}

class RepoStats(BaseModel):
    """Repo statistics"""
    total_count: int = 0
    created_today: int = 0
    updated_today: int = 0
    deleted_today: int = 0
    field_cardinalities: Dict[str, int] = {}
    storage_size_bytes: Optional[int] = None
    index_stats: Dict[str, Any] = {}
    query_performance: Dict[str, float] = {}  # operation -> avg ms

class RepoMetadata(BaseModel):
    """Metadata for repo configuration"""
    name: str
    description: Optional[str] = None
    resource_name: str  # Links to XResource
    model_name: str     # Links to Model in registry
    indexes: List[List[str]] = []
    constraints: List[Dict[str, Any]] = []
    enable_audit: bool = True
    enable_statistics: bool = True
    cache_enabled: bool = False
    cache_ttl: int = 300  # seconds
    acl_rules: List[ACLRule] = []
    soft_delete: bool = True
    auto_timestamps: bool = True

class XRepo(XObjPrototype, Generic[T], ABC):
    """Abstract base repo for all data operations"""

    metadata: RepoMetadata
    _resource: XResource
    _model_class: Type[T]
    _audit_logger: Optional['AuditLogger'] = None
    _stats_collector: Optional['StatsCollector'] = None
    _cache: Optional['CacheManager'] = None
    _acl_manager: Optional['ACLManager'] = None

    def __init__(self, resource: XResource, model_class: Type[T], metadata: RepoMetadata):
        super().__init__(metadata=metadata)
        self._resource = resource
        self._model_class = model_class
        self._initialize_components()

    def get_ns(self) -> str:
        """Repo ns based on model"""
        return f"repos.{self.metadata.name}"

    def _initialize_components(self):
        """Initialize audit, stats, cache, and ACL components"""
        if self.metadata.enable_audit:
            self._audit_logger = AuditLogger(self)
        if self.metadata.enable_statistics:
            self._stats_collector = StatsCollector(self)
        if self.metadata.cache_enabled:
            self._cache = CacheManager(self, ttl=self.metadata.cache_ttl)
        if self.metadata.acl_rules:
            self._acl_manager = ACLManager(self, self.metadata.acl_rules)

    async def _check_permission(self, permission: Permission, user_id: Optional[str], data: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has permission for operation"""
        if not self._acl_manager:
            return True  # No ACL configured, allow all
        return await self._acl_manager.check_permission(permission, user_id, data)

    async def _audit_operation(self, operation: str, user_id: Optional[str], entity_id: Optional[Any] = None, changes: Optional[Dict[str, Any]] = None):
        """Log audit entry for operation"""
        if self._audit_logger:
            await self._audit_logger.log(AuditEntry(
                user_id=user_id,
                operation=operation,
                entity_id=entity_id,
                changes=changes
            ))

    async def _update_stats(self, operation: str, duration_ms: float):
        """Update statistics for operation"""
        if self._stats_collector:
            await self._stats_collector.record_operation(operation, duration_ms)

    @abstractmethod
    async def create(self, data: Union[T, Dict[str, Any]], user_id: Optional[str] = None) -> T:
        """Create a new record with audit and permission check"""
        pass

    @abstractmethod
    async def read(self, id: Any, user_id: Optional[str] = None) -> Optional[T]:
        """Read a single record by ID with permission check"""
        pass

    @abstractmethod
    async def update(self, id: Any, data: Dict[str, Any], user_id: Optional[str] = None) -> Optional[T]:
        """Update a record with audit and permission check"""
        pass

    @abstractmethod
    async def delete(self, id: Any, soft: bool = True, user_id: Optional[str] = None) -> bool:
        """Delete a record (soft or hard) with audit and permission check"""
        pass

    @abstractmethod
    async def list(self, filters: Optional[Dict[str, Any]] = None, skip: int = 0, limit: int = 100, sort: Optional[List[Tuple[str, int]]] = None, user_id: Optional[str] = None) -> List[T]:
        """List records with filters, pagination, sorting and permission check"""
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> int:
        """Count records matching filters with permission check"""
        pass

    @abstractmethod
    async def exists(self, id: Any, user_id: Optional[str] = None) -> bool:
        """Check if record exists"""
        pass

    @abstractmethod
    async def bulk_create(self, data: List[Union[T, Dict[str, Any]]], user_id: Optional[str] = None) -> List[T]:
        """Bulk create records"""
        pass

    @abstractmethod
    async def bulk_update(self, updates: List[Tuple[Any, Dict[str, Any]]], user_id: Optional[str] = None) -> int:
        """Bulk update records, returns count of updated records"""
        pass

    @abstractmethod
    async def bulk_delete(self, ids: List[Any], soft: bool = True, user_id: Optional[str] = None) -> int:
        """Bulk delete records, returns count of deleted records"""
        pass

    @abstractmethod
    async def inspect_schema(self) -> Dict[str, Any]:
        """Inspect and return schema information"""
        pass

    @abstractmethod
    async def inspect_collection(self) -> Dict[str, Any]:
        """Inspect collection/table metadata including indexes, constraints, size"""
        pass

    @abstractmethod
    async def get_statistics(self) -> RepoStats:
        """Get repo statistics"""
        pass

    @abstractmethod
    async def get_audit_log(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[AuditEntry]:
        """Get audit log entries"""
        pass

    @abstractmethod
    async def aggregate(self, pipeline: List[Dict[str, Any]], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run aggregation pipeline (MongoDB-style)"""
        pass

    @abstractmethod
    async def find_one(self, filters: Dict[str, Any], user_id: Optional[str] = None) -> Optional[T]:
        """Find single record matching filters"""
        pass

    @abstractmethod
    async def distinct(self, field: str, filters: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> List[Any]:
        """Get distinct values for a field"""
        pass
```

## Design Decision Q&A

### Q1: Repo Base Class Design

**Decision**: Repo inherits from XObjPrototype (Option A selected)

**Rationale**:

- Maintains consistency across all abstractions in the system
- Provides built-in validation and ns support
- Enables repo instances to be managed through the ns system
- Repo metadata becomes validated through Pydantic

**Implementation Note**: While repos are not data objects in the traditional sense, they are configurable components that benefit from validation and ns management.

## Ultrathink Repo Factory Design

The repo factory uses auto-detection and schema-driven patterns to unify all repo creation into a single, intuitive API. The key insight: **schema determines repo behavior**.

### Core Repo Types

Only two repo implementations are needed:

```python
class ConnectedRepo(XRepo):
    """Repo with live connection to external resource (MongoDB, PostgreSQL, etc.)"""

    def __init__(self, schema: Dict[str, Any], resource: XResource, model_class: Type = None):
        self.schema = schema
        self.resource = resource
        self.model_class = model_class or self._generate_model_from_schema(schema)

    async def list(self, filters: Dict = None, **kwargs) -> List[Any]:
        """Query external resource"""
        return await self.resource.query(filters, **kwargs)

class MaterializedRepo(XRepo):
    """Repo with in-memory data (views, virtual, computed, joined data)"""

    def __init__(self, schema: Dict[str, Any], data: List[Dict[str, Any]]):
        self.schema = schema
        self.data = data
        self.model_class = self._generate_model_from_schema(schema)

    async def list(self, filters: Dict = None, **kwargs) -> List[Any]:
        """Filter in-memory data"""
        return self._filter_data(self.data, filters, **kwargs)
```

### Auto-Detection Factory

```python
class XRepoFactory:
    """Ultrathink auto-detection factory driven by schema analysis"""

    @classmethod
    async def create(
        cls,
        # Input can be any of these patterns
        model_class: Optional[Type[XObjPrototype]] = None,
        resource: Optional[XResource] = None,
        data: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[Union[Tuple[XResource, str], XResource, List[Dict]]]] = None,
        join_config: Optional[Dict[str, Any]] = None,
        # Schema override
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'XRepo':
        """Auto-detect and create repo based on inputs"""

        # 1. Extract schema from inputs
        computed_schema = schema or await cls._extract_schema(
            model_class=model_class,
            resource=resource,
            data=data,
            sources=sources,
            join_config=join_config
        )

        # 2. Determine repo type from schema
        repo_type = cls._determine_repo_type(computed_schema)

        # 3. Create repo
        return cls._create_repo(
            repo_type=repo_type,
            schema=computed_schema,
            model_class=model_class,
            resource=resource,
            data=data,
            sources=sources,
            **kwargs
        )

    @classmethod
    async def _extract_schema(cls, **inputs) -> Dict[str, Any]:
        """Extract unified schema from various input types"""

        if inputs.get('model_class'):
            # Schema from model
            return cls._schema_from_model(inputs['model_class'])

        elif inputs.get('resource') and not inputs.get('sources'):
            # Schema from resource inspection
            resource = inputs['resource']
            await cls._ensure_connected(resource)
            return await resource.discover_schema()

        elif inputs.get('data'):
            # Schema from data analysis
            return cls._schema_from_data(inputs['data'])

        elif inputs.get('sources'):
            # Schema from multiple sources (join/union)
            return await cls._schema_from_sources(
                inputs['sources'],
                inputs.get('join_config', {})
            )

        raise ValueError("Cannot determine schema from provided inputs")

    @classmethod
    def _determine_repo_type(cls, schema: Dict[str, Any]) -> str:
        """Determine repo type based on schema characteristics"""

        # Simple heuristic based on schema metadata
        if 'resource_connection' in schema:
            return 'connected'  # Live connection to external resource
        else:
            return 'materialized'  # In-memory data with schema
```

### Strategy Pattern Foundation

While keeping the API simple, the factory internally uses strategies for extensibility:

```python
from abc import ABC, abstractmethod

class RepoCreationStrategy(ABC):
    """Strategy interface for repo creation patterns"""

    @abstractmethod
    def can_handle(self, **inputs) -> bool:
        """Check if this strategy can handle the given inputs"""
        pass

    @abstractmethod
    async def create_repo(self, **inputs) -> 'XRepo':
        """Create repo from inputs"""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Strategy priority (higher = checked first)"""
        pass

class XRepoFactory:
    """Auto-detection factory with pluggable strategies"""

    _strategies: List[RepoCreationStrategy] = []

    @classmethod
    def register_strategy(cls, strategy: RepoCreationStrategy) -> None:
        """Register a custom creation strategy"""
        cls._strategies.append(strategy)
        cls._strategies.sort(key=lambda s: s.priority, reverse=True)

    @classmethod
    async def create(cls, **inputs) -> 'XRepo':
        """Auto-detect creation pattern using registered strategies"""

        # Try strategies in priority order
        for strategy in cls._strategies:
            if strategy.can_handle(**inputs):
                return await strategy.create_repo(**inputs)

        raise ValueError(f"No strategy found for inputs: {list(inputs.keys())}")
```

## ModelGenerator - Dynamic XObjPrototype Creation

The ModelGenerator creates XObjPrototype model classes dynamically from discovered resource schemas, enabling type-safe repos without manual model definition. For detailed schema discovery implementation, see [XINSPECTOR.md](./XINSPECTOR.md).

### ModelGenerator Implementation

```python
from typing import Type, Dict, Any, Optional
from pydantic import Field, create_model
from src.server.core.xobj_prototype import XObjPrototype
from datetime import datetime
from decimal import Decimal

class ModelGenerator:
    """Generates XObjPrototype models from resource schemas"""

    @classmethod
    def generate_model_from_schema(
        cls,
        schema: Dict[str, Any],
        model_name: str,
        ns: str = None
    ) -> Type[XObjPrototype]:
        """Generate XObjPrototype model from discovered schema"""

        # Extract field definitions from schema
        field_definitions = cls._build_field_definitions(schema)

        # Create ns and collection info
        ns = ns or cls._infer_ns(schema, model_name)
        collection = cls._infer_collection(schema)

        # Create Meta class
        meta_attrs = {
            'ns': ns,
            'collection': collection,
            'indexes': cls._extract_indexes(schema),
            'auto_generated': True,
            'source_schema': schema
        }

        # Create the model dynamically
        model_attrs = {
            **field_definitions,
            'Meta': type('Meta', (), meta_attrs),
            'get_ns': lambda self: ns,
            '__module__': 'dynamically_generated'
        }

        # Use Pydantic's create_model with XObjPrototype as base
        DynamicModel = create_model(
            model_name,
            __base__=XObjPrototype,
            **field_definitions
        )

        # Add Meta class and methods
        DynamicModel.Meta = type('Meta', (), meta_attrs)
        DynamicModel.get_ns = lambda self: ns

        return DynamicModel

    @classmethod
    def _build_field_definitions(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert schema fields to Pydantic field definitions"""
        field_definitions = {}
        fields = schema.get('fields', {})

        for field_name, field_info in fields.items():
            python_type = cls._convert_type_string_to_python_type(field_info['type'])
            is_required = field_info.get('required', True)

            # Create field with appropriate constraints
            field_kwargs = cls._build_field_constraints(field_info)

            if is_required:
                field_definitions[field_name] = (python_type, Field(**field_kwargs))
            else:
                field_definitions[field_name] = (Optional[python_type], Field(default=None, **field_kwargs))

        return field_definitions

    @classmethod
    def _convert_type_string_to_python_type(cls, type_string: str) -> Type:
        """Convert string type to actual Python type"""
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'datetime': datetime,
            'Decimal': Decimal,
            'List[Any]': list,
            'Dict[str, Any]': dict,
            'EmailStr': str,  # Could import from pydantic
            'HttpUrl': str,   # Could import from pydantic
            'Any': object
        }

        return type_mapping.get(type_string, str)

    @classmethod
    def _build_field_constraints(cls, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build Pydantic Field constraints from schema analysis"""
        constraints = {}

        # Add description from cardinality and samples
        cardinality = field_info.get('cardinality', 'unknown')
        sample_values = field_info.get('sample_values', [])

        description_parts = [f"Cardinality: {cardinality}"]
        if sample_values:
            description_parts.append(f"Sample values: {', '.join(map(str, sample_values[:3]))}")

        constraints['description'] = ' | '.join(description_parts)

        # Add validation constraints based on type and cardinality
        if cardinality == 'low' and len(sample_values) <= 20:
            # Could add enum constraint for low cardinality fields
            pass

        # Add type-specific constraints
        field_type = field_info.get('type', 'str')
        if field_type in ['EmailStr']:
            constraints['regex'] = r'^[^@]+@[^@]+\.[^@]+$'
        elif field_type == 'HttpUrl':
            constraints['regex'] = r'^https?://.+'

        return constraints

    @classmethod
    def _infer_ns(cls, schema: Dict[str, Any], model_name: str) -> str:
        """Infer ns from schema and model name"""
        # Check if schema has explicit ns info
        if 'ns' in schema:
            return schema['ns']

        # Infer from collection/table name
        collection = cls._infer_collection(schema)
        if collection:
            return f"models.{collection}"

        # Default ns
        return f"models.{model_name.lower()}"

    @classmethod
    def _infer_collection(cls, schema: Dict[str, Any]) -> str:
        """Infer collection/table name from schema"""
        # MongoDB collection
        if 'collection' in schema:
            return schema['collection']

        # CSV file
        if 'file' in schema:
            from pathlib import Path
            return Path(schema['file']).stem

        # PostgreSQL table
        if 'table' in schema:
            return schema['table']

        return None

    @classmethod
    def _extract_indexes(cls, schema: Dict[str, Any]) -> List[List[str]]:
        """Extract index definitions from schema"""
        indexes = []

        # MongoDB indexes
        if 'indexes' in schema:
            for index_info in schema['indexes']:
                if isinstance(index_info, dict) and 'fields' in index_info:
                    indexes.append(index_info['fields'])
                elif isinstance(index_info, list):
                    indexes.append(index_info)

        # Add indexes for high-cardinality fields (likely unique identifiers)
        fields = schema.get('fields', {})
        for field_name, field_info in fields.items():
            if field_info.get('cardinality') == 'high' and field_name.endswith('_id'):
                indexes.append([field_name])

        return indexes

class ModelRegistry:
    """Registry for dynamically generated models"""

    _models: Dict[str, Type[XObjPrototype]] = {}
    _schemas: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_generated_model(
        cls,
        model_name: str,
        model_class: Type[XObjPrototype],
        schema: Dict[str, Any]
    ) -> None:
        """Register a dynamically generated model"""
        cls._models[model_name] = model_class
        cls._schemas[model_name] = schema

    @classmethod
    def get_model(cls, model_name: str) -> Optional[Type[XObjPrototype]]:
        """Get registered model by name"""
        return cls._models.get(model_name)

    @classmethod
    def get_schema(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get original schema for a model"""
        return cls._schemas.get(model_name)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names"""
        return list(cls._models.keys())

    @classmethod
    def is_auto_generated(cls, model_name: str) -> bool:
        """Check if model was auto-generated"""
        model = cls.get_model(model_name)
        return hasattr(model, 'Meta') and getattr(model.Meta, 'auto_generated', False)
```

### Usage Examples

```python
# Example 1: Generate model from MongoDB collection schema
mongo_resource = ResourceFactory.create("mongodb", connection_string="mongodb://localhost/app")
await mongo_resource.connect()

# Discover schema - delegates to Inspector (see XINSPECTOR.md)
user_schema = await mongo_resource.discover_schema("users")

# Generate model
User = ModelGenerator.generate_model_from_schema(
    schema=user_schema,
    model_name="User",
    ns="models.users"
)

# Register the model
ModelRegistry.register_generated_model("User", User, user_schema)

# Use the generated model
user_instance = User(
    name="John Doe",
    email="john@example.com",
    age=30
)

# Example 2: Generate model from CSV schema
csv_resource = ResourceFactory.create("csv", file_path="/data/products.csv")
await csv_resource.connect()

product_schema = await csv_resource.discover_schema()
Product = ModelGenerator.generate_model_from_schema(
    schema=product_schema,
    model_name="Product",
    ns="models.products"
)

# Example 3: Inspect generated model
print(f"Model fields: {list(User.model_fields.keys())}")
print(f"Namespace: {User().get_ns()}")
print(f"Auto-generated: {ModelRegistry.is_auto_generated('User')}")

# Example 4: Schema evolution - regenerate model when schema changes
updated_schema = await mongo_resource.discover_schema("users")
if updated_schema != ModelRegistry.get_schema("User"):
    # Schema changed, regenerate model
    UpdatedUser = ModelGenerator.generate_model_from_schema(
        schema=updated_schema,
        model_name="User",
        ns="models.users"
    )
    ModelRegistry.register_generated_model("User", UpdatedUser, updated_schema)
```

## Enhanced XRepoFactory - Resource-Based Creation

Extended factory methods that enable repo creation directly from resources, with optional auto-model generation.

### Extended Factory Methods

```python
class XRepoFactory:
    """Extended factory for resource-based repo creation"""

    # ... existing methods ...

    @classmethod
    async def create_from_resource(
        cls,
        resource: XResource,
        collection: str = None,
        auto_generate_model: bool = False,
        model_name: str = None,
        ns: str = None,
        repo_type: RepoType = RepoType.SINGLE_RESOURCE
    ) -> 'XRepo':
        """Create repo directly from resource with optional model generation"""

        # Ensure resource is connected
        if not await resource.validate_connection():
            await resource.connect()

        # Discover schema from resource
        schema = await resource.discover_schema(collection)

        if auto_generate_model:
            # Generate model dynamically
            model_name = model_name or cls._infer_model_name(schema, collection)
            ns = ns or f"models.{collection or 'generated'}"

            # Check if model already exists
            existing_model = ModelRegistry.get_model(model_name)
            if existing_model and ModelRegistry.get_schema(model_name) == schema:
                # Use existing model if schema matches
                model_class = existing_model
            else:
                # Generate new model
                model_class = ModelGenerator.generate_model_from_schema(
                    schema=schema,
                    model_name=model_name,
                    ns=ns
                )
                ModelRegistry.register_generated_model(model_name, model_class, schema)

            # Create typed repo
            strategy_class = cls._strategies[repo_type]
            metadata = RepoMetadata(
                name=f"{model_name.lower()}_repo",
                resource_name=resource.metadata.name,
                model_name=model_name,
                ns=ns,
                auto_generated=True,
                source_schema=schema
            )

            return strategy_class(
                model_class=model_class,
                resource=resource,
                metadata=metadata
            )
        else:
            # Create schema-aware repo without specific model
            return SchemaAwareRepo(
                resource=resource,
                schema=schema,
                collection=collection,
                metadata=RepoMetadata(
                    name=f"{collection or 'resource'}_repo",
                    resource_name=resource.metadata.name,
                    model_name="Dict",
                    ns=ns or "repos.schema_aware",
                    schema_based=True,
                    source_schema=schema
                )
            )

    @classmethod
    async def create_view_from_resources(
        cls,
        name: str,
        resources: List[Tuple[XResource, str]],  # (resource, collection/table) pairs
        join_config: Dict[str, Any],
        auto_generate_model: bool = True,
        result_model_name: str = None
    ) -> 'XRepo':
        """Create view repo from multiple resources"""

        # Discover schemas from all resources
        schemas = {}
        for resource, collection_table in resources:
            if not await resource.validate_connection():
                await resource.connect()
            schemas[f"{resource.metadata.name}.{collection_table}"] = await resource.discover_schema(collection_table)

        if auto_generate_model:
            # Generate joined schema
            joined_schema = cls._merge_schemas(schemas, join_config)

            # Generate model for joined result
            result_model_name = result_model_name or f"{name}_view"
            result_model = ModelGenerator.generate_model_from_schema(
                schema=joined_schema,
                model_name=result_model_name,
                ns=f"views.{name}"
            )

            # Create source repos
            source_repos = []
            for resource, collection_table in resources:
                source_repo = await cls.create_from_resource(
                    resource=resource,
                    collection=collection_table,
                    auto_generate_model=True
                )
                source_repos.append(source_repo)

            # Create view repo
            return ViewRepo(
                source_repos=source_repos,
                join_config=join_config,
                result_model=result_model,
                metadata=RepoMetadata(
                    name=name,
                    resource_name="virtual_view",
                    model_name=result_model_name,
                    ns=f"views.{name}",
                    source_schemas=schemas,
                    join_config=join_config
                )
            )
        else:
            # Create untyped view repo
            raise NotImplementedError("Untyped view repos not yet implemented")

    @classmethod
    async def create_aggregate_repo(
        cls,
        name: str,
        source_repos: List['XRepo'],
        aggregation_config: Dict[str, Any],
        auto_generate_model: bool = True
    ) -> 'XRepo':
        """Create repo that aggregates data from multiple sources"""

        if auto_generate_model:
            # Merge schemas from source repos
            source_schemas = {}
            for repo in source_repos:
                if hasattr(repo.metadata, 'source_schema'):
                    source_schemas[repo.metadata.name] = repo.metadata.source_schema

            # Generate aggregated schema based on aggregation config
            aggregated_schema = cls._create_aggregated_schema(source_schemas, aggregation_config)

            # Generate model
            result_model = ModelGenerator.generate_model_from_schema(
                schema=aggregated_schema,
                model_name=f"{name}_aggregate",
                ns=f"aggregates.{name}"
            )

            return AggregateRepo(
                source_repos=source_repos,
                aggregation_config=aggregation_config,
                result_model=result_model,
                metadata=RepoMetadata(
                    name=name,
                    resource_name="virtual_aggregate",
                    model_name=f"{name}_aggregate",
                    ns=f"aggregates.{name}",
                    aggregation_config=aggregation_config
                )
            )
        else:
            raise NotImplementedError("Untyped aggregate repos not yet implemented")

    @classmethod
    def _infer_model_name(cls, schema: Dict[str, Any], collection: str) -> str:
        """Infer model name from schema and collection/table name"""
        if collection:
            # Convert to PascalCase
            return ''.join(word.capitalize() for word in collection.replace('_', ' ').split())

        # Fallback to schema-based naming
        if 'collection' in schema:
            return ''.join(word.capitalize() for word in schema['collection'].replace('_', ' ').split())
        elif 'file' in schema:
            from pathlib import Path
            filename = Path(schema['file']).stem
            return ''.join(word.capitalize() for word in filename.replace('_', ' ').split())

        return "GeneratedModel"

    @classmethod
    def _merge_schemas(cls, schemas: Dict[str, Dict[str, Any]], join_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple schemas for view creation"""
        merged_fields = {}

        # Merge all fields from all schemas
        for source_name, schema in schemas.items():
            source_prefix = source_name.split('.')[-1]  # Use table/collection name as prefix

            for field_name, field_info in schema.get('fields', {}).items():
                # Prefix field names to avoid conflicts (except join keys)
                join_key = join_config.get('on', 'id')
                if field_name == join_key:
                    merged_field_name = field_name
                else:
                    merged_field_name = f"{source_prefix}_{field_name}"

                merged_fields[merged_field_name] = field_info.copy()
                merged_fields[merged_field_name]['source'] = source_name

        return {
            'fields': merged_fields,
            'view_type': 'joined',
            'source_schemas': schemas,
            'join_config': join_config
        }

    @classmethod
    def _create_aggregated_schema(cls, schemas: Dict[str, Dict[str, Any]], aggregation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create schema for aggregated data"""
        # This would implement aggregation schema logic
        # For now, simple implementation
        aggregated_fields = {}

        # Add grouping fields
        group_by = aggregation_config.get('group_by', [])
        for field in group_by:
            # Find field in source schemas
            for schema in schemas.values():
                if field in schema.get('fields', {}):
                    aggregated_fields[field] = schema['fields'][field].copy()
                    break

        # Add aggregated fields
        aggregations = aggregation_config.get('aggregations', {})
        for agg_name, agg_config in aggregations.items():
            aggregated_fields[agg_name] = {
                'type': 'float',  # Most aggregations return numeric values
                'required': True,
                'aggregation': agg_config
            }

        return {
            'fields': aggregated_fields,
            'view_type': 'aggregated',
            'source_schemas': schemas,
            'aggregation_config': aggregation_config
        }
```

### Usage Examples - Resource-Based Creation

```python
# Example 1: Simple repo from MongoDB collection
mongo_resource = ResourceFactory.create("mongodb", connection_string="mongodb://localhost/app")
user_repo = await XRepoFactory.create_from_resource(
    resource=mongo_resource,
    collection="users"
)

# Schema-aware repo (works with Dict objects)
users = await user_repo.list({})  # Returns List[Dict[str, Any]]

# Example 2: Typed repo with auto-generated model
user_repo_typed = await XRepoFactory.create_from_resource(
    resource=mongo_resource,
    collection="users",
    auto_generate_model=True,
    model_name="User"
)

# Typed repo (returns proper User instances)
users_typed = await user_repo_typed.list({})  # Returns List[User]

# Example 3: CSV-based repo
csv_resource = ResourceFactory.create("csv", file_path="/data/products.csv")
product_repo = await XRepoFactory.create_from_resource(
    resource=csv_resource,
    auto_generate_model=True,
    model_name="Product"
)

# Example 4: View repo from multiple resources
user_order_view = await XRepoFactory.create_view_from_resources(
    name="user_orders",
    resources=[
        (mongo_resource, "users"),
        (mongo_resource, "orders")
    ],
    join_config={
        "on": "user_id",
        "type": "left_join"
    },
    auto_generate_model=True,
    result_model_name="UserOrderView"
)

# Query the view
user_orders = await user_order_view.list({
    "user_filters": {"status": "active"},
    "order_filters": {"created_at__gte": "2024-01-01"}
})

# Example 5: List available collections/tables
collections = await mongo_resource.list_collections_or_tables()
print(f"Available collections: {collections}")

# Create repos for all collections
repos = {}
for collection in collections:
    repos[collection] = await XRepoFactory.create_from_resource(
        resource=mongo_resource,
        collection=collection,
        auto_generate_model=True
    )
```
