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

## Architectural Decisions

Based on the architecture review:

1. **Factory with Smart Defaults**: Explicit configuration with optional `materialized` parameter
2. **Auto-Detection Logic**: REST/WebSocket → MaterializedRepo, File/DB → ConnectedRepo
3. **Model Generation Removed**: All model generation delegated to XInspector via two-step process
4. **Simplified Repo Types**: Only ConnectedRepo and MaterializedRepo needed
5. **Error Handling**: Domain-specific exceptions (XRepoError hierarchy)

## Ultrathink Repo Factory Design

The repo factory uses auto-detection and schema-driven patterns to unify all repo creation into a single, intuitive API. The key insight: **resource type determines default repo behavior**.

### Core Repo Types

Only two repo implementations are needed:

```python
class ConnectedRepo(XRepo):
    """Repo with live connection to external resource (MongoDB, PostgreSQL, etc.)"""

    def __init__(self, schema: Dict[str, Any], resource: XResource, model_class: Type = None):
        self.schema = schema
        self.resource = resource
        self.model_class = model_class  # Model must be provided or generated by XInspector

    async def list(self, filters: Dict = None, **kwargs) -> List[Any]:
        """Query external resource"""
        return await self.resource.query(filters, **kwargs)

class MaterializedRepo(XRepo):
    """Repo with in-memory data (views, virtual, computed, joined data)"""

    def __init__(self, schema: Dict[str, Any], data: List[Dict[str, Any]], model_class: Type = None):
        self.schema = schema
        self.data = data
        self.model_class = model_class  # Model generated by XInspector and provided during creation

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

        # 2. Determine repo type with smart defaults
        repo_type = cls._determine_repo_type(
            computed_schema,
            resource=resource,
            data=data,
            materialized=kwargs.get('materialized')  # Optional override
        )

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
            # Schema should be provided by XInspector
            # Repo does not perform schema discovery
            raise ValueError("Schema must be provided. Use XInspector to discover schemas.")

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
    def _determine_repo_type(
        cls, 
        schema: Dict[str, Any],
        resource: Optional[XResource] = None,
        data: Optional[List[Dict[str, Any]]] = None,
        materialized: Optional[bool] = None
    ) -> str:
        """Determine repo type with smart defaults and optional override"""

        # 1. Explicit override takes precedence
        if materialized is not None:
            return 'materialized' if materialized else 'connected'

        # 2. If data provided, use MaterializedRepo
        if data is not None:
            return 'materialized'

        # 3. Auto-detect based on resource type
        if resource:
            connection_type = getattr(resource, 'connection_type', None)
            
            # REST/WebSocket/EventStream default to MaterializedRepo
            if connection_type in ['rest_api', 'websocket', 'event_stream']:
                return 'materialized'
            
            # Database/File resources default to ConnectedRepo
            elif connection_type in ['mongodb', 'postgresql', 'mysql', 'csv', 'json']:
                return 'connected'

        # 4. Default fallback
        return 'connected'
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

## Model Integration - Delegated to XInspector

Model generation from schemas is handled by XInspector to maintain clear separation of concerns. XInspector handles all schema discovery and model generation, while Repo focuses purely on data access patterns. The integration follows a two-step process: (1) Inspector generates models, (2) Registry registers them for runtime use. For model generation details, see [XINSPECTOR.md](./XINSPECTOR.md).

### Integration with XInspector

```python
# Model generation is handled by XInspector
from src.server.core.xinspector import XInspector

# Create inspector and generate model
inspector = XInspector(mongo_resource)
User = await inspector.generate_model("users")

# Models are then registered with XRegistry
from src.server.core.xregistry import model_registry
model_registry.register(User, namespace="ns.models.User")
```

### Usage Examples

```python
# Example 1: Create repo with auto-detection
mongo_resource = ResourceFactory.create("mongodb", connection_string="mongodb://localhost/app")
await mongo_resource.connect()

# Inspector generates model
inspector = XInspector(mongo_resource)
User = await inspector.generate_model("users")

# Create repo - auto-detects ConnectedRepo for database
user_repo = await XRepoFactory.create(
    resource=mongo_resource,
    model_class=User,
    collection="users"
)

# Example 2: Create MaterializedRepo for REST/WebSocket
api_resource = ResourceFactory.create("rest_api", endpoint="https://api.example.com")
await api_resource.connect()

# Fetch data and create materialized repo
data = await api_resource.get("/users")
user_repo = await XRepoFactory.create(
    data=data,
    model_class=User,
    materialized=True  # Optional - would auto-detect anyway
)

# Example 3: Override auto-detection
# Force MaterializedRepo for a database resource
user_repo = await XRepoFactory.create(
    resource=mongo_resource,
    model_class=User,
    collection="users",
    materialized=True  # Override auto-detection
)
```

## Enhanced XRepoFactory - Resource-Based Creation

Extended factory methods that enable repo creation directly from resources. Model generation is handled separately by XInspector.

### Extended Factory Methods

```python
class XRepoFactory:
    """Extended factory for resource-based repo creation"""

    # ... existing methods ...

    @classmethod
    async def create_from_resource(
        cls,
        resource: XResource,
        model_class: Type[XObjPrototype],
        collection: str = None,
        materialized: bool = None,
        **kwargs
    ) -> 'XRepo':
        """Create repo from resource with pre-generated model (via XInspector)"""

        # Ensure resource is connected
        if not await resource.validate_connection():
            await resource.connect()

        # Schema must be provided - discovery handled by XInspector
        if not kwargs.get('schema'):
            raise ValueError("Schema must be provided. Use XInspector to discover schemas.")
        schema = kwargs['schema']

        # Determine repo type with smart defaults
        if materialized is None:
            # Auto-detect based on resource type
            materialized = resource.connection_type in ['rest', 'websocket']

        # Create repo based on type
        if materialized:
            # MaterializedRepo for REST/WebSocket data
            return MaterializedRepo(
                schema=schema,
                data=[],  # Will be populated via queries
                model_class=model_class
            )
        else:
            # ConnectedRepo for Database/File resources
            return ConnectedRepo(
                schema=schema,
                resource=resource,
                model_class=model_class
            )

    @classmethod
    async def create_view_from_resources(
        cls,
        name: str,
        resources: List[Tuple[XResource, str]],  # (resource, collection/table) pairs
        join_config: Dict[str, Any],
        result_model_class: Type[XObjPrototype],  # Model must be pre-generated by XInspector
    ) -> 'XRepo':
        """Create view repo from multiple resources using pre-generated model"""

        # Schemas must be provided - discovery handled by XInspector
        if 'schemas' not in kwargs:
            raise ValueError("Schemas must be provided. Use XInspector to discover schemas.")
        schemas = kwargs['schemas']
        source_repos = []
        
        for resource, collection_table in resources:
            if not await resource.validate_connection():
                await resource.connect()
            
            # Create source repo (assuming models exist in registry)
            source_repo = await cls.create_from_resource(
                resource=resource,
                model_class=result_model_class,  # Simplified for now
                collection=collection_table
            )
            source_repos.append(source_repo)

        # Create materialized view repo with joined data
        return MaterializedRepo(
            schema=cls._merge_schemas(schemas, join_config),
            data=[],  # Will be populated by join operations
            model_class=result_model_class
        )

    @classmethod
    async def create_aggregate_repo(
        cls,
        name: str,
        source_repos: List['XRepo'],
        aggregation_config: Dict[str, Any],
        result_model_class: Type[XObjPrototype]  # Model must be pre-generated by XInspector
    ) -> 'XRepo':
        """Create repo that aggregates data from multiple sources"""

        # Merge schemas from source repos
        source_schemas = {}
        for repo in source_repos:
            if hasattr(repo.metadata, 'source_schema'):
                source_schemas[repo.metadata.name] = repo.metadata.source_schema

        # Generate aggregated schema based on aggregation config
        aggregated_schema = cls._create_aggregated_schema(source_schemas, aggregation_config)

        # Create materialized repo for aggregated data
        return MaterializedRepo(
            schema=aggregated_schema,
            data=[],  # Will be populated by aggregation operations
            model_class=result_model_class
        )


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
# Example 1: Simple repo from MongoDB collection (with pre-discovered schema)
mongo_resource = ResourceFactory.create("mongodb", connection_string="mongodb://localhost/app")

# Use XInspector to discover schema
inspector = XInspector(mongo_resource)
schema = await inspector.discover_schema("users")
User = await inspector.generate_model("users")

user_repo = await XRepoFactory.create_from_resource(
    resource=mongo_resource,
    model_class=User,
    collection="users",
    schema=schema
)

# Typed repo (returns proper User instances)
users = await user_repo.list({})  # Returns List[User]

# Example 2: CSV-based repo
csv_resource = ResourceFactory.create("csv", file_path="/data/products.csv")

# Use XInspector for schema discovery and model generation
inspector = XInspector(csv_resource)
schema = await inspector.discover_schema()
Product = await inspector.generate_model("products")

product_repo = await XRepoFactory.create_from_resource(
    resource=csv_resource,
    model_class=Product,
    schema=schema
)

# Example 3: View repo from multiple resources
# First use XInspector to discover schemas
inspector = XInspector(mongo_resource)
user_schema = await inspector.discover_schema("users")
order_schema = await inspector.discover_schema("orders")

# Generate joined model
UserOrderView = await inspector.generate_joined_model({
    "users": user_schema,
    "orders": order_schema
}, join_config={"on": "user_id", "type": "left_join"})

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
    result_model_class=UserOrderView,
    schemas={
        "mongo.users": user_schema,
        "mongo.orders": order_schema
    }
)

# Query the view
user_orders = await user_order_view.list({
    "user_filters": {"status": "active"},
    "order_filters": {"created_at__gte": "2024-01-01"}
})

# Example 4: List available collections/tables and create repos
collections = await mongo_resource.list_collections_or_tables()
print(f"Available collections: {collections}")

# Create repos for all collections using XInspector
inspector = XInspector(mongo_resource)
repos = {}
for collection in collections:
    schema = await inspector.discover_schema(collection)
    model = await inspector.generate_model(collection)
    repos[collection] = await XRepoFactory.create_from_resource(
        resource=mongo_resource,
        model_class=model,
        collection=collection,
        schema=schema
    )
```

## Exception Hierarchy

```python
class XRepoError(Exception):
    """Base exception for all repo operations"""
    pass

class XRepoConnectionError(XRepoError):
    """Connection-related errors"""
    pass

class XRepoPermissionError(XRepoError):
    """Permission/ACL violations"""
    pass

class XRepoValidationError(XRepoError):
    """Data validation errors"""
    pass

class XRepoNotFoundError(XRepoError):
    """Resource not found errors"""
    pass

class XRepoConfigurationError(XRepoError):
    """Repo configuration errors"""
    pass
```

## Key Integration Points

### With XInspector
- Inspector discovers schemas and generates models
- Repo uses models for type-safe data access
- Clear separation: Inspector analyzes, Repo accesses

### With XRegistry
- Generated models are registered in XRegistry
- Repo queries registry for model metadata
- Registry provides UI hints and permissions

### With XResource
- Repo delegates all connection management to Resource
- Resource provides query/update capabilities
- Repo adds high-level abstractions on top

## CacheManager Namespace Extension

Based on architectural decisions, the CacheManager from XRepo is extended to provide namespace registration capabilities, avoiding the need for a separate NameService.

### Namespace Registration Methods

```python
class CacheManager:
    """Extended cache manager with namespace capabilities"""
    
    def __init__(self, backend: str = "memory"):
        """Initialize with memory or redis backend"""
        if backend == "memory":
            self._cache = InMemoryRepo()
        elif backend == "redis":
            self._cache = RedisRepo()
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
    
    # Namespace registration methods
    async def register_ns(self, namespace_path: str, value: Any) -> None:
        """Register object in namespace"""
        await self._cache.set(namespace_path, value)
    
    async def get_ns(self, namespace_path: str, default: Any = None) -> Any:
        """Get object from namespace"""
        return await self._cache.get(namespace_path, default)
    
    async def register_indirect(self, key: str, namespace_path: str) -> None:
        """Register indirect mapping"""
        await self._cache.set(f"_indirect.{key}", namespace_path)
    
    async def get_ns_by_key(self, key: str) -> str:
        """Get namespace path by indirect key"""
        return await self._cache.get(f"_indirect.{key}")
    
    async def list_ns(self, pattern: str) -> list[str]:
        """List namespace paths matching pattern"""
        return await self._cache.keys(pattern)
    
    async def fuzzy_search(self, pattern: str, search_term: str) -> list[str]:
        """Search for namespace paths containing term"""
        all_keys = await self.list_ns(pattern)
        results = []
        for key in all_keys:
            value = await self.get_ns(key, "")
            if search_term.lower() in str(value).lower():
                results.append(key)
        return results
```

### Integration with Component Registration

```python
# Manual registration order during startup
async def initialize_namespace_system(cache_manager: CacheManager):
    """Initialize namespace system with manual registration order"""
    
    # 1. Settings
    settings = XSettings()
    await cache_manager.register_ns("ns.settings.app", settings)
    
    # 2. Resources (with internal pooling)
    db_resource = DatabaseResource(settings.database)  # Manages own pool
    await cache_manager.register_ns("ns.resources.db", db_resource)
    
    # 3. Inspector
    inspector = XInspector(db_resource)
    await cache_manager.register_ns("ns.inspector.db", inspector)
    
    # 4. Repos (with smart factory)
    user_repo = await XRepoFactory.create_from_resource(
        resource=db_resource,
        model_class=UserModel,  # Pre-generated by Inspector
        materialized=None  # Auto-detects: DB → ConnectedRepo
    )
    await cache_manager.register_ns("ns.repos.users", user_repo)
    
    # 5. Registry (registration only - generation done by Inspector)
    UserModel = await inspector.generate_model("users")  # Inspector generates
    model_registry.register(UserModel)  # Registry only registers
    await cache_manager.register_ns("ns.models.User", UserModel)
    
    # 6. Fuzzy Search Index Registration
    # Concatenate all fuzzy searchable fields into single string
    user = UserModel(name="John Doe", email="john@example.com", role="admin")
    fuzzy_text = user.get_fuzzy_text()  # "John Doe john@example.com"
    await cache_manager.register_ns("ns.fuzzy.users.user123", fuzzy_text)
```

### Namespace Structure

- `ns.models.*` - All registered models
- `ns.repos.*` - All repo instances  
- `ns.resources.*` - Resource connections
- `ns.settings.*` - Configuration objects
- `ns.fuzzy.{entity_type}.{id}` - Fuzzy searchable field concatenations

## Summary

The Repo abstraction provides:
1. **Unified Interface**: Same API regardless of data source
2. **Smart Defaults**: Auto-detection of repo type based on resource
3. **Type Safety**: Full Pydantic model support
4. **Flexibility**: Optional override of auto-detection
5. **Clean Architecture**: Clear separation from schema discovery (XInspector) and model registry (XRegistry)
6. **Namespace Integration**: CacheManager extension provides namespace registration without separate service
