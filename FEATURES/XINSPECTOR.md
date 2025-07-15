# Inspector Architecture

## Overview

The Inspector module provides a unified interface for data inspection, schema discovery, and model generation across all data sources in the XObjPrototype ecosystem. It consolidates functionality previously scattered across XResource, Repo, and Registry components.

## Cross-References

- **Used By**:
  - [XRESOURCE.md](./XRESOURCE.md) - Resources delegate schema discovery to Inspector
  - [XREPO.md](./XREPO.md) - Repo uses Inspector for model generation
  - [XREGISTRY.md](./XREGISTRY.md) - Registry uses Inspector-generated models
- **Generates**: Models that inherit from [XOBJPROTOTYPE.md](./XOBJPROTOTYPE.md)
- **See Also**: [CLAUDE.md](./CLAUDE.md) for architecture overview

## Core Responsibilities

1. **Schema Discovery** - Inspect and extract schema from any data source
2. **Data Profiling** - Statistical analysis and data quality assessment
3. **Model Generation** - Create Pydantic models from discovered schemas (**Sole responsibility in the architecture**)
4. **Data Preview** - Sample data extraction and visualization
5. **Metadata Extraction** - Gather comprehensive metadata about data sources
6. **Categorical Field Detection** - Identify and manage enumerable fields with distinct values

**Important**: XInspector is the only component responsible for model generation. Other components (XRepo, XRegistry) use Inspector-generated models but never generate models themselves.

## Design Principles

- **Separation of Concerns**: Inspector handles all inspection-related tasks
- **Ultrathin Design**: Minimal abstraction layers
- **Resource Agnostic**: Works with any XResource implementation
- **Lazy Evaluation**: Inspection operations are performed on-demand
- **Caching**: Results are cached to avoid redundant operations

## Architecture

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel
from src.server.core.xobj_prototype import XObjPrototype
from src.server.core.xresource import XResource

class CategoricalFieldInfo(BaseModel):
    """Information about a categorical field"""
    field_name: str
    distinct_values: List[Any]  # Sorted list of unique values
    count: int  # Number of distinct values
    percentage: float  # Percentage of distinct vs total records
    ns_path: str  # Namespace path e.g., "ns.enum.users.status"
    uri: str  # URI endpoint e.g., "/enum/users/status"

class InspectionResult(XObjPrototype):
    """Result of an inspection operation"""
    schema: Dict[str, Any]
    statistics: Dict[str, Any]
    sample_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    categorical_fields: Dict[str, CategoricalFieldInfo]  # Categorical field analysis
    model_name: str
    ns: str = "inspector.results"

class BaseInspector(ABC):
    """Abstract base inspector"""

    def __init__(self, resource: XResource):
        self.resource = resource
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    async def discover_schema(self) -> Dict[str, Any]:
        """Discover schema from resource"""
        pass

    @abstractmethod
    async def profile_data(self) -> Dict[str, Any]:
        """Generate statistical profile of data"""
        pass

    @abstractmethod
    async def preview_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sample data"""
        pass

    @abstractmethod
    async def detect_categorical_fields(self) -> Dict[str, CategoricalFieldInfo]:
        """Detect categorical fields with distinct values"""
        pass

    async def _register_enum_namespace(self, field_info: CategoricalFieldInfo) -> None:
        """Register categorical field values in namespace"""
        # Will be implemented with CacheManager integration
        pass

    async def inspect(self) -> InspectionResult:
        """Perform full inspection"""
        schema = await self.discover_schema()
        statistics = await self.profile_data()
        sample_data = await self.preview_data()
        metadata = await self.resource.get_metadata()
        categorical_fields = await self.detect_categorical_fields()

        # Register categorical fields in namespace
        for field_info in categorical_fields.values():
            await self._register_enum_namespace(field_info)

        return InspectionResult(
            schema=schema,
            statistics=statistics,
            sample_data=sample_data,
            metadata=metadata,
            categorical_fields=categorical_fields,
            model_name=self._generate_model_name()
        )
    
    async def generate_model(self, collection_name: Optional[str] = None) -> Type[XObjPrototype]:
        """Convenience method to inspect and generate model in one step"""
        result = await self.inspect()
        if collection_name:
            result.model_name = collection_name
        return ModelGenerator.generate_model(result)
```

## Inspector Implementations

### DatabaseInspector

```python
class DatabaseInspector(BaseInspector):
    """Inspector for database resources"""

    async def discover_schema(self) -> Dict[str, Any]:
        """Discover database schema"""
        # Implementation for MongoDB, PostgreSQL, MySQL etc.
        pass

    async def profile_data(self) -> Dict[str, Any]:
        """Profile database collections/tables"""
        # Count, distinct values, null counts, data types
        pass

    async def detect_categorical_fields(self) -> Dict[str, CategoricalFieldInfo]:
        """Detect categorical fields in database"""
        categorical_fields = {}
        collection_name = self.resource.metadata.get('collection')
        
        # Get field statistics
        schema = await self.discover_schema()
        total_count = await self._get_total_count()
        
        for field_name, field_info in schema.get('fields', {}).items():
            # Skip numeric and date fields
            if field_info.get('type') in ['integer', 'float', 'datetime']:
                continue
                
            # Get distinct values
            distinct_values = await self._get_distinct_values(field_name)
            distinct_count = len(distinct_values)
            
            # Check if field is categorical (configurable threshold)
            max_threshold = self.resource.settings.get('categorical_max_distinct', 100)
            if distinct_count <= max_threshold and distinct_count > 0:
                # Sort values for consistency
                sorted_values = sorted(distinct_values, key=str)
                
                # Create categorical field info
                ns_path = f"ns.enum.{collection_name}.{field_name}"
                uri = f"/enum/{collection_name}/{field_name}"
                
                categorical_fields[field_name] = CategoricalFieldInfo(
                    field_name=field_name,
                    distinct_values=sorted_values,
                    count=distinct_count,
                    percentage=(distinct_count / total_count * 100) if total_count > 0 else 0,
                    ns_path=ns_path,
                    uri=uri
                )
        
        return categorical_fields
```

### FileInspector

```python
class FileInspector(BaseInspector):
    """Inspector for file-based resources"""

    async def discover_schema(self) -> Dict[str, Any]:
        """Infer schema from file content"""
        # CSV, JSON, Excel, Parquet etc.
        pass
```

### APIInspector

```python
class APIInspector(BaseInspector):
    """Inspector for REST/GraphQL endpoints"""

    async def discover_schema(self) -> Dict[str, Any]:
        """Discover API schema from OpenAPI/GraphQL introspection"""
        pass
```

## Model Generator

```python
class ModelGenerator:
    """Generate Pydantic models from inspection results"""

    @staticmethod
    def generate_model(
        inspection_result: InspectionResult,
        base_class: Type[BaseModel] = XObjPrototype
    ) -> Type[BaseModel]:
        """
        Generate a Pydantic model from inspection result

        Args:
            inspection_result: Result from inspector
            base_class: Base class for generated model

        Returns:
            Dynamically created Pydantic model class
        """
        schema = inspection_result.schema
        model_name = inspection_result.model_name

        # Convert schema to Pydantic field definitions
        fields = ModelGenerator._schema_to_fields(schema)

        # Create model dynamically
        model = type(model_name, (base_class,), {
            '__annotations__': fields,
            '__module__': 'inspector.generated',
            'ns': f"models.generated.{model_name.lower()}"
        })

        return model

    @staticmethod
    def _schema_to_fields(schema: Dict[str, Any], categorical_fields: Dict[str, CategoricalFieldInfo] = None) -> Dict[str, Any]:
        """Convert schema definition to Pydantic field types"""
        fields = {}
        categorical_fields = categorical_fields or {}
        
        for field_name, field_info in schema.get('fields', {}).items():
            # Check if field is categorical
            if field_name in categorical_fields:
                # Create Enum type for categorical field
                cat_info = categorical_fields[field_name]
                enum_name = f"{field_name.title()}Enum"
                enum_values = {str(v).upper().replace(' ', '_'): v for v in cat_info.distinct_values}
                field_enum = Enum(enum_name, enum_values)
                fields[field_name] = (field_enum, Field(...))
            else:
                # Standard field type mapping
                field_type = ModelGenerator._map_field_type(field_info.get('type'))
                fields[field_name] = (field_type, Field(...))
        
        return fields
```

## Inspector Factory

```python
class InspectorFactory:
    """Factory for creating appropriate inspector instances"""

    _inspectors = {
        'mongodb': DatabaseInspector,
        'postgresql': DatabaseInspector,
        'mysql': DatabaseInspector,
        'file': FileInspector,
        'csv': FileInspector,
        'json': FileInspector,
        'rest': APIInspector,
        'graphql': APIInspector,
    }

    @classmethod
    def create(cls, resource: XResource) -> BaseInspector:
        """Create inspector for given resource"""
        resource_type = resource.get_type()
        inspector_class = cls._inspectors.get(resource_type)

        if not inspector_class:
            raise ValueError(f"No inspector for resource type: {resource_type}")

        return inspector_class(resource)
```

## Usage Examples

### Basic Inspection

```python
from src.server.core.xresource import XResourceFactory
from src.server.core.inspector import InspectorFactory, ModelGenerator

# Create resource
resource = XResourceFactory.create({
    'type': 'mongodb',
    'uri': 'mongodb://localhost:27017/mydb',
    'collection': 'users'
})

# Create inspector
inspector = InspectorFactory.create(resource)

# Perform inspection
result = await inspector.inspect()

# Generate model
UserModel = ModelGenerator.generate_model(result)

# Use generated model
user = UserModel(name="John", email="john@example.com")
```

### Schema Discovery Only

```python
# Just discover schema without full inspection
schema = await inspector.discover_schema()
print(f"Discovered schema: {schema}")
```

### Data Profiling

```python
# Profile data for quality assessment
profile = await inspector.profile_data()
print(f"Null count: {profile['null_count']}")
print(f"Unique values: {profile['unique_count']}")
```

### Categorical Field Detection

```python
# Inspect with categorical field detection
result = await inspector.inspect()

# Access categorical fields
for field_name, cat_info in result.categorical_fields.items():
    print(f"Field: {field_name}")
    print(f"  Distinct values: {cat_info.distinct_values}")
    print(f"  Count: {cat_info.count}")
    print(f"  Namespace: {cat_info.ns_path}")
    print(f"  URI: {cat_info.uri}")

# Example output for a 'status' field:
# Field: status
#   Distinct values: ['active', 'inactive', 'pending']
#   Count: 3
#   Namespace: ns.enum.users.status
#   URI: /enum/users/status
```

## Integration Points

### With XResource (see [XRESOURCE.md](./XRESOURCE.md))

```python
class XResource:
    """Updated to delegate to Inspector"""

    async def discover_schema(self) -> Dict[str, Any]:
        """Delegate to inspector"""
        inspector = InspectorFactory.create(self)
        return await inspector.discover_schema()
```

### With Repo (see [XREPO.md](./XREPO.md))

```python
class Repo:
    """Updated to use Inspector for model generation"""

    async def generate_model(self) -> Type[BaseModel]:
        """Generate model using Inspector"""
        inspector = InspectorFactory.create(self.resource)
        result = await inspector.inspect()
        return ModelGenerator.generate_model(result)
```

### With Dynamic Model Registry

```python
class ModelRegistry:
    """Use Inspector for automatic model registration"""

    async def register_from_resource(self, resource: XResource):
        """Register model discovered from resource"""
        inspector = InspectorFactory.create(resource)
        result = await inspector.inspect()
        model = ModelGenerator.generate_model(result)

        self.register(
            name=result.model_name,
            model=model,
            metadata=result.metadata
        )
```

## Advanced Features

### Schema Evolution Detection

```python
class SchemaEvolutionDetector:
    """Detect schema changes over time"""

    async def detect_changes(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any]
    ) -> List[SchemaChange]:
        """Detect schema evolution"""
        # Implementation
        pass
```

### Data Quality Rules

```python
class DataQualityInspector(BaseInspector):
    """Enhanced inspector with data quality rules"""

    def __init__(self, resource: XResource, rules: List[QualityRule]):
        super().__init__(resource)
        self.rules = rules

    async def validate_quality(self) -> QualityReport:
        """Validate data against quality rules"""
        # Implementation
        pass
```

## Performance Considerations

1. **Lazy Loading**: Inspection operations are performed only when needed
2. **Caching**: Results are cached with configurable TTL
3. **Sampling**: For large datasets, use statistical sampling
4. **Async Operations**: All inspection methods are async for non-blocking execution
5. **Resource Limits**: Configurable limits on preview data and profiling depth

## Error Handling

```python
class InspectionError(Exception):
    """Base exception for inspection errors"""
    pass

class SchemaDiscoveryError(InspectionError):
    """Failed to discover schema"""
    pass

class InsufficientDataError(InspectionError):
    """Not enough data for meaningful inspection"""
    pass
```

## Configuration

```toml
# settings.toml
[inspector]
cache_ttl = 3600  # Cache results for 1 hour
preview_limit = 100  # Default preview rows
profile_sample_size = 10000  # Sample size for profiling
enable_schema_evolution = true

[inspector.categorical]
max_distinct_threshold = 100  # Max distinct values to consider categorical
min_sample_size = 100  # Min records to analyze for categorical detection
cache_ttl = 3600  # Cache enum values for 1 hour
auto_refresh = true  # Refresh enums on schema changes
include_numeric = false  # Whether to check numeric fields for categorical
```

## Testing

```python
import pytest
from unittest.mock import Mock

@pytest.mark.asyncio
async def test_inspector_factory():
    """Test inspector creation"""
    resource = Mock(spec=XResource)
    resource.get_type.return_value = 'mongodb'

    inspector = InspectorFactory.create(resource)
    assert isinstance(inspector, DatabaseInspector)

@pytest.mark.asyncio
async def test_model_generation():
    """Test dynamic model generation"""
    result = InspectionResult(
        schema={'name': 'string', 'age': 'integer'},
        statistics={},
        sample_data=[],
        metadata={},
        model_name='Person'
    )

    PersonModel = ModelGenerator.generate_model(result)
    person = PersonModel(name="Alice", age=30)
    assert person.name == "Alice"
```

## API Endpoints

### Categorical Field Access

```python
from fastapi import FastAPI, HTTPException
from src.server.core.cache_manager import CacheManager

app = FastAPI()
cache_manager = CacheManager()

@app.get("/enum/{collection}/{field}")
async def get_enum_values(collection: str, field: str):
    """Get valid enum values for a categorical field"""
    ns_path = f"ns.enum.{collection}.{field}"
    
    # Retrieve from namespace cache
    values = await cache_manager.get(ns_path)
    
    if values is None:
        raise HTTPException(404, f"Enum values not found for {collection}.{field}")
    
    return {
        "collection": collection,
        "field": field,
        "values": values,
        "count": len(values),
        "ns_path": ns_path
    }

@app.get("/enum/{collection}")
async def list_categorical_fields(collection: str):
    """List all categorical fields for a collection"""
    # Get all enum namespaces for collection
    pattern = f"ns.enum.{collection}.*"
    enum_fields = await cache_manager.search(pattern)
    
    fields = []
    for ns_path in enum_fields:
        field_name = ns_path.split('.')[-1]
        values = await cache_manager.get(ns_path)
        fields.append({
            "field": field_name,
            "count": len(values),
            "uri": f"/enum/{collection}/{field_name}"
        })
    
    return {
        "collection": collection,
        "categorical_fields": fields
    }
```

## Future Enhancements

1. **ML-based Type Inference**: Use machine learning for better type detection
2. **Schema Recommendation**: Suggest optimal schema based on data patterns
3. **Performance Profiling**: Inspect query performance characteristics
4. **Data Lineage**: Track data flow and transformations
5. **Schema Registry Integration**: Connect with external schema registries
6. **Dynamic Enum Updates**: Auto-detect new categorical values and update enums
7. **Hierarchical Categories**: Support for nested categorical structures
