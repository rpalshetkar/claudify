# XResource - Resource Abstraction Documentation

## Overview

XResource is a unified abstraction layer for managing various types of data connections and resources. It implements a factory pattern to create and manage connections to files, databases, REST endpoints, and websockets while inheriting from XObjPrototype for consistent validation and behavior.

## Design Principles

- **Ultrathin abstraction**: Minimal overhead while providing essential functionality
- **Factory pattern**: Creates appropriate connection instances based on resource type
- **Metadata-rich**: Stores comprehensive connection parameters and metadata
- **Type-safe**: Leverages Pydantic for validation and type checking
- **Async-first**: Built for asynchronous operations where applicable

## Resource Types

### File Resources

- **CSV**: Tabular data with configurable delimiters and encoding
- **Excel (XLS/XLSX)**: Spreadsheet data with sheet selection support
- **JSON**: Structured data with schema validation
- **YAML**: Configuration and structured data

### Database Resources

- **MongoDB**: Document database with collection metadata
- **PostgreSQL**: Relational database with schema awareness
- **MySQL**: Relational database with connection pooling

### Network Resources

- **REST API**: HTTP/HTTPS endpoints with authentication support
- **WebSocket**: Real-time bidirectional communication

## Architecture

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, Field
from src.server.core.xobj_prototype import XObjPrototype

T = TypeVar('T', bound='XResource')

class ResourceMetadata(BaseModel):
    """Base metadata for all resources"""
    name: str
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

class XResource(XObjPrototype, ABC):
    """Abstract base class for all resources"""

    metadata: ResourceMetadata
    connection_params: Dict[str, Any]
    _connection: Optional[Any] = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the resource"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the resource"""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate if connection is active and healthy"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return resource-specific metadata"""
        pass

    @abstractmethod
    async def discover_schema(self, collection: str = None) -> Dict[str, Any]:
        """Discover schema information from the resource

        Delegates to Inspector - see XINSPECTOR.md for implementation details
        """
        pass

    @abstractmethod
    async def list_collections_or_tables(self) -> List[str]:
        """List available collections/tables in the resource"""
        pass

class FileResource(XResource):
    """Base class for file-based resources"""
    file_path: str
    encoding: str = "utf-8"

class DatabaseResource(XResource):
    """Base class for database resources"""
    connection_string: str
    pool_size: int = 10

class NetworkResource(XResource):
    """Base class for network resources"""
    endpoint: str
    timeout: int = 30
    auth_params: Optional[Dict[str, Any]] = None
```

## Resource Factory

```python
class ResourceFactory:
    """Factory for creating resource instances"""

    _registry: Dict[str, Type[XResource]] = {}

    @classmethod
    def register(cls, resource_type: str, resource_class: Type[XResource]) -> None:
        """Register a resource type"""
        cls._registry[resource_type] = resource_class

    @classmethod
    def create(cls, resource_type: str, **kwargs) -> XResource:
        """Create a resource instance"""
        if resource_type not in cls._registry:
            raise ValueError(f"Unknown resource type: {resource_type}")

        resource_class = cls._registry[resource_type]
        return resource_class(**kwargs)
```

## Usage Examples

### File Resource Example

```python
# CSV Resource
csv_resource = ResourceFactory.create(
    "csv",
    name="sales_data",
    file_path="/data/sales_2024.csv",
    delimiter=",",
    has_header=True,
    metadata=ResourceMetadata(
        name="sales_data",
        description="Monthly sales data for 2024",
        tags=["sales", "2024", "monthly"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
)

await csv_resource.connect()
data = await csv_resource.read()
await csv_resource.disconnect()
```

### Database Resource Example

```python
# MongoDB Resource
mongo_resource = ResourceFactory.create(
    "mongodb",
    name="user_db",
    connection_string="mongodb://localhost:27017/myapp",
    database="myapp",
    metadata=ResourceMetadata(
        name="user_db",
        description="User database",
        tags=["users", "mongodb"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
)

await mongo_resource.connect()

# Get collection metadata
collections = await mongo_resource.get_collections()
# collections = ["users", "sessions", "profiles"]

# Access specific collection
users_collection = await mongo_resource.get_collection("users")
await mongo_resource.disconnect()
```

### REST API Resource Example

```python
# REST API Resource
api_resource = ResourceFactory.create(
    "rest_api",
    name="weather_api",
    endpoint="https://api.weather.com/v1",
    auth_params={"api_key": settings.WEATHER_API_KEY},
    timeout=30,
    metadata=ResourceMetadata(
        name="weather_api",
        description="Weather data API",
        tags=["weather", "external", "api"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
)

await api_resource.connect()
weather_data = await api_resource.get("/current", params={"city": "London"})
await api_resource.disconnect()
```

## Implementation Details

### MongoDB Resource

```python
class MongoDBResource(DatabaseResource):
    """MongoDB database resource"""

    database: str
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        """Connect to MongoDB"""
        self._client = AsyncIOMotorClient(self.connection_string)
        self._db = self._client[self.database]
        self._connection = self._db

    async def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self._client:
            self._client.close()

    async def validate_connection(self) -> bool:
        """Check if MongoDB connection is alive"""
        try:
            await self._client.admin.command('ping')
            return True
        except Exception:
            return False

    async def get_collections(self) -> list[str]:
        """Get list of collections in database"""
        return await self._db.list_collections()

    async def get_collection(self, name: str) -> AsyncIOMotorCollection:
        """Get specific collection"""
        return self._db[name]

    def get_metadata(self) -> Dict[str, Any]:
        """Return MongoDB-specific metadata"""
        return {
            "database": self.database,
            "connection_string": self.connection_string.split('@')[-1],  # Hide credentials
            "type": "mongodb"
        }

    async def discover_schema(self, collection: str = None) -> Dict[str, Any]:
        """Discover schema from MongoDB collection by analyzing sample documents"""
        if not collection:
            raise ValueError("collection is required for MongoDB schema discovery")

        collection = self._db[collection]

        # Sample documents to infer schema
        sample_size = 100
        sample_docs = await collection.aggregate([
            {"$sample": {"size": sample_size}},
            {"$limit": sample_size}
        ]).to_list(length=sample_size)

        if not sample_docs:
            return {
                "collection": collection,
                "fields": {},
                "sample_count": 0,
                "indexes": await self._get_collection_indexes(collection)
            }

        # Analyze field types and patterns
        field_analysis = {}
        for doc in sample_docs:
            self._analyze_document_fields(doc, field_analysis)

        # Convert analysis to schema format
        schema = {
            "collection": collection,
            "fields": self._build_field_schema(field_analysis, len(sample_docs)),
            "sample_count": len(sample_docs),
            "document_count": await collection.count_documents({}),
            "indexes": await self._get_collection_indexes(collection)
        }

        return schema

    async def list_collections_or_tables(self) -> List[str]:
        """List all collections in the MongoDB database"""
        return await self._db.list_collections()

    def _analyze_document_fields(self, doc: Dict[str, Any], field_analysis: Dict[str, Dict]):
        """Recursively analyze document fields"""
        for key, value in doc.items():
            if key not in field_analysis:
                field_analysis[key] = {
                    "types": {},
                    "null_count": 0,
                    "total_count": 0,
                    "sample_values": set(),
                    "nested_fields": {}
                }

            field_info = field_analysis[key]
            field_info["total_count"] += 1

            if value is None:
                field_info["null_count"] += 1
            else:
                value_type = type(value).__name__
                field_info["types"][value_type] = field_info["types"].get(value_type, 0) + 1

                # Store sample values for type inference
                if len(field_info["sample_values"]) < 10:
                    field_info["sample_values"].add(str(value)[:100])  # Truncate long values

                # Handle nested documents
                if isinstance(value, dict):
                    self._analyze_document_fields(value, field_info["nested_fields"])
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # Analyze first few array elements if they're objects
                    for item in value[:5]:
                        if isinstance(item, dict):
                            self._analyze_document_fields(item, field_info["nested_fields"])

    def _build_field_schema(self, field_analysis: Dict[str, Dict], total_docs: int) -> Dict[str, Dict]:
        """Convert field analysis to standardized schema format"""
        schema_fields = {}

        for field_name, analysis in field_analysis.items():
            # Determine primary type
            if analysis["types"]:
                primary_type = max(analysis["types"], key=analysis["types"].get)
            else:
                primary_type = "null"

            # Calculate statistics
            null_percentage = (analysis["null_count"] / total_docs) * 100
            is_required = null_percentage < 10  # Less than 10% null = required

            # Map MongoDB types to Python/Pydantic types
            python_type = self._map_mongodb_type_to_python(primary_type, analysis)

            schema_fields[field_name] = {
                "type": python_type,
                "required": is_required,
                "null_percentage": null_percentage,
                "mongodb_types": analysis["types"],
                "sample_values": list(analysis["sample_values"]),
                "cardinality": self._estimate_cardinality(analysis["sample_values"], analysis["total_count"])
            }

            # Add nested schema if applicable
            if analysis["nested_fields"]:
                schema_fields[field_name]["nested_schema"] = self._build_field_schema(
                    analysis["nested_fields"],
                    analysis["total_count"]
                )

        return schema_fields

    def _map_mongodb_type_to_python(self, mongo_type: str, analysis: Dict) -> str:
        """Map MongoDB types to Python types for model generation"""
        type_mapping = {
            "str": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "list": "List[Any]",
            "dict": "Dict[str, Any]",
            "datetime": "datetime",
            "ObjectId": "str",  # Convert ObjectId to string
            "Decimal128": "Decimal"
        }

        base_type = type_mapping.get(mongo_type, "Any")

        # Enhance type based on analysis
        if mongo_type == "str":
            # Check if it looks like an email, URL, etc.
            sample_values = analysis.get("sample_values", [])
            if any("@" in val for val in sample_values):
                return "EmailStr"
            elif any(val.startswith(("http://", "https://")) for val in sample_values):
                return "HttpUrl"

        return base_type

    def _estimate_cardinality(self, sample_values: set, total_count: int) -> str:
        """Estimate field cardinality for UI widget suggestions"""
        if not sample_values:
            return "unknown"

        unique_ratio = len(sample_values) / total_count if total_count > 0 else 0

        if unique_ratio < 0.05:
            return "low"    # Good for select dropdowns
        elif unique_ratio < 0.5:
            return "medium" # Might be good for autocomplete
        else:
            return "high"   # Probably unique identifiers

    async def _get_collection_indexes(self, collection: str) -> List[Dict[str, Any]]:
        """Get index information for a collection"""
        collection = self._db[collection]
        indexes = []

        async for index in collection.list_indexes():
            indexes.append({
                "name": index.get("name"),
                "fields": list(index.get("key", {}).keys()),
                "unique": index.get("unique", False),
                "sparse": index.get("sparse", False)
            })

        return indexes
```

### CSV Resource

```python
class CSVResource(FileResource):
    """CSV file resource"""

    delimiter: str = ","
    has_header: bool = True
    quotechar: str = '"'

    async def connect(self) -> None:
        """Open CSV file"""
        # For async file operations, could use aiofiles
        self._connection = Path(self.file_path)
        if not self._connection.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

    async def disconnect(self) -> None:
        """Close file connection"""
        self._connection = None

    async def validate_connection(self) -> bool:
        """Check if file exists and is readable"""
        return self._connection and self._connection.exists()

    async def read(self) -> list[Dict[str, Any]]:
        """Read CSV data"""
        # Implementation would use csv.DictReader or pandas
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return CSV-specific metadata"""
        return {
            "file_path": self.file_path,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
            "encoding": self.encoding,
            "type": "csv"
        }

    async def discover_schema(self, collection: str = None) -> Dict[str, Any]:
        """Discover schema from CSV file by analyzing headers and sample data"""
        import csv
        import io
        from pathlib import Path

        if not self._connection or not self._connection.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        # Read CSV file and analyze
        with open(self.file_path, 'r', encoding=self.encoding) as file:
            # Detect delimiter if not specified
            if self.delimiter == ",":
                dialect = csv.Sniffer().sniff(file.read(1024))
                file.seek(0)
                actual_delimiter = dialect.delimiter
            else:
                actual_delimiter = self.delimiter

            reader = csv.DictReader(file, delimiter=actual_delimiter)

            # Get field names from header
            if not self.has_header:
                # Generate column names if no header
                first_row = next(reader, [])
                fieldnames = [f"column_{i}" for i in range(len(first_row))]
                file.seek(0)
                reader = csv.DictReader(file, fieldnames=fieldnames, delimiter=actual_delimiter)

            fieldnames = reader.fieldnames or []

            # Analyze sample data
            field_analysis = {name: {"values": [], "types": {}} for name in fieldnames}
            row_count = 0

            for i, row in enumerate(reader):
                if i >= 100:  # Limit sample size
                    break
                row_count += 1

                for field_name, value in row.items():
                    if field_name in field_analysis:
                        field_analysis[field_name]["values"].append(value)

                        # Type detection
                        detected_type = self._detect_csv_type(value)
                        field_analysis[field_name]["types"][detected_type] = \
                            field_analysis[field_name]["types"].get(detected_type, 0) + 1

        # Build schema
        schema = {
            "file": self.file_path,
            "fields": {},
            "row_count": row_count,
            "delimiter": actual_delimiter,
            "has_header": self.has_header
        }

        for field_name, analysis in field_analysis.items():
            # Determine primary type
            if analysis["types"]:
                primary_type = max(analysis["types"], key=analysis["types"].get)
            else:
                primary_type = "str"

            # Calculate statistics
            null_count = analysis["values"].count("") + analysis["values"].count(None)
            null_percentage = (null_count / len(analysis["values"])) * 100 if analysis["values"] else 0

            schema["fields"][field_name] = {
                "type": primary_type,
                "required": null_percentage < 10,
                "null_percentage": null_percentage,
                "sample_values": analysis["values"][:10],
                "cardinality": self._estimate_cardinality(set(analysis["values"]), len(analysis["values"]))
            }

        return schema

    async def list_collections_or_tables(self) -> List[str]:
        """For CSV, return the filename as the single 'table'"""
        return [Path(self.file_path).stem]

    def _detect_csv_type(self, value: str) -> str:
        """Detect Python type from CSV string value"""
        if not value or value.strip() == "":
            return "null"

        value = value.strip()

        # Try integer
        try:
            int(value)
            return "int"
        except ValueError:
            pass

        # Try float
        try:
            float(value)
            return "float"
        except ValueError:
            pass

        # Try boolean
        if value.lower() in ("true", "false", "yes", "no", "1", "0"):
            return "bool"

        # Try date/datetime patterns
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]

        for pattern in date_patterns:
            if re.match(pattern, value):
                return "datetime"

        # Default to string
        return "str"
```

## Testing Strategy

### Unit Tests

```python
import pytest
from datetime import datetime
from src.server.core.resources import ResourceFactory, CSVResource, MongoDBResource

@pytest.fixture
def csv_resource():
    """Create a test CSV resource"""
    return ResourceFactory.create(
        "csv",
        name="test_csv",
        file_path="/tmp/test.csv",
        metadata=ResourceMetadata(
            name="test_csv",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    )

@pytest.mark.asyncio
async def test_csv_resource_connect(csv_resource, tmp_path):
    """Test CSV resource connection"""
    # Create test file
    test_file = tmp_path / "test.csv"
    test_file.write_text("name,age\nJohn,30\nJane,25")

    csv_resource.file_path = str(test_file)
    await csv_resource.connect()

    assert await csv_resource.validate_connection()
    await csv_resource.disconnect()

@pytest.mark.asyncio
async def test_resource_factory_registry():
    """Test resource factory registration"""
    # Register custom resource
    class CustomResource(XResource):
        async def connect(self): pass
        async def disconnect(self): pass
        async def validate_connection(self): return True
        def get_metadata(self): return {}

    ResourceFactory.register("custom", CustomResource)
    resource = ResourceFactory.create("custom", name="test")

    assert isinstance(resource, CustomResource)
```

## Integration with Inspector

The Resource component delegates all schema discovery operations to the Inspector (see [XINSPECTOR.md](./XINSPECTOR.md)). This separation of concerns ensures that Resource focuses on connection management while Inspector handles the complex logic of schema inference and data profiling.

### Schema Discovery Flow

1. Resource establishes connection to data source
2. Passes connection handle to Inspector
3. Inspector performs schema discovery and profiling
4. Returns discovered schema to Resource
5. Resource caches schema metadata

For detailed schema discovery implementation, refer to [XINSPECTOR.md](./XINSPECTOR.md).

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_mongodb_resource_operations():
    """Test MongoDB resource with real connection"""
    mongo_resource = ResourceFactory.create(
        "mongodb",
        name="test_db",
        connection_string="mongodb://localhost:27017/test",
        database="test",
        metadata=ResourceMetadata(
            name="test_db",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    )

    await mongo_resource.connect()

    # Test connection validation
    assert await mongo_resource.validate_connection()

    # Test collection operations
    collections = await mongo_resource.get_collections()
    assert isinstance(collections, list)

    # Test metadata
    metadata = mongo_resource.get_metadata()
    assert metadata["type"] == "mongodb"
    assert metadata["database"] == "test"

    await mongo_resource.disconnect()
```

## Configuration Integration

Resources can be configured via DynaConf settings:

```toml
# settings.toml
[resources.mongodb]
default_connection_string = "mongodb://localhost:27017"
default_pool_size = 10
connection_timeout = 5000

[resources.csv]
default_encoding = "utf-8"
default_delimiter = ","

[resources.api]
default_timeout = 30
retry_attempts = 3
```

## Namespace Integration

Will be added later

## Future Enhancements

1. **Connection Pooling**: Implement connection pooling for database resources
2. **Caching**: Add caching layer for frequently accessed resources
3. **Monitoring**: Resource usage metrics and health checks
4. **Schema Discovery**: Automatic schema detection for databases
5. **Transaction Support**: Distributed transaction coordination
6. **Resource Lifecycle**: Automatic cleanup and resource management

## Security Considerations

- Connection strings must never contain plaintext passwords
- Use settings/secrets management for credentials
- Implement connection encryption where supported
- Add request signing for API resources
- Audit trail for resource access

## Performance Optimization

- Lazy loading of connections
- Connection pooling for databases
- Batch operations support
- Streaming for large files
- Async/await throughout
