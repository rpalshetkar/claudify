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
        return await self._db.list_collection_names()

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
