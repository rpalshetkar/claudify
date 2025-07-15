# XResource - Resource Abstraction Documentation

## Overview

XResource is a unified abstraction layer for managing various types of data connections and resources. It implements a factory pattern to create and manage connections to files, databases, REST endpoints, and websockets while inheriting from XObjPrototype for consistent validation and behavior.

**Key Architectural Decision**: Each resource type manages its own internal connection pooling, optimizing for its specific needs (e.g., database pools, HTTP connection reuse, file handle management).

## Design Principles

- **Ultrathin abstraction**: Minimal overhead while providing essential functionality
- **Factory pattern**: Creates appropriate connection instances based on resource type
- **Self-contained pooling**: Each resource type manages its own connection pooling
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

### Event Stream Resources

- **WebSocket**: Real-time bidirectional communication
- **Redis Pub/Sub**: Channel-based messaging
- **Kafka**: Distributed event streaming
- **RabbitMQ**: Message queue protocols
- **MQTT**: IoT messaging protocol
- **Server-Sent Events (SSE)**: One-way event streams

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

    # Schema discovery removed - delegated to XInspector
    # See XINSPECTOR.md for schema discovery implementation

    @abstractmethod
    async def list_collections_or_tables(self) -> List[str]:
        """List available collections/tables in the resource"""
        pass

class FileResource(XResource):
    """Base class for file-based resources"""
    file_path: str
    encoding: str = "utf-8"

class DatabaseResource(XResource):
    """Base class for database resources with internal pooling"""
    connection_string: str
    pool_size: int = 10
    _pool: Optional[Any] = None  # Each DB type manages its own pool

class NetworkResource(XResource):
    """Base class for network resources"""
    endpoint: str
    timeout: int = 30
    auth_params: Optional[Dict[str, Any]] = None

class EventStreamResource(XResource):
    """Base class for event streaming resources"""
    connection_params: Dict[str, Any]
    event_handlers: Dict[str, Callable] = Field(default_factory=dict)
    reconnect_interval: int = 5  # seconds
    max_reconnect_attempts: int = 10
    _active_subscriptions: Set[str] = Field(default_factory=set)
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

### Event Stream Resource Examples

```python
# WebSocket Resource
websocket_resource = ResourceFactory.create(
    "websocket",
    name="trading_stream",
    endpoint="wss://stream.example.com/trades",
    auth_params={"token": settings.WS_TOKEN},
    metadata=ResourceMetadata(
        name="trading_stream",
        description="Real-time trading data stream",
        tags=["trading", "realtime", "websocket"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
)

# Set up event handlers
async def on_trade(data):
    print(f"New trade: {data}")

async def on_error(error):
    print(f"Stream error: {error}")

websocket_resource.on("trade", on_trade)
websocket_resource.on("error", on_error)

await websocket_resource.connect()
await websocket_resource.subscribe("BTC/USD")

# Redis Pub/Sub Resource
redis_stream = ResourceFactory.create(
    "redis_pubsub",
    name="event_bus",
    connection_string="redis://localhost:6379",
    metadata=ResourceMetadata(
        name="event_bus",
        description="Application event bus",
        tags=["events", "pubsub", "redis"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
)

await redis_stream.connect()
await redis_stream.subscribe("user_events", lambda msg: print(f"User event: {msg}"))
await redis_stream.publish("user_events", {"type": "login", "user_id": "123"})

# Kafka Resource
kafka_resource = ResourceFactory.create(
    "kafka",
    name="event_stream",
    bootstrap_servers="localhost:9092",
    consumer_group="my_service",
    metadata=ResourceMetadata(
        name="event_stream",
        description="Kafka event stream",
        tags=["kafka", "events", "streaming"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
)

await kafka_resource.connect()
async for event in kafka_resource.consume("user_events"):
    print(f"Kafka event: {event}")
```

## Implementation Details

### MongoDB Resource

```python
class MongoDBResource(DatabaseResource):
    """MongoDB database resource with internal connection pooling"""

    database: str
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None
    
    # Internal pooling configuration
    max_pool_size: int = 100
    min_pool_size: int = 10
    max_idle_time_ms: int = 60000  # 1 minute

    async def connect(self) -> None:
        """Connect to MongoDB with internal pooling"""
        # Motor (AsyncIOMotorClient) manages its own connection pool internally
        self._client = AsyncIOMotorClient(
            self.connection_string,
            maxPoolSize=self.max_pool_size,
            minPoolSize=self.min_pool_size,
            maxIdleTimeMS=self.max_idle_time_ms
        )
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

    # Schema discovery removed - use XInspector instead
    # Example usage:
    # inspector = XInspector(mongo_resource)
    # schema = await inspector.discover_schema("users")

    async def list_collections_or_tables(self) -> List[str]:
        """List all collections in the MongoDB database"""
        return await self._db.list_collections()

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if not self._client:
            return {"status": "not_connected"}
        
        # Motor exposes pool stats through the underlying pymongo client
        stats = {
            "max_pool_size": self.max_pool_size,
            "min_pool_size": self.min_pool_size,
            "active_connections": len(self._client.nodes),
            "idle_time_ms": self.max_idle_time_ms
        }
        return stats
```

### CSV Resource

```python
class CSVResource(FileResource):
    """CSV file resource with efficient file handle management"""

    delimiter: str = ","
    has_header: bool = True
    quotechar: str = '"'
    
    # File handle management
    _file_handle: Optional[Any] = None
    buffer_size: int = 8192  # 8KB buffer for streaming

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

    # Schema discovery removed - use XInspector instead
    # Example usage:
    # inspector = XInspector(csv_resource)
    # schema = await inspector.discover_schema()

    async def list_collections_or_tables(self) -> List[str]:
        """For CSV, return the filename as the single 'table'"""
        return [Path(self.file_path).stem]

    async def read_stream(self, chunk_size: int = 1000) -> AsyncIterator[List[Dict[str, Any]]]:
        """Stream CSV data in chunks for memory efficiency"""
        import csv
        import aiofiles
        
        async with aiofiles.open(self.file_path, mode='r', encoding=self.encoding) as file:
            content = await file.read()
            
        # Use StringIO for CSV parsing
        from io import StringIO
        buffer = StringIO(content)
        reader = csv.DictReader(buffer, delimiter=self.delimiter)
        
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        
        if chunk:  # Yield remaining rows
            yield chunk
```

### WebSocket Event Stream Resource

```python
class WebSocketEventStream(EventStreamResource):
    """WebSocket resource for real-time bidirectional communication"""
    
    endpoint: str
    protocols: List[str] = Field(default_factory=list)
    _websocket: Optional[Any] = None  # websockets.WebSocketClientProtocol
    _background_tasks: Set[asyncio.Task] = Field(default_factory=set)
    
    async def connect(self) -> None:
        """Connect to WebSocket endpoint"""
        import websockets
        
        try:
            self._websocket = await websockets.connect(
                self.endpoint,
                subprotocols=self.protocols,
                **self.connection_params
            )
            self._connection = self._websocket
            
            # Start message handler
            task = asyncio.create_task(self._handle_messages())
            self._background_tasks.add(task)
            
        except Exception as e:
            raise XResourceConnectionError(f"WebSocket connection failed: {e}")
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self._websocket:
                await self._dispatch_event("message", message)
                
                # Parse message type if JSON
                try:
                    data = json.loads(message)
                    if "type" in data:
                        await self._dispatch_event(data["type"], data)
                except json.JSONDecodeError:
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            await self._dispatch_event("close", None)
            await self._handle_reconnect()
    
    async def send(self, data: Union[str, Dict[str, Any]]) -> None:
        """Send data through WebSocket"""
        if isinstance(data, dict):
            data = json.dumps(data)
        await self._websocket.send(data)
    
    async def subscribe(self, channel: str) -> None:
        """Subscribe to a specific channel/topic"""
        await self.send({
            "action": "subscribe",
            "channel": channel
        })
        self._active_subscriptions.add(channel)
    
    def on(self, event: str, handler: Callable) -> None:
        """Register event handler"""
        self.event_handlers[event] = handler
    
    async def _dispatch_event(self, event: str, data: Any) -> None:
        """Dispatch event to registered handlers"""
        if event in self.event_handlers:
            await self.event_handlers[event](data)
```

### Redis Event Stream Resource

```python
class RedisEventStream(EventStreamResource):
    """Redis Pub/Sub resource for message passing"""
    
    connection_string: str
    _redis: Optional[Any] = None  # aioredis.Redis
    _pubsub: Optional[Any] = None
    _subscriber_tasks: Dict[str, asyncio.Task] = Field(default_factory=dict)
    
    async def connect(self) -> None:
        """Connect to Redis"""
        import aioredis
        
        self._redis = await aioredis.from_url(
            self.connection_string,
            **self.connection_params
        )
        self._pubsub = self._redis.pubsub()
        self._connection = self._redis
    
    async def subscribe(self, channel: str, handler: Callable) -> None:
        """Subscribe to a channel with handler"""
        await self._pubsub.subscribe(channel)
        self._active_subscriptions.add(channel)
        
        # Start listening task
        task = asyncio.create_task(self._listen_channel(channel, handler))
        self._subscriber_tasks[channel] = task
    
    async def _listen_channel(self, channel: str, handler: Callable) -> None:
        """Listen for messages on a channel"""
        async for message in self._pubsub.listen():
            if message["type"] == "message":
                await handler(message["data"])
    
    async def publish(self, channel: str, data: Any) -> None:
        """Publish message to channel"""
        if isinstance(data, dict):
            data = json.dumps(data)
        await self._redis.publish(channel, data)
    
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from channel"""
        await self._pubsub.unsubscribe(channel)
        self._active_subscriptions.discard(channel)
        
        # Cancel listener task
        if channel in self._subscriber_tasks:
            self._subscriber_tasks[channel].cancel()
            del self._subscriber_tasks[channel]
```

### Kafka Event Stream Resource

```python
class KafkaEventStream(EventStreamResource):
    """Kafka resource for distributed event streaming"""
    
    bootstrap_servers: str
    consumer_group: Optional[str] = None
    producer_config: Dict[str, Any] = Field(default_factory=dict)
    consumer_config: Dict[str, Any] = Field(default_factory=dict)
    _producer: Optional[Any] = None  # aiokafka.AIOKafkaProducer
    _consumer: Optional[Any] = None  # aiokafka.AIOKafkaConsumer
    
    async def connect(self) -> None:
        """Connect to Kafka cluster"""
        from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
        
        # Create producer
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            **self.producer_config
        )
        await self._producer.start()
        
        # Create consumer if group specified
        if self.consumer_group:
            self._consumer = AIOKafkaConsumer(
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                **self.consumer_config
            )
            await self._consumer.start()
        
        self._connection = self._producer
    
    async def produce(self, topic: str, value: Any, key: Optional[str] = None) -> None:
        """Produce message to topic"""
        if isinstance(value, dict):
            value = json.dumps(value).encode()
        elif isinstance(value, str):
            value = value.encode()
            
        await self._producer.send(
            topic,
            value=value,
            key=key.encode() if key else None
        )
    
    async def consume(self, topics: Union[str, List[str]]) -> AsyncIterator[Dict[str, Any]]:
        """Consume messages from topics"""
        if isinstance(topics, str):
            topics = [topics]
            
        self._consumer.subscribe(topics)
        
        async for msg in self._consumer:
            yield {
                "topic": msg.topic,
                "partition": msg.partition,
                "offset": msg.offset,
                "key": msg.key.decode() if msg.key else None,
                "value": json.loads(msg.value.decode()),
                "timestamp": msg.timestamp
            }
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka"""
        if self._producer:
            await self._producer.stop()
        if self._consumer:
            await self._consumer.stop()
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

## Connection Pooling Architecture

Each resource type implements its own optimized connection pooling strategy:

### Database Resources
- **MongoDB**: Uses Motor's built-in connection pooling (maxPoolSize, minPoolSize)
- **PostgreSQL**: Leverages asyncpg's connection pool
- **MySQL**: Uses aiomysql's pool implementation

### File Resources
- **CSV/Excel**: Manages file handles with buffer pools for streaming
- **JSON/YAML**: Uses memory-mapped files for large datasets

### Network Resources
- **REST API**: HTTP connection pooling via aiohttp sessions

### Event Stream Resources
- **WebSocket**: Persistent connections with automatic reconnection
- **Redis Pub/Sub**: Connection multiplexing for multiple channels
- **Kafka**: Producer/Consumer connection pools per topic
- **RabbitMQ**: Channel multiplexing over AMQP connections
- **MQTT**: QoS-aware connection management

## Integration with Inspector

The Resource component delegates all schema discovery operations to the Inspector (see [XINSPECTOR.md](./XINSPECTOR.md)). This separation of concerns ensures that Resource focuses purely on connection management while Inspector handles the complex logic of schema inference and data profiling.

### Integration Pattern

```python
# Create resource and connect
mongo_resource = ResourceFactory.create("mongodb", ...)
await mongo_resource.connect()

# Pass resource to Inspector for schema discovery
inspector = XInspector(mongo_resource)
schema = await inspector.discover_schema("users")

# Inspector uses resource's connection internally
model = await inspector.generate_model("users")
```

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

## Architectural Decisions

Based on the architecture review:

1. **Internal Pooling**: Each resource type manages its own connection pooling
2. **No Schema Discovery**: All schema operations delegated to XInspector
3. **Self-Contained**: Resources handle their complete lifecycle internally
4. **Error Handling**: Resource-specific exceptions (XResourceError hierarchy)

## Exception Hierarchy

```python
class XResourceError(Exception):
    """Base exception for resource operations"""
    pass

class XResourceConnectionError(XResourceError):
    """Connection-related errors"""
    pass

class XResourceTimeoutError(XResourceError):
    """Timeout errors"""
    pass

class XResourcePoolExhaustedError(XResourceError):
    """Connection pool exhausted"""
    pass
```

## Future Enhancements

1. **Advanced Pooling**: Dynamic pool sizing based on load
2. **Caching**: Add caching layer for frequently accessed resources
3. **Monitoring**: Resource usage metrics and pool statistics
4. **Transaction Support**: Distributed transaction coordination
5. **Resource Groups**: Manage related resources as a unit

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
