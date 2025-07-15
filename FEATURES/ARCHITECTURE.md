# System Architecture

## Overview

This document provides a comprehensive view of the claudify FEATURES architecture, showing the clear separation of concerns, relationships, and interactions between all major components.

## Key Architectural Decisions

Based on the architecture review (see TODO.md):

1. **Strict ABC Pattern**: XObjPrototype uses runtime enforcement to prevent direct instantiation
2. **Clear Separation of Concerns**:
   - **XInspector**: Schema discovery and model generation (analysis phase)
   - **XRegistry**: Model registration and management (runtime phase)
   - **XResource**: Connection management with internal pooling
   - **XRepo**: Data access with smart factory defaults
3. **Model Generation Flow**: Inspector → generates models → Registry registers them
4. **Resource Pooling**: Each resource type manages its own connection pooling internally
5. **Flat Metadata**: Simple Dict[str, Any] for maximum flexibility

## High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "Configuration Layer"
        XSettings[XSettings<br/>- DynaConf Integration<br/>- Environment Management<br/>- Validation]
    end
    
    subgraph "Base Layer"
        XObjPrototype[XObjPrototype<br/>- Abstract Base Class<br/>- Validation Framework<br/>- Metadata Storage<br/>- Namespace Support]
    end
    
    subgraph "Connection Layer"
        XResource[XResource<br/>- Connection Management<br/>- Resource Lifecycle<br/>- Internal Pooling<br/>- No Schema Discovery]
    end
    
    subgraph "Analysis Layer"
        XInspector[XInspector<br/>- Schema Discovery<br/>- Data Profiling<br/>- Model Generation<br/>- Statistics & Sampling]
    end
    
    subgraph "Registry Layer"
        XRegistry[XRegistry<br/>- Model Registry<br/>- UI Widget Detection<br/>- Permissions<br/>- Audit Logging]
    end
    
    subgraph "Data Access Layer"
        XRepo[XRepo<br/>- CRUD Operations<br/>- Query Building<br/>- Transactions<br/>- Multi-Source Support]
    end
    
    subgraph "Namespace Layer"
        CacheManager[CacheManager<br/>- Namespace Registration<br/>- Object Resolution<br/>- Fuzzy Search<br/>- Indirect Mapping]
    end
    
    %% Dependencies
    XSettings --> XObjPrototype
    XResource --> XObjPrototype
    XInspector --> XResource
    XRepo --> XResource
    XRepo --> XObjPrototype
    XRegistry --> XObjPrototype
    CacheManager --> XRepo
    
    %% Data Flow
    XInspector -.->|Generates Models| XRegistry
    XRegistry -.->|Registers Models| CacheManager
    XRepo -.->|Uses Models| XRegistry
    XInspector -.->|Analyzes via| XResource
```

## Component Relationships

```mermaid
classDiagram
    class XObjPrototype {
        <<abstract>>
        +ns: str
        +meta: Dict
        +validate()
        +serialize()
        +deserialize()
    }

    class XSettings {
        +environment: str
        +debug: bool
        +database: DatabaseSettings
        +api: APISettings
        +load_config()
        +validate_settings()
    }

    class XResource {
        +resource_type: ResourceType
        +connection_params: Dict
        +metadata: ResourceMetadata
        +connect()
        +disconnect()
        +get_inspector()
    }

    class XInspector {
        +inspect_schema()
        +profile_data()
        +generate_model()
        +get_statistics()
        +discover_relationships()
    }

    class XRepository {
        <<interface>>
        +create()
        +read()
        +update()
        +delete()
        +query()
    }

    class ConnectedRepository {
        +resource: XResource
        +acl: ACLManager
        +audit: AuditLogger
        +execute_query()
        +apply_permissions()
    }

    class MaterializedRepository {
        +data: List[Model]
        +indexes: Dict
        +refresh()
        +sync()
    }

    class XRegistry {
        +registry: Dict[str, Model]
        +ui_mappings: Dict
        +permissions: Dict
        +register_model()
        +get_widget()
        +check_permission()
    }

    class DynaConf {
        <<external>>
        +settings: Dict
        +load()
        +merge()
    }

    XObjPrototype <|-- XSettings : inherits
    XObjPrototype <|-- XResource : inherits
    XObjPrototype <|-- XRepository : inherits
    XRepository <|-- ConnectedRepository : implements
    XRepository <|-- MaterializedRepository : implements

    XSettings --> DynaConf : uses
    XResource --> XInspector : delegates to
    ConnectedRepository --> XResource : uses
    ConnectedRepository --> XInspector : uses
    MaterializedRepository --> XInspector : uses
    XRegistry --> XInspector : uses
    XRegistry --> XObjPrototype : generates

    class ResourceType {
        <<enumeration>>
        FILE
        DATABASE
        REST_API
        EVENT_STREAM
    }
```

## Component Responsibilities

### 1. **XSettings** (Configuration Management)
- **Purpose**: Centralized configuration with environment support
- **Responsibilities**:
  - Load settings from multiple sources (env, files, secrets)
  - Environment-aware configuration
  - Validation of configuration values
  - Integration with DynaConf
- **Does NOT**: Handle schema discovery or model generation

### 2. **XObjPrototype** (Base Model)
- **Purpose**: Abstract base class for all data models
- **Responsibilities**:
  - Pydantic-based validation
  - Metadata management (flat dictionary)
  - Namespace path generation
  - Fuzzy search field marking
- **Does NOT**: Handle connections, persistence, or schema discovery

### 3. **XResource** (Connection Management)
- **Purpose**: Unified abstraction for data connections
- **Responsibilities**:
  - Connection lifecycle management
  - Internal connection pooling
  - Resource-specific optimizations
  - Connection validation
- **Does NOT**: Discover schemas or generate models

### 4. **XInspector** (Schema Analysis)
- **Purpose**: Intelligent schema discovery and model generation
- **Responsibilities**:
  - Schema discovery from any data source
  - Data profiling and statistics
  - Dynamic model generation
  - Preview and sampling capabilities
- **Does NOT**: Register models or handle runtime operations

### 5. **XRegistry** (Model Registry)
- **Purpose**: Runtime model management and metadata
- **Responsibilities**:
  - Model registration (not generation)
  - UI widget type detection
  - Permission management
  - Audit trail configuration
- **Does NOT**: Generate models or discover schemas

### 6. **XRepo** (Data Access)
- **Purpose**: High-level data operations
- **Responsibilities**:
  - CRUD operations
  - Query building and execution
  - Transaction management
  - Multi-source data operations
- **Does NOT**: Discover schemas or generate models

### 7. **CacheManager** (Namespace System)
- **Purpose**: Object registration and resolution
- **Responsibilities**:
  - Namespace registration and lookup
  - Indirect key mapping
  - Fuzzy search capabilities
  - Cache backend management
- **Does NOT**: Generate content, only stores references

## Data Flow Architecture

```mermaid
flowchart TB
    subgraph "Data Sources"
        DS1[CSV/Excel Files]
        DS2[JSON/YAML Files]
        DS3[MongoDB]
        DS4[PostgreSQL]
        DS5[REST APIs]
        DS6[WebSockets]
        DS7[Event Streams]
    end

    subgraph "Resource Layer"
        R1[FileResource]
        R2[DatabaseResource]
        R3[NetworkResource]
        R4[EventStreamResource]
    end

    subgraph "Inspector Layer"
        I1[XInspector]
        I2[Schema Discovery]
        I3[Data Profiling]
        I4[Model Generation]
    end

    subgraph "Repository Layer"
        RP1[XRepoFactory]
        RP2[ConnectedRepo<br/>Live Connection]
        RP3[MaterializedRepo<br/>In-Memory]
    end

    subgraph "Model Registry"
        M1[XRegistry]
        M2[UI Schema]
        M3[Permissions]
        M4[Audit]
    end

    subgraph "Application Layer"
        A1[Business Logic]
        A2[API Endpoints]
        A3[Background Tasks]
    end

    DS1 & DS2 --> R1
    DS3 --> R2
    DS4 --> R2
    DS5 --> R3
    DS6 & DS7 --> R4

    R1 & R2 & R3 & R4 --> I1
    I1 --> I2 & I3
    I2 --> I4
    I4 --> M1

    R1 & R2 --> RP1
    R3 & R4 --> RP1
    RP1 --> RP2
    RP1 --> RP3

    M1 --> M2 & M3 & M4
    RP2 & RP3 --> A1
    M1 --> A1
    A1 --> A2 & A3

    style I4 fill:#f9f,stroke:#333,stroke-width:4px
    style M1 fill:#9ff,stroke:#333,stroke-width:2px
```

## Data Flow Examples

### Model Creation Flow
```mermaid
sequenceDiagram
    participant User
    participant XResource
    participant XInspector
    participant XRegistry
    participant CacheManager
    
    User->>XResource: Create & Connect
    User->>XInspector: Pass Resource
    XInspector->>XResource: Analyze Data
    XInspector->>XInspector: Generate Model
    XInspector-->>User: Return Model Class
    User->>XRegistry: Register Model
    XRegistry->>CacheManager: Store in Namespace
```

### Data Access Flow
```mermaid
sequenceDiagram
    participant User
    participant XRepo
    participant XResource
    participant XRegistry
    participant CacheManager
    
    User->>XRepo: Query Request
    XRepo->>CacheManager: Lookup Model
    CacheManager-->>XRepo: Return Model
    XRepo->>XResource: Execute Query
    XResource-->>XRepo: Return Data
    XRepo-->>User: Return Typed Results
```

## Resource Type Auto-Detection

```mermaid
flowchart LR
    subgraph "XRepoFactory.create()"
        RF[Resource Provided?]
        RF -->|Yes| RT{Resource Type?}
        RF -->|No| DT[Data Provided?]
        
        RT -->|Database/File| CR[ConnectedRepo]
        RT -->|REST/WebSocket/EventStream| MR[MaterializedRepo]
        
        DT -->|Yes| MR
        
        OM[Materialized Override?]
        OM -->|True| MR
        OM -->|False| CR
        
        RT --> OM
    end
    
    style MR fill:#f96,stroke:#333,stroke-width:2px
    style CR fill:#6f9,stroke:#333,stroke-width:2px
```

## Key Design Principles

1. **Single Responsibility**: Each component has one clear purpose
2. **Clear Boundaries**: No overlapping functionality between components
3. **Delegation Pattern**: Components delegate to specialists (e.g., Inspector for schema discovery)
4. **Dependency Direction**: Lower layers don't depend on higher layers
5. **Interface Segregation**: Components expose only necessary methods

## Startup Sequence

```python
# 1. Settings (Configuration)
settings = XSettings()
cache.register_ns("ns.settings.app", settings)

# 2. Resources (Connections)
db_resource = DatabaseResource(settings.database)
cache.register_ns("ns.resources.db", db_resource)

# 3. Inspector (Analysis)
inspector = XInspector(db_resource)
cache.register_ns("ns.inspector.db", inspector)

# 4. Generate Models (via Inspector)
UserModel = await inspector.generate_model("users")

# 5. Register Models (via XRegistry)
model_registry.register(UserModel)
cache.register_ns("ns.models.User", UserModel)

# 6. Create Repos (Data Access)
user_repo = await XRepoFactory.create_from_resource(
    resource=db_resource,
    model_class=UserModel,
    schema=schema  # From Inspector
)
cache.register_ns("ns.repos.users", user_repo)
```

## Integration Points

### XInspector → XRegistry
- Inspector generates model classes
- Registry stores them
- Clear handoff point

### XResource → XInspector
- Resource provides connection
- Inspector uses it for analysis
- Resource knows nothing about schemas

### XRepo → XResource
- Repo uses Resource for data operations
- Resource handles connection details
- Clean separation of concerns

### CacheManager Extension
- Extended from XRepo's cache functionality
- Provides namespace registration
- No separate NameService needed

## Component Interactions

```mermaid
sequenceDiagram
    participant App as Application
    participant XR as XResource
    participant XI as XInspector
    participant XM as XRegistry
    participant XRP as XRepository

    App->>XR: create_resource(config)
    XR->>XR: validate_config()
    XR->>XI: get_inspector(resource)
    XI->>XI: inspect_schema()
    XI->>XI: profile_data()
    XI-->>XR: schema_info
    XR->>XI: generate_model(schema)
    XI-->>XR: XObjPrototype model
    XR->>XM: register_model(model)
    XM->>XM: detect_ui_widgets()
    XM->>XM: setup_permissions()
    XM-->>App: model_registered
```

## Design Patterns

1. **Abstract Factory**: XResource creates appropriate connection types
2. **Strategy**: Repository implementations for different access patterns
3. **Registry**: XRegistry maintains central model registry
4. **Delegation**: Components delegate specialized tasks to Inspector
5. **Template Method**: XObjPrototype defines model structure

## Benefits of This Architecture

1. **Maintainability**: Each component can be modified independently
2. **Testability**: Clear boundaries make unit testing easier
3. **Flexibility**: Components can be swapped or extended
4. **Clarity**: Purpose of each component is immediately clear
5. **Reusability**: Components can be used in different contexts

## Component Responsibilities Summary

| Component         | Primary Responsibility   | Key Features                                  |
| ----------------- | ------------------------ | --------------------------------------------- |
| **XObjPrototype** | Base model abstraction   | Validation, serialization, ns support         |
| **XSettings**     | Configuration management | Environment-aware, DynaConf integration       |
| **XResource**     | Connection factory       | Multi-source support, metadata management     |
| **XInspector**    | Schema discovery         | Profiling, model generation, statistics       |
| **XRepository**   | Data access patterns     | CRUD operations, ACL, audit logging           |
| **XRegistry**     | Model registry           | Dynamic registration, UI mapping, permissions |

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