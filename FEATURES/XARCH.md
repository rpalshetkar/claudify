# System Architecture

## Architecture Overview

This document provides a comprehensive view of the system architecture, showing the relationships and interactions between all major components.

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

    class FileResource {
        +file_path: str
        +format: FileFormat
        +read_file()
        +write_file()
    }

    class DatabaseResource {
        +connection_string: str
        +engine: DatabaseEngine
        +execute_query()
        +get_schema()
    }

    class NetworkResource {
        +endpoint: str
        +headers: Dict
        +make_request()
        +stream_data()
    }

    class EventStreamResource {
        +connection_params: Dict
        +event_handlers: Dict
        +subscribe()
        +publish()
        +on()
    }

    XResource <|-- FileResource : extends
    XResource <|-- DatabaseResource : extends
    XResource <|-- NetworkResource : extends
    XResource <|-- EventStreamResource : extends
```

## Data Flow Diagram

```mermaid
flowchart TB
    subgraph "Data Source Layer"
        DS1[MongoDB]
        DS2[PostgreSQL]
        DS3[CSV Files]
        DS4[REST APIs]
        DS5[WebSocket]
        DS6[Redis Pub/Sub]
        DS7[Kafka]
    end

    subgraph "Resource Layer"
        R1[DatabaseResource<br/>Internal Pooling]
        R2[FileResource<br/>Buffer Management]
        R3[NetworkResource<br/>HTTP Pooling]
        R4[EventStreamResource<br/>Persistent Connections]
    end

    subgraph "Inspection Layer"
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
    DS4 --> R3
    DS5 & DS6 & DS7 --> R4

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

## Model Generation Flow

```mermaid
sequenceDiagram
    participant Resource as XResource
    participant Inspector as XInspector
    participant ModelGen as ModelGenerator
    participant Registry as XRegistry
    participant Repo as XRepo

    Resource->>Inspector: Create Inspector(resource)
    Inspector->>Resource: discover_schema()
    Resource-->>Inspector: Schema data
    Inspector->>Inspector: profile_data()
    Inspector->>Inspector: detect_categorical_fields()
    
    Inspector->>ModelGen: generate_model(InspectionResult)
    ModelGen->>ModelGen: _schema_to_fields()
    ModelGen->>ModelGen: Create Pydantic model
    ModelGen-->>Inspector: Generated Model Class
    
    Inspector-->>Registry: Register model
    Registry->>Registry: Generate UI schema
    Registry->>Registry: Setup permissions
    Registry->>Registry: Configure audit
    
    Registry-->>Repo: Model available for use
    Repo->>Repo: Use model for type-safe operations
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
    style CR fill:#69f,stroke:#333,stroke-width:2px
```

```mermaid
flowchart TB
    subgraph "DataSources"
        CSV[CSV Files]
        DB[(Databases)]
        API[REST]
        WS[WebSockets]
    end

    subgraph "ConnectionLayer"
        XR[XResourceFactory]
        FR[FileResource]
        DR[DatabaseResource]
        NR[NetworkResource]
    end

    subgraph "InspectionLayer"
        XI[XInspector]
        SP[SchemaProfiler]
        MG[ModelGenerator]
        DS[DataStatistics]
    end

    subgraph "DataAccessLayer"
        XRP[XRepository]
        CR[ConnectedRepository]
        MR[MaterializedRepository]
        ACL[ACL]
        AL[Audit]
    end

    subgraph "ModelLayer"
        XM[XRegistry]
        DM[DynamicModels]
        UI[UIWidgets]
        PM[Permissions]
    end

    subgraph "ApplicationLayer"
        APP[Application]
        SET[XSettings]
    end

    CSV --> FR
    DB --> DR
    API --> NR
    WS --> NR

    FR --> XR
    DR --> XR
    NR --> XR

    XR --> XI
    XI --> SP
    XI --> MG
    XI --> DS

    XI --> CR
    XI --> MR
    CR --> ACL
    CR --> AL
    MR --> ACL
    MR --> AL

    CR --> XRP
    MR --> XRP

    MG --> DM
    DM --> XM
    XM --> UI
    XM --> PM

    XRP --> APP
    XM --> APP
    SET --> APP
```

## Sequence Diagrams

### Schema Discovery Flow

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

### Data Access Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant XRP as XRepository
    participant XR as XResource
    participant ACL as ACL
    participant AL as Audit
    participant DS as DataSource

    App->>XRP: query(criteria)
    XRP->>ACL: check_permissions(user, action)
    ACL-->>XRP: allowed/denied
    alt Permission Granted
        XRP->>XR: get_connection()
        XR->>DS: execute_query()
        DS-->>XR: raw_data
        XR-->>XRP: result_set
        XRP->>AL: log_access(user, action, data)
        XRP-->>App: XObjPrototype models
    else Permission Denied
        XRP->>AL: log_denial(user, action)
        XRP-->>App: AccessDeniedError
    end
```

## Component Responsibilities

### Core Components

| Component         | Primary Responsibility   | Key Features                                  |
| ----------------- | ------------------------ | --------------------------------------------- |
| **XObjPrototype** | Base model abstraction   | Validation, serialization, ns support         |
| **XSettings**     | Configuration management | Environment-aware, DynaConf integration       |
| **XResource**     | Connection factory       | Multi-source support, metadata management     |
| **XInspector**    | Schema discovery         | Profiling, model generation, statistics       |
| **XRepository**   | Data access patterns     | CRUD operations, ACL, audit logging           |
| **XRegistry**      | Model registry           | Dynamic registration, UI mapping, permissions |

### Design Patterns

1. **Abstract Factory**: XResource creates appropriate connection types
2. **Strategy**: Repository implementations for different access patterns
3. **Registry**: XRegistry maintains central model registry
4. **Delegation**: Components delegate specialized tasks to Inspector
5. **Template Method**: XObjPrototype defines model structure

## Integration Points

### Inspector Integration

- **XResource**: Delegates schema discovery
- **XRepository**: Uses for model generation
- **XRegistry**: Leverages for dynamic model creation

### XObjPrototype Inheritance

- All models inherit validation and ns support
- Ensures consistency across the system
- Provides common interface for all data objects

### Settings Integration

- Central configuration through XSettings
- Environment-specific overrides
- Validation at startup

## Future Considerations

Based on architectural analysis, the following improvements are recommended:

1. **Consolidate Inspection**: Move all schema discovery to XInspector
2. **Simplify Repository**: Reduce complexity in repository patterns
3. **Enhance Caching**: Add caching layer for improved performance
4. **Standardize Errors**: Implement consistent error handling across components
