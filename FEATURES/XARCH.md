# System Architecture

## Architecture Overview

This document provides a comprehensive view of the system architecture, showing the relationships and interactions between all major components.

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

    class XModels {
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
    XModels --> XInspector : uses
    XModels --> XObjPrototype : generates

    class ResourceType {
        <<enumeration>>
        FILE
        DATABASE
        REST_API
        WEBSOCKET
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

    XResource <|-- FileResource : extends
    XResource <|-- DatabaseResource : extends
    XResource <|-- NetworkResource : extends
```

## Data Flow Diagram

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
        XM[XModelsRegistry]
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
    participant XM as XModels
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
| **XModels**       | Model registry           | Dynamic registration, UI mapping, permissions |

### Design Patterns

1. **Abstract Factory**: XResource creates appropriate connection types
2. **Strategy**: Repository implementations for different access patterns
3. **Registry**: XModels maintains central model registry
4. **Delegation**: Components delegate specialized tasks to Inspector
5. **Template Method**: XObjPrototype defines model structure

## Integration Points

### Inspector Integration

- **XResource**: Delegates schema discovery
- **XRepository**: Uses for model generation
- **XModels**: Leverages for dynamic model creation

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
