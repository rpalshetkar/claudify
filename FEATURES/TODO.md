# Architecture Refactoring TODO

## Architecture Decision Summary (Updated)

After thorough review and decision-making process, the following architectural decisions have been finalized:

### Core Architecture Decisions

1. **Namespace Implementation**: Extend CacheManager rather than creating separate NamespaceRegistry
2. **Inspector vs Registry**: Clear separation - Inspector generates models, Registry registers them
3. **Repo Factory**: Explicit configuration with smart defaults (optional `materialized` parameter)
4. **Fuzzy Search**: Field-level metadata using Pydantic Field(fuzzy_searchable=True)
5. **XObjPrototype**: Strict Abstract Base Class with runtime enforcement
6. **Audit Storage**: Dedicated audit collection/table with centralized AuditLog model
7. **Resource Pooling**: Internal connection pooling per resource type
8. **Error Handling**: Domain-specific exceptions per component
9. **Metadata Structure**: Flat dictionary (Dict[str, Any]) for maximum flexibility
10. **Registration Order**: Manual registration with explicit dependency ordering

## Critical Assessment Summary

After reviewing the documentation, I've identified significant architectural overlap and complexity that needs to be addressed:

### Key Issues

1. **Scattered Inspection Functionality**
   - Schema discovery split between XResource and Repo
   - Model generation duplicated in REGISTRY.md and REPO.md
   - Statistics and profiling mixed into multiple modules

2. **Overlapping Responsibilities**
   - REGISTRY.md doing too much (UI, permissions, audit, statistics, inspection)
   - Repo pattern overly complex with too many strategies
   - Unclear separation between XObjPrototype models and Dynamic Model Registry

3. **Excessive Abstraction Layers**
   - Too many factory patterns
   - Redundant abstraction levels
   - Complex inheritance hierarchies

## Completed Work

✅ **INSPECTOR.md Created** - Consolidates all inspection functionality:
- Schema discovery from any data source
- Data profiling and statistics
- Model generation from schemas
- Preview and sampling capabilities
- Clear integration points with other modules

✅ **REGISTRY_SIMPLIFIED.md Created** - Focused version that:
- Removes inspection/statistics functionality (moved to Inspector)
- Focuses on core registry responsibilities
- Maintains UI widget detection
- Keeps permissions and audit systems
- Provides clear integration with Inspector

✅ **Inspector vs Registry Boundary** - DECIDED:
- Inspector generates models (analysis phase)
- Registry registers them (runtime phase)
- Clear two-step process maintains separation of concerns

✅ **Repo Factory Auto-Detection** - DECIDED:
- Explicit configuration with smart defaults
- Optional `materialized` parameter allows override
- Auto-detection based on resource.connection_type
- Default: REST/WebSocket → MaterializedRepo, File/DB → ConnectedRepo

✅ **Fuzzy Search Implementation** - DECIDED:
- Field-level metadata using Pydantic Field()
- `fuzzy_searchable=True` parameter on fields
- Self-documenting and type-safe
- Automatic index generation from model introspection

✅ **XObjPrototype Instantiation** - DECIDED:
- Strict Abstract Base Class with ABC
- Runtime enforcement in __init__
- Clear TypeError if instantiated directly
- Must inherit to use

✅ **Audit Storage Strategy** - DECIDED:
- Dedicated audit collection/table
- Centralized AuditLog model inheriting from XObjPrototype
- Separate MaterializedRepo for audit storage
- Enables cross-entity queries and retention policies

✅ **XResource Connection Lifecycle** - DECIDED:
- Internal connection pooling per resource
- Each resource type optimizes its own pooling
- Self-contained resource management
- Consistent interface across all resource types

✅ **Error Handling Philosophy** - DECIDED:
- Domain-specific exceptions per component
- Clear exception hierarchy (XResourceError, XRepoError, etc.)
- Pythonic error handling with try/except
- Good for API error mapping and debugging

✅ **Metadata Storage Structure** - DECIDED:
- Flat dictionary approach (Dict[str, Any])
- Maximum flexibility for different use cases
- Simple key-value pairs
- No schema constraints on metadata

✅ **Component Registration Order** - DECIDED:
- Manual registration in startup sequence
- Explicit dependency order (Settings → Resources → Inspector → Repos → Registry)
- Clear control flow for debugging
- No magic or hidden registration

## Remaining Work

### 1. Update REPO.md
- Remove ModelGenerator class (now in Inspector)
- Simplify to focus on data access patterns only
- Update to use Inspector for any schema-related operations
- Reduce strategy complexity

### 2. Update XRESOURCE.md
- Remove discover_schema method
- Delegate all schema discovery to Inspector
- Focus purely on connection management
- Simplify resource lifecycle

### 3. Update XSETTINGS.md
- Ensure it uses simplified patterns
- Remove any inspection-related configuration
- Focus on settings management only

### 4. Create Architecture Diagram
- Show clear separation of concerns
- Illustrate data flow between modules
- Highlight integration points
- Document dependencies

## Recommended Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Inspector     │     │    Registry     │     │   Repo    │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ Schema Discovery│     │ Model Registry  │     │ Data Access     │
│ Data Profiling  │     │ UI Widgets      │     │ CRUD Operations │
│ Model Generation│     │ Permissions     │     │ Query Building  │
│ Statistics      │     │ Audit Logging   │     │ Transactions    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                        ┌────────┴────────┐
                        │    XResource    │
                        ├─────────────────┤
                        │ Connections     │
                        │ Lifecycle Mgmt  │
                        └─────────────────┘
```

## Design Principles Going Forward

1. **Single Responsibility** - Each module has one clear purpose
2. **Minimal Abstraction** - Only abstract when necessary
3. **Clear Integration** - Well-defined interfaces between modules
4. **Delegation Over Duplication** - Modules delegate to specialists
5. **Ultrathin Design** - Remove unnecessary layers
6. **Explicit Over Implicit** - Manual registration, clear error types, explicit configuration
7. **Flexibility with Defaults** - Smart defaults with override capabilities
8. **Self-Contained Resources** - Resources manage their own lifecycle and pooling

## Next Steps

1. ✅ Finalized 10 core architectural decisions through systematic review
2. Update all component documentation with decided patterns:
   - XOBJPROTOTYPE.md: Add strict ABC pattern, flat metadata
   - XRESOURCE.md: Add internal pooling, remove schema discovery
   - XINSPECTOR.md: Confirm as sole model generator
   - XREPO.md: Add factory with smart defaults, remove ModelGenerator
   - XREGISTRY.md: Update to show it only registers (not generates)
   - XSETTINGS.md: Ensure alignment with decisions
3. Create comprehensive test suite validating all decisions
4. Document standard startup sequence with manual registration
5. Create migration guide from current to new architecture

## Namespace Implementation (HIGH PRIORITY)

### Overview
Implement namespace registration system leveraging CacheManager from XRepo, avoiding separate NameService.

### Design Decisions
1. **Extend CacheManager** - Add namespace registration/resolution capabilities ✅ DECIDED
2. **Dual Index Strategy** - Direct (ns → object) and indirect (repo_key → ns_id) mappings
3. **Cache Backends** - InMemoryRepo and RedisRepo implementations
4. **Namespace Structure**:
   - `ns.models.*` - All registered models
   - `ns.repos.*` - All repo instances
   - `ns.{collection}.*` - Collection-specific namespaces
   - `ns.resources.*` - Resource connections
   - `ns.settings.*` - Configuration objects
   - `ns.fuzzy.{index}.{id}` - Searchable fields concatenated for fuzzy search

### Implementation Tasks
1. ✅ Rename all Repository → Repo across documents
2. ⏳ Extend CacheManager with namespace methods
3. ⏳ Create InMemoryRepo for caching
4. ⏳ Create RedisRepo for caching
5. ⏳ Add indirect indexing support
6. ⏳ Update components to auto-register in namespace
7. ⏳ Create test suite for namespace system

### API Design
```python
# Registration (with new decisions incorporated)
cache.register_ns("ns.models.User", user_model)
cache.register_indirect("user_repo", "ns.repos.user")

# Component initialization order (manual registration)
# 1. Settings
settings = XSettings()
cache.register_ns("ns.settings.app", settings)

# 2. Resources (with internal pooling)
db_resource = DatabaseResource(settings.database)  # Manages own pool
cache.register_ns("ns.resources.db", db_resource)

# 3. Inspector
inspector = XInspector(db_resource)
cache.register_ns("ns.inspector.db", inspector)

# 4. Repos (with smart factory)
user_repo = create_repo(db_resource, materialized=None)  # Auto-detects
cache.register_ns("ns.repos.users", user_repo)

# 5. Registry (registration only)
UserModel = inspector.generate_model("users")  # Inspector generates
model_registry.register(UserModel)  # Registry only registers

# Fuzzy Search Index Registration
# Concatenate all fuzzy searchable fields (defined in model) into single string
# For counterparty with fields: name="ABC CORP", code="ABC", country="USA"
cpty_fuzzy_text = "ABC CORP ABC USA"  # All fuzzy searchable fields concatenated
cache.register_ns("ns.fuzzy.counterparties.cpty123", cpty_fuzzy_text)

# For another counterparty: name="XYZ ABC Bank", code="XYZ", city="London"  
cpty_fuzzy_text2 = "XYZ ABC Bank XYZ London"
cache.register_ns("ns.fuzzy.counterparties.cpty456", cpty_fuzzy_text2)

# For users with fuzzy searchable fields
user_fuzzy_text = "john.doe@example.com John Doe Admin Sales"
cache.register_ns("ns.fuzzy.users.user789", user_fuzzy_text)

# Resolution
user_model = cache.get_ns("ns.models.User")
ns_id = cache.get_ns_by_key("user_repo")

# Fuzzy Search - search "ABC" in counterparties
matches = cache.fuzzy_search("ns.fuzzy.counterparties.*", "ABC")
# Returns: ["ns.fuzzy.counterparties.cpty123", "ns.fuzzy.counterparties.cpty456"]
# Both returned because "ABC" appears in both concatenated strings

# Get actual counterparty objects
cpty_ids = [match.split('.')[-1] for match in matches]
counterparties = [cache.get_ns(f"ns.counterparties.{id}") for id in cpty_ids]

# Listing
models = cache.list_ns("ns.models.*")
repos = cache.list_ns("ns.repos.*")
```

### Implementation Guidelines Based on Decisions

1. **Error Handling**: Each component defines its own exception hierarchy
   ```python
   class XResourceError(Exception): pass
   class XResourceConnectionError(XResourceError): pass
   class XResourceTimeoutError(XResourceError): pass
   ```

2. **Metadata Pattern**: Keep it simple with flat dictionaries
   ```python
   user._metadata["created_by"] = "admin"
   user._metadata["tags"] = ["vip", "active"]
   user._metadata["last_sync"] = datetime.now()
   ```

3. **Resource Lifecycle**: Each resource type handles its own pooling
   ```python
   class DatabaseResource(XResource):
       def __init__(self, config):
           self._pool = asyncpg.create_pool(**config)
           self._pool_size = config.get("pool_size", 10)
   ```

4. **Model Generation Flow**: Two-step process
   ```python
   # Step 1: Inspector analyzes and generates
   model_class = inspector.generate_model("users")
   
   # Step 2: Registry registers for runtime use
   model_registry.register(model_class, namespace="ns.models.User")
   ```

### Fuzzy Search Implementation Details
- **Index Structure**: `ns.fuzzy.{entity_type}.{id}` → `{concatenated_searchable_fields}`
  - Entity type: counterparties, users, transactions, etc.
  - Value: All fuzzy searchable fields concatenated as defined in model
- **Model Definition**:
  ```python
  class Counterparty(XObjPrototype):
      name: str = Field(..., fuzzy_searchable=True)
      code: str = Field(..., fuzzy_searchable=True) 
      country: str = Field(..., fuzzy_searchable=True)
      address: str  # Not fuzzy searchable
  ```
- **Search Flow**:
  1. Define fuzzy searchable fields in model with `fuzzy_searchable=True`
  2. On save, concatenate all fuzzy searchable field values
  3. Store in fuzzy namespace: `ns.fuzzy.{entity_type}.{id}`
  4. Search finds all entries where search term appears in concatenated text
- **Benefits**:
  - Single search across multiple fields
  - Model-driven fuzzy field selection
  - Efficient partial text matching
