# Repository Abstraction - Technical Specification

## Overview

The Repository abstraction provides a unified interface for data access operations across different storage backends. It builds upon the XResource layer to provide high-level data operations including CRUD, schema inspection, access control, auditing, statistics, and dynamic views. The Repository pattern abstracts the complexities of different data sources while providing consistent APIs for data manipulation and analysis.

## Design Principles

- **Resource-Agnostic**: Works with any XResource implementation (MongoDB, PostgreSQL, CSV, etc.)
- **Async-First**: All operations are asynchronous for optimal performance
- **Type-Safe**: Leverages Pydantic models and Python type hints throughout
- **Ultrathin**: Minimal overhead while providing essential functionality
- **Composable**: Supports repository composition and aggregation
- **Observable**: Built-in hooks for auditing, metrics, and change tracking
- **Secure**: Fine-grained access control at row and column levels

## Architecture

### Repository Hierarchy

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from pydantic import BaseModel
from src.server.core.xobj_prototype import XObjPrototype
from src.server.core.xresource import XResource

T = TypeVar('T', bound=XObjPrototype)

class RepositoryMetadata(BaseModel):
    """Metadata for repository configuration"""
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

class XRepository(XObjPrototype, Generic[T], ABC):
    """Abstract base repository for all data operations"""
    
    metadata: RepositoryMetadata
    _resource: XResource
    _model_class: Type[T]
    _audit_logger: Optional[Any] = None
    _stats_collector: Optional[Any] = None
    _cache: Optional[Any] = None
    
    def __init__(self, resource: XResource, model_class: Type[T], metadata: RepositoryMetadata):
        super().__init__(metadata=metadata)
        self._resource = resource
        self._model_class = model_class
        self._initialize_components()
    
    def get_namespace(self) -> str:
        """Repository namespace based on model"""
        return f"repositories.{self.metadata.name}"
    
    @abstractmethod
    async def create(self, data: T, user_id: Optional[str] = None) -> T:
        """Create a new record"""
        pass
    
    @abstractmethod
    async def read(self, id: Any, user_id: Optional[str] = None) -> Optional[T]:
        """Read a single record by ID"""
        pass
    
    @abstractmethod
    async def update(self, id: Any, data: Dict[str, Any], user_id: Optional[str] = None) -> Optional[T]:
        """Update a record"""
        pass
    
    @abstractmethod
    async def delete(self, id: Any, soft: bool = True, user_id: Optional[str] = None) -> bool:
        """Delete a record (soft or hard)"""
        pass
    
    @abstractmethod
    async def list(self, filters: Dict[str, Any], skip: int = 0, limit: int = 100, user_id: Optional[str] = None) -> List[T]:
        """List records with filters and pagination"""
        pass
    
    @abstractmethod
    async def count(self, filters: Dict[str, Any], user_id: Optional[str] = None) -> int:
        """Count records matching filters"""
        pass
    
    @abstractmethod
    async def inspect_schema(self) -> Dict[str, Any]:
        """Inspect and return schema information"""
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics"""
        pass
```

## Design Decision Q&A

### Q1: Repository Base Class Design

**Question**: Should Repository inherit from XObjPrototype, and how should it link to XResource?

**Options**:
1. **Option A**: Repository inherits from XObjPrototype (current design)
   - Pros: Consistent with other abstractions, gets validation and namespace support
   - Cons: Repository is not really a "data object prototype"

2. **Option B**: Repository as standalone class that uses XObjPrototype models
   - Pros: Clearer separation of concerns
   - Cons: Loses some integration benefits

3. **Option C**: Repository as a composite that contains both XResource and Model references
   - Pros: Maximum flexibility
   - Cons: More complex initialization

**Recommendation**: I suggest Option A (current design) as it provides consistency with your architecture, but I'd like your input.

### Q2: Resource Linking Strategy

**Question**: Should a Repository support multiple XResources or be 1:1?

**Options**:
1. **Single Resource**: One repository maps to one XResource
   - Pros: Simple, clear ownership
   - Cons: No cross-resource operations

2. **Multiple Resources**: Repository can aggregate multiple XResources
   - Pros: Enables joins, cross-database queries
   - Cons: Complex transaction management

3. **Primary + Secondary**: One primary resource, optional secondary resources
   - Pros: Balance of simplicity and flexibility
   - Cons: Need to define clear semantics

**Recommendation**: Start with Single Resource (Option 1) and extend later if needed. What do you think?
