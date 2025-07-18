# Development Plan - Test-First Architecture Implementation

## Overview

This document outlines the test-first development approach for building the XArchitecture framework. We'll create comprehensive test suites with mocks and fixtures before implementing the actual components.

## Phase 0: Foundation Setup

### Task 0.1: Project Structure
- **0.1.1** Create sandbox directory structure
  ```
  sandbox/
  ├── mock/         # Mock factories and test doubles
  ├── fixtures/     # Test data fixtures
  ├── configs/      # Test configuration files
  └── tests/        # Unit test suites
  ```
- **0.1.2** Set up test infrastructure
  - pytest configuration
  - pytest-asyncio for async tests
  - factory_boy for mock generation
  - faker for realistic test data

### Task 0.2: Base Test Utilities
- **0.2.1** Create base test case classes
- **0.2.2** Set up test database connections (SQLite for tests)
- **0.2.3** Create test data generators
- **0.2.4** Configure test logging and debugging

## Phase 1: XObjPrototype - Base Model Tests

### Task 1.1: Mock Factory Design
- **1.1.1** Create `BaseModelFactory` using factory_boy
  - Define field generation strategies
  - Support for optional fields
  - Nested model support
- **1.1.2** Create specialized factories
  - `UserModelFactory`
  - `ProductModelFactory`
  - `ConfigModelFactory`
- **1.1.3** Metadata generation utilities
  - Fuzzy search metadata
  - Field-level validation metadata

### Task 1.2: Fixture Creation
- **1.2.1** Base model fixtures
  - Valid model instances
  - Invalid model instances (for validation testing)
  - Edge case data (nulls, empty strings, max values)
- **1.2.2** Namespace fixtures
  - Nested namespace structures
  - Namespace collision scenarios

### Task 1.3: Unit Test Design
- **1.3.1** Validation tests
  - Field validation (required, optional, constraints)
  - Custom validator tests
  - Nested model validation
- **1.3.2** Serialization tests
  - JSON serialization/deserialization
  - Dict conversion
  - Custom serializer support
- **1.3.3** Metadata tests
  - Metadata storage and retrieval
  - Fuzzy search functionality
  - Field metadata inheritance
- **1.3.4** Namespace tests
  - Namespace generation
  - Namespace conflicts
  - Dynamic namespace creation

## Phase 2: XResource - Connection Management Tests

### Task 2.1: Mock Connection Factories
- **2.1.1** File resource mocks
  - `MockCSVResource`
  - `MockExcelResource`
  - `MockJSONResource`
- **2.1.2** Database resource mocks
  - `MockPostgreSQLResource`
  - `MockMongoDBResource`
  - `MockSQLiteResource`
- **2.1.3** API resource mocks
  - `MockRESTResource`
  - `MockGraphQLResource`
- **2.1.4** Stream resource mocks
  - `MockWebSocketResource`
  - `MockKafkaResource`

### Task 2.2: Connection Fixtures
- **2.2.1** Connection string fixtures
  - Valid connection strings
  - Invalid formats
  - Authentication scenarios
- **2.2.2** Mock data sources
  - Sample CSV files
  - JSON data structures
  - Database schemas
- **2.2.3** Connection pool fixtures
  - Pool configuration
  - Connection limits
  - Timeout scenarios

### Task 2.3: Resource Unit Tests
- **2.3.1** Connection lifecycle tests
  - Connect/disconnect
  - Connection pooling
  - Reconnection logic
- **2.3.2** Resource type detection tests
  - Auto-detection from URLs
  - Manual type specification
  - Invalid resource types
- **2.3.3** Error handling tests
  - Connection failures
  - Timeout handling
  - Invalid credentials

## Phase 3: XInspector - Schema Discovery Tests

### Task 3.1: Mock Schema Generators
- **3.1.1** Schema discovery mocks
  - `MockTableSchemaGenerator`
  - `MockJSONSchemaGenerator`
  - `MockAPISchemaGenerator`
- **3.1.2** Data profiling mocks
  - Statistics generation
  - Data type inference
  - Categorical detection

### Task 3.2: Schema Fixtures
- **3.2.1** Database schema fixtures
  - Simple tables
  - Complex relationships
  - Views and materialized views
- **3.2.2** File schema fixtures
  - CSV headers and data types
  - Nested JSON structures
  - Excel multi-sheet layouts
- **3.2.3** API schema fixtures
  - OpenAPI specifications
  - GraphQL schemas
  - Custom API formats

### Task 3.3: Inspector Unit Tests
- **3.3.1** Schema discovery tests
  - Column detection
  - Data type inference
  - Relationship discovery
- **3.3.2** Model generation tests
  - Pydantic model creation
  - Field validation rules
  - Enum generation for categoricals
- **3.3.3** Preview and sampling tests
  - Data preview generation
  - Sampling strategies
  - Large dataset handling

## Phase 4: XRepository - Data Access Tests

### Task 4.1: Repository Mock Factories
- **4.1.1** Connected repository mocks
  - `MockDatabaseRepository`
  - `MockFileRepository`
- **4.1.2** Materialized repository mocks
  - `MockAPIRepository`
  - `MockCachedRepository`
- **4.1.3** Transaction mocks
  - Transaction begin/commit/rollback
  - Nested transactions

### Task 4.2: CRUD Fixtures
- **4.2.1** Test data sets
  - Create operation data
  - Update scenarios
  - Delete cascades
- **4.2.2** Query fixtures
  - Simple queries
  - Complex filters
  - Joins and aggregations
- **4.2.3** ACL fixtures
  - User permissions
  - Role definitions
  - Access control rules

### Task 4.3: Repository Unit Tests
- **4.3.1** CRUD operation tests
  - Create with validation
  - Read with filters
  - Update partial/full
  - Delete with cascades
- **4.3.2** Query builder tests
  - Filter construction
  - Sorting and pagination
  - Aggregation support
- **4.3.3** Transaction tests
  - ACID compliance
  - Rollback scenarios
  - Concurrent access
- **4.3.4** ACL and audit tests
  - Permission checking
  - Audit trail generation
  - Access denial handling

## Phase 5: XRegistry - Registration System Tests

### Task 5.1: Registry Mock Components
- **5.1.1** Model registration mocks
  - Dynamic model registration
  - Model versioning
- **5.1.2** Function registration mocks
  - Utility function registry
  - Plugin system mocks
- **5.1.3** UI mapping mocks
  - Widget detection
  - Form generation rules

### Task 5.2: Registry Fixtures
- **5.2.1** Model registry fixtures
  - Pre-registered models
  - Namespace hierarchies
  - Version conflicts
- **5.2.2** Function registry fixtures
  - Utility functions
  - Transformation functions
  - Validation functions
- **5.2.3** Permission fixtures
  - User roles
  - Access matrices
  - Inheritance rules

### Task 5.3: Registry Unit Tests
- **5.3.1** Registration tests
  - Model registration/deregistration
  - Function registration
  - Duplicate detection
- **5.3.2** Lookup tests
  - Namespace traversal
  - Fuzzy matching
  - Version resolution
- **5.3.3** Permission tests
  - Access control
  - Role-based permissions
  - Audit trail verification

## Phase 6: Integration Tests

### Task 6.1: Component Integration
- **6.1.1** XResource + XInspector integration
- **6.1.2** XInspector + XRegistry integration
- **6.1.3** XRegistry + XRepository integration
- **6.1.4** Full stack integration

### Task 6.2: End-to-End Scenarios
- **6.2.1** Database table to CRUD API
- **6.2.2** CSV file to queryable repository
- **6.2.3** REST API to cached repository
- **6.2.4** Multi-source data federation

## Development Questions & Decisions

### Q1: Mock Library Selection
**Options:**
1. **factory_boy** - Django-style factories, good Pydantic support
2. **pytest-mock** - Simple mocking, integrates with pytest
3. **faker + custom factories** - Maximum flexibility

**Recommendation:** factory_boy for model factories + faker for data generation

### Q2: Test Database Strategy
**Options:**
1. **SQLite in-memory** - Fast, no cleanup needed
2. **PostgreSQL with transactions** - Real DB features
3. **MongoDB mock** - For document stores

**Recommendation:** SQLite for unit tests, PostgreSQL for integration tests

### Q3: Async Testing Approach
**Options:**
1. **pytest-asyncio** - Standard async testing
2. **pytest-trio** - Better async primitives
3. **asyncio + custom fixtures** - Full control

**Recommendation:** pytest-asyncio for consistency with FastAPI

### Q4: Test Data Organization
**Options:**
1. **Per-component fixtures** - Isolated test data
2. **Shared fixture pool** - Reusable test data
3. **Dynamic generation** - Fresh data each run

**Recommendation:** Hybrid - shared base fixtures + dynamic variations

### Q5: Mock Complexity Levels
**Options:**
1. **Simple stubs** - Return fixed values
2. **Smart mocks** - Behavior simulation
3. **Full fakes** - In-memory implementations

**Recommendation:** Start simple, add complexity as needed

## Next Steps

1. **Review and finalize test strategy**
2. **Set up sandbox structure**
3. **Create base mock factories**
4. **Write first XObjPrototype tests**
5. **Iterate through phases**

## Success Criteria

- [ ] 100% test coverage for public APIs
- [ ] All edge cases documented and tested
- [ ] Mock data realistic and comprehensive
- [ ] Tests run in < 30 seconds
- [ ] Clear test naming and documentation
- [ ] Easy to add new test scenarios