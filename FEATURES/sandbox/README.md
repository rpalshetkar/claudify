# XArchitecture Sandbox

Test-first development environment for the XArchitecture framework.

## Structure

```
sandbox/
├── mock/           # Mock factories and test doubles
├── fixtures/       # Test data fixtures
├── configs/        # Test configuration files
└── tests/          # Unit and integration tests
    ├── unit/       # Unit tests
    └── integration/ # Integration tests
```

## Setup

1. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```

2. Run tests:
   ```bash
   ./run_tests.py
   ```

## Test Categories

- **Unit Tests** (`pytest -m unit`): Fast, isolated tests
- **Integration Tests** (`pytest -m integration`): Component interaction tests
- **Slow Tests** (`pytest -m slow`): Long-running tests

## Development Workflow

1. **Write Tests First**: Create comprehensive test suites before implementation
2. **Mock Everything**: Use mock factories for external dependencies
3. **Test All Paths**: Cover success, failure, and edge cases
4. **Async Support**: Use pytest-asyncio for async operations

## Base Classes

### `BaseTestCase`
- Common test utilities
- Mock creation and tracking
- Setup/teardown patterns

### `AsyncTestCase`
- Async test support
- Event loop management
- Task cleanup

### `ModelTestCase`
- Model validation testing
- Serialization/deserialization
- Common model patterns

### `DatabaseTestCase`
- Database operation testing
- Connection management
- Transaction handling

### `APITestCase`
- API client testing
- HTTP method mocking
- Response handling

## Mock Factories

### `BaseModelFactory`
- Pydantic model creation
- Factory Boy integration
- Realistic test data

### `MockResourceFactory`
- File, database, API resources
- Connection simulation
- Error condition testing

### `MockInspectorFactory`
- Schema discovery simulation
- Model generation mocking
- Data profiling simulation

## Configuration

### `test_settings.py`
- Database configurations
- API endpoints
- File paths
- Sample data

### `pytest.ini`
- Test discovery
- Markers
- Async configuration
- Logging setup

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# With coverage
pytest --cov=sandbox

# Specific test
pytest sandbox/tests/unit/test_base.py::TestBaseTestCase::test_setup_method

# With ruff checks
./run_tests.py
```

## Code Quality

- **Ruff**: Linting and formatting (autofix enabled)
- **MyPy**: Type checking
- **Pytest**: Testing framework
- **Coverage**: Minimum 80% coverage required

## Test Data

Use `TestDataFactory` for creating:
- Sample records
- CSV/JSON data
- Database schemas
- API responses

## Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One Assertion**: Focus on single behavior
3. **Descriptive Names**: Make test intent clear
4. **Independent Tests**: No test dependencies
5. **Fast Execution**: Keep tests under 30 seconds total

## Fixtures

- `sample_metadata`: Test metadata dictionary
- `sample_namespace`: Test namespace string
- `test_model`: Single test model instance
- `test_models`: List of test model instances
- `temp_dir`: Temporary directory for file tests

## Integration with DEV.md

This sandbox implements Phase 0 of the test-first development plan:

- ✅ Project structure created
- ✅ Test infrastructure setup
- ✅ Base test utilities implemented
- ✅ Mock factories available
- ✅ Configuration management

Next: Implement Phase 1 (XObjPrototype tests) from DEV.md