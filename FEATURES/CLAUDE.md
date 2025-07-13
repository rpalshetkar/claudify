# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the FEATURES module of the claudify project - a sophisticated Python architecture framework with multiple abstraction layers. Currently in documentation phase, awaiting implementation.

## Architecture Abstractions

### Implemented (Documented)
- **XObjPrototype** - Abstract base model using Pydantic (see XOBJPROTOTYPE.md)
- **XSettings** - Configuration via DynaConf inheriting from XObjPrototype (see XSETTINGS.md)
- **XResource** - Connection factory for file/database/REST/websocket sources (see XRESOURCE.md)
- **XRepository** - Data access with ACL, audit, and multi-source operations (see XREPO.md)
- **XInspector** - Schema discovery and data profiling (see XINSPECTOR.md)
- **XModels** - Dynamic model registry with UI mappings and permissions (see XMODELS.md)

### Pending Implementation
- Data Xlator/Adapters/Translator/Mutators
- Data Pipeline
- Namespace system (via CacheManager)
- Function/Model registries

## Development Commands

```bash
# Initialize project structure
/xinit server

# Package management (ALWAYS use uv, NEVER pip/poetry)
uv venv && uv pip install -e .
uv pip install package-name

# Development server
uvicorn src.server.main:app --reload --port 8000

# Code quality (run before commits)
uv run lint       # ruff check . --fix
uv run format     # ruff format .
uv run typecheck  # mypy src --strict
uv run quality    # All quality checks
uv run test       # pytest with coverage

# Database operations
uv run db-upgrade # Apply migrations
uv run db-migrate # Create migration
uv run db-reset   # Reset database
```

## Stack Requirements

### Core (MANDATORY - from ~/dev/claudify standards)
- Python 3.12+
- FastAPI for REST APIs
- Pydantic 2.10+ for validation
- DynaConf for configuration
- dependency-injector for IoC
- MongoDB with motor (async driver)
- uv as package manager

### Code Quality
- ruff for linting/formatting
- mypy with --strict mode
- pytest with 80% minimum coverage
- Security headers on ALL APIs

## Architecture Principles

1. **Ultrathin Design** - Minimal abstractions, maximum clarity
2. **Namespace-Aware** - All objects accessible via dot notation paths
3. **Abstract Base Classes** - Cannot instantiate directly, must inherit
4. **Metadata-Rich** - Every component tracks its own metadata
5. **Audit Everything** - Built-in audit logging with compression

## Implementation Workflow

When implementing features:
1. Read the corresponding .md file in this directory
2. Follow the exact specifications - no deviations
3. Interactive decision process - ask questions, show pros/cons
4. One step at a time - don't rush implementations
5. Test coverage for ALL functionality

## Project Structure (When Implemented)

```
src/
├── core/
│   ├── base/
│   │   ├── xobj.py          # XObjPrototype implementation
│   │   └── settings.py      # XSettings with DynaConf
│   ├── resources/
│   │   └── factory.py       # XResource implementations
│   ├── repositories/
│   │   ├── connected.py     # ConnectedRepo
│   │   └── materialized.py  # MaterializedRepo
│   └── inspector/
│       └── schema.py        # XInspector
├── models/
│   └── registry.py          # XModels dynamic registry
├── server/
│   ├── main.py             # FastAPI app
│   └── middleware/         # Security headers
└── tests/
    └── (mirror src structure)
```

## Critical Rules

1. **NEVER deviate from Python standards** in ~/dev/claudify
2. **ALWAYS use uv**, never pip/poetry/pipenv
3. **Abstract classes cannot be instantiated** - enforce this
4. **Security headers mandatory** on all API endpoints
5. **80% test coverage minimum** - no exceptions
6. **Namespace paths use dots** - maintain discipline

## Custom Commands

- `/xinit server` - Initialize with interactive setup
- `/xfix` - Auto-fix code issues
- `/xverify` - Verify standards compliance
- `/xrefactor` - Analyze and improve
- `/xsync` - Update dependencies

## Implementation Status

Currently in documentation phase. When ready to implement:
1. Run `/xinit server` for interactive setup
2. Implement namespace system first (foundation)
3. Build XObjPrototype base class
4. Implement remaining abstractions in order
5. Create comprehensive test suite

Remember: This is a standards-driven project. Always refer to parent claudify standards and never deviate.

## Communication Guidelines

- Always ask one question at a time and explain to me choices and pros and cons with approaches and usage to help me make decisions