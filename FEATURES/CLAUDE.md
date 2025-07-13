# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Setup

- Working directory will be only current folder. Don't write files anywhere.
- Prompt for making decisions on stack and lets go through each question and choices one by one
- Read all the python files and documentations and standards from ~/dev/claudify and stick to decisions from there and don't deviate from it.
- You will be running /xinit server command and guide my decisions by planning things here

## Feature List

Remember - For each feature step by step in the planning mode. Be as critical as you can and ask me questions so that we can freeze the spec. Again we need to stick to our design principles around python and architecture and in no case we deviate. In each feature document we need to have usage and test functions covering all functionality

As we plan if there are many steps lets go one step at a time. suggest approaches give pros and cons and usage example and do critically assess the approach and ask for decision. Make it very interactive kind of process

- Remember when multiple decisions are being made go through each step and ask right questions, show sample, pros and cons and run through critical review process with me to help us reach a decision following best arch design principle. ultrathin everything

#### Abstraction - Base Model called XObjPrototype

- Models.md give good indication re what is required here as base construct. XObjPrototype is used to validate against schemas which are pydandic. Namespace support see how this would work for Settings

#### Settings via DynaConf and uses XObjPrototype

- ORDER of loading parameter is configurable
- Defaults are not within python files but .env.default
- Support multiple environments to be loaded (exports/imports use case)
- Always use settings derived from DynaConf

#### Abstraction - Resource

- Do we need a connector or we can use resource. This is connection/resource factory and actual implementaition is in resource.
- file connection, csv, xls, json, yaml
- database connection, mongodb, mysql, postgres, etc.
- rest endpoint connection
- websocket connection
- Metadata about connection params

Eg Mongodb - will have more information populated around collections in db. This has right parameters to point to the right resource eg mongodb uri etc.

#### Abstraction - Repository

With the resource now handy, define repo abstraction, lets start with crud methods, inspect method on collections, stats, describing the schema and it needs to be linked to model capabilities in MODELS.md

ACL, Audit, Stats, Schema Inspection are main areas which would be considered, It would also be having some dynamic read only views like joins or some agreggation available. Linked to Resource also and should be having that metadata

#### Abstraction - Data Xlator/Adapters/Translator/Mutators

#### Abstraction - Data Pipeline

#### Abstraction - Data Inspector/Schema Inspector

#### Namespace

- To enable refering to any object created via XObjPrototype we need a way to refer to it. This is what namespace is for.
- would be managed via caching/redis infra repositories
- Simple get/set by '.' limited path. Disciplined naming

### Registry - Functions/Models/Dynamic Models

## Stack Decisions

### Database

- MongoDB (local development)
- Use motor for async MongoDB driver

### Core Stack (from ~/dev/claudify standards)

- Python 3.12+
- FastAPI for REST APIs
- Pydantic 2.10+ for data validation
- DynaConf for configuration management
- dependency-injector for IoC container
- uv as package manager (NO pip, poetry, pipenv)

### Code Quality

- ruff for linting and formatting
- mypy for static type checking (--strict mode)
- pytest for testing

### Required Security

- Security headers middleware on ALL APIs
- Input validation and sanitization
- Request ID tracking

## Development Commands

```bash
# Package management (use uv, not pip)
uv pip install -r requirements.txt
uv pip install package-name

# Code quality
ruff check .
ruff format .
mypy src/ --strict

# Testing
pytest
pytest -v
pytest --cov=src --cov-report=term-missing

# Run server
uvicorn src.server.main:app --reload --port 8000
```

## Project Structure

```
server/
├── src/
│   └── server/
│       ├── __init__.py
│       ├── main.py
│       ├── api/
│       ├── core/
│       ├── models/
│       └── services/
├── tests/
├── .env.example
├── settings.toml
├── .secrets.toml
├── pyproject.toml
└── ruff.toml
```

## Important Notes

- Follow Python standards from ~/dev/claudify/PYTHON_STANDARDS.md
- Use approved packages from ~/dev/claudify/PYTHON_STACK.md
- Implement security requirements from ~/dev/claudify/PYTHON_MUST.md
- No deviations from established patterns
