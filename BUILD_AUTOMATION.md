# Build Automation Standards

Modern build automation for Python projects - local first, cloud ready.

## 🎯 Core Principle: Simple, Modern Tools

**Use pyproject.toml scripts for local dev, CI/CD for deployment.**

Why:
- One config file (pyproject.toml) for all Python needs
- Native Python ecosystem integration
- Simple commands that just work
- Easy transition from local to cloud
- No legacy tool baggage

## 📋 Build Automation Hierarchy

### Local Development (Start Here)
1. **pyproject.toml scripts** - All Python commands
2. **taskfile.yml** - Complex orchestration (if needed)
3. **just** - Modern command runner (alternative to task)

### CI/CD (When Ready to Deploy)
1. **GitHub Actions** - Open source projects
2. **Google Cloud Build** - GCP deployment
3. ~~Makefile~~ - Legacy, avoid

## 🏠 Local Development First

### pyproject.toml Scripts (Recommended)

Add these to your pyproject.toml for instant productivity:

```toml
[tool.uv.scripts]
# Development
dev = "python -m myapp.main --reload"
test = "pytest -v"
test-watch = "pytest-watch"
test-cov = "pytest --cov=src --cov-report=html"

# Quality
lint = "ruff check . --fix"
format = "ruff format ."
typecheck = "mypy src"
quality = ["lint", "format", "typecheck"]

# Database
db-upgrade = "alembic upgrade head"
db-migrate = "alembic revision --autogenerate -m"
db-reset = ["alembic downgrade base", "alembic upgrade head"]

# Build
build = "uv build"
clean = "rm -rf dist/ build/ *.egg-info .coverage htmlcov/ .pytest_cache/"

# Combined commands
check = ["quality", "test"]
ci = ["clean", "check", "build"]
```

Usage:
```bash
# Run any script
uv run dev
uv run test
uv run check  # Runs quality + test
```

### Task Runner (For Complex Workflows)

When pyproject.toml scripts aren't enough:

```yaml
# taskfile.yml
version: '3'

vars:
  APP: myapp
  SRC: src

tasks:
  default:
    cmds:
      - task: help

  help:
    desc: Show available tasks
    cmd: task --list

  install:
    desc: Setup development environment
    cmds:
      - uv venv
      - uv pip install -e ".[dev]"
      - echo "✅ Environment ready!"

  dev:
    desc: Run development server with hot reload
    deps: [install]
    cmds:
      - uv run python -m {{.APP}}.main --reload

  test:
    desc: Run tests with coverage
    cmds:
      - uv run pytest -v --cov={{.SRC}}

  test:unit:
    desc: Run only unit tests
    cmds:
      - uv run pytest tests/unit -v

  test:integration:
    desc: Run only integration tests
    cmds:
      - uv run pytest tests/integration -v

  db:init:
    desc: Initialize database
    cmds:
      - uv run alembic upgrade head
      - uv run python -m {{.APP}}.scripts.seed_db

  watch:
    desc: Watch for changes and run tests
    cmds:
      - uv run pytest-watch

  clean:
    desc: Clean all build artifacts
    cmds:
      - rm -rf dist/ build/ *.egg-info
      - find . -type d -name __pycache__ -exec rm -rf {} +
      - find . -type f -name "*.pyc" -delete
      - rm -rf .coverage htmlcov/ .pytest_cache/
```

### Just (Alternative to Task)

Some prefer `just` for its simplicity:

```makefile
# justfile
set dotenv-load

# Default recipe
default:
    @just --list

# Setup development environment
install:
    uv venv
    uv pip install -e ".[dev]"

# Run development server
dev:
    uv run python -m myapp.main --reload

# Run all tests
test:
    uv run pytest -v

# Run specific test
test-one TEST:
    uv run pytest -k {{TEST}} -v

# Quality checks
lint:
    uv run ruff check . --fix

format:
    uv run ruff format .

typecheck:
    uv run mypy src

# Run all quality checks
quality: lint format typecheck

# Full CI simulation
ci: quality test
    uv build
```

## 🚀 GitHub Actions (When Ready)

### Python CI/CD Template

```yaml
# .github/workflows/ci.yml
name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.12"
  UV_VERSION: "0.5.0"

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: ${{ env.UV_VERSION }}
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          uv venv
          uv pip install -e ".[dev]"
      
      - name: Run quality checks
        run: |
          uv run ruff check .
          uv run ruff format . --check
          uv run mypy src
      
      - name: Run tests
        run: |
          uv run pytest -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run security scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

  deploy:
    needs: [quality, security]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}
      
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: ${{ github.event.repository.name }}
          region: us-central1
          source: .
```

## 🏗️ Google Cloud Build Standard

### cloudbuild.yaml Template

```yaml
# cloudbuild.yaml
steps:
  # Build the container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/${_SERVICE_NAME}', '.']
  
  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/${_SERVICE_NAME}']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - '${_SERVICE_NAME}'
      - '--image=gcr.io/$PROJECT_ID/${_SERVICE_NAME}'
      - '--region=${_REGION}'
      - '--platform=managed'
      - '--allow-unauthenticated'

substitutions:
  _SERVICE_NAME: myapp
  _REGION: us-central1

options:
  logging: CLOUD_LOGGING_ONLY
```


## 🎮 Daily Development Workflow

### Quick Start (pyproject.toml scripts)

```bash
# One-time setup
uv venv
uv pip install -e ".[dev]"

# Daily commands
uv run dev        # Start dev server
uv run test       # Run tests
uv run lint       # Fix linting issues
uv run format     # Format code
uv run check      # Run all checks
```

### With Task Runner

```bash
# See all available tasks
task --list

# Common workflows
task install      # Setup environment
task dev          # Start development
task test         # Run tests
task quality      # Check code quality
task watch        # Auto-run tests on changes
```

### With Just

```bash
# See recipes
just

# Common commands
just install
just dev
just test
just quality
just ci          # Full CI locally
```

### When You're Ready for CI/CD

```bash
# GitHub Actions triggers automatically on:
git push origin main        # Deploy
git push origin feature-x   # Test only
git push --tags            # Release

# Google Cloud Build (when ready):
gcloud builds submit --config=cloudbuild.yaml
```

## 🔐 Local Secrets Management

### Development (.env file)

```bash
# .env (git ignored)
DATABASE_URL=postgresql://localhost/myapp_dev
REDIS_URL=redis://localhost:6379
API_KEY=dev-key-12345
SECRET_KEY=dev-secret-key
```

### Using python-dotenv

```python
# Automatically loaded by dynaconf
# Or manually:
from dotenv import load_dotenv
load_dotenv()
```

### CI/CD Secrets (When Ready)

- **GitHub Actions**: Repository secrets
- **Google Cloud**: Secret Manager
- **Local**: .env files (never commit!)

## ⚡ Local Performance Tips

### Speed Up Testing

```toml
# pyproject.toml
[tool.pytest.ini_options]
# Run tests in parallel
addopts = "-n auto"

# Cache test results
cache_dir = ".pytest_cache"
```

### Fast Dependency Installation

```bash
# Use uv for 10-100x faster installs
uv pip install -e ".[dev]"  # Instead of pip

# Cache dependencies
export UV_CACHE_DIR=~/.cache/uv
```

### Development Server

```python
# Use --reload only in dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "myapp.main:app",
        reload=True,  # Hot reload
        reload_dirs=["src"],  # Watch only src
    )
```

## 🚨 Common Pitfalls

❌ **DON'T:**
- Use Makefiles for Python projects
- Mix build logic with application code
- Hardcode paths or environment values
- Create complex shell scripts
- Forget to add scripts to pyproject.toml

✅ **DO:**
- Start with pyproject.toml scripts
- Keep commands simple and clear
- Use environment variables
- Document your commands
- Test locally before CI/CD

## 📚 Quick Reference

### Choose Your Tool

| Need | Use | Why |
|------|-----|-----|
| Basic Python commands | pyproject.toml scripts | Built-in, simple |
| Complex workflows | taskfile.yml | Cross-platform, powerful |
| Simple automation | just | Clean syntax, fast |
| CI/CD | GitHub Actions | Free for open source |
| Cloud deployment | Google Cloud Build | Native GCP integration |

### Migration Path

1. Start with `pyproject.toml` scripts
2. Add `taskfile.yml` when needed
3. Setup GitHub Actions when ready
4. Deploy to cloud when stable

## 🔗 Integration with Standards

This complements:
- [PYTHON_MUST.md](./PYTHON_MUST.md) - Project structure
- [PYTHON_STANDARDS.md](./PYTHON_STANDARDS.md) - Code quality
- [PYTHON_STACK.md](./PYTHON_STACK.md) - Package choices

### Example: Complete pyproject.toml

```toml
[tool.uv.scripts]
# From PYTHON_MUST.md commands
dev = "python -m myapp.main --reload"
test = "pytest -v"
lint = "ruff check . --fix"
format = "ruff format ."
typecheck = "mypy src"

# Combined workflows
quality = ["lint", "format", "typecheck"]
check = ["quality", "test"]

# From this guide
test-watch = "pytest-watch"
test-cov = "pytest --cov=src --cov-report=html"
clean = "rm -rf dist/ build/ *.egg-info .coverage htmlcov/"
build = "uv build"
ci = ["clean", "check", "build"]
```

Remember: **Start simple, grow as needed.**