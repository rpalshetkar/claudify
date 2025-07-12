# Claude Slash Commands

All custom commands prefixed with `x` for consistency.

## Project Initialization

### `/xinit <stack>`
Creates new project based on stack type, referring to appropriate standards.

Supported stacks:
- `python` - Uses PYTHON_STANDARDS.md
- `nextjs` - Uses FRONTEND_STANDARDS.md  
- `react` - Uses FRONTEND_STANDARDS.md
- `fastapi` - Uses PYTHON_STANDARDS.md + API patterns
- `django` - Uses PYTHON_STANDARDS.md + Django patterns
- `flutter` - Uses FLUTTER_STANDARDS.md
- `chat` - Uses CHAT_APP_STANDARDS.md
- `ml` - Uses ML_PROJECT_STANDARDS.md

Example:
```
/xinit fastapi
> Project name? myapi
> Create in subfolder? n
✅ Created FastAPI project in current directory following PYTHON_STANDARDS.md

Directory structure:
./
├── src/
│   └── myapi/          # Python package
│       ├── __init__.py
│       └── main.py
├── tests/
├── pyproject.toml
└── README.md
```

**Important**: By default, creates project files in the CURRENT directory. Only creates a subfolder if you answer 'y' to "Create in subfolder?".

### How it works:
1. Detects stack type
2. Loads appropriate standards doc
3. Creates project structure based on standards
4. Applies stack-specific patterns
5. Installs standard dependencies

## Code Quality Commands

### `/xfix`
Fixes all code quality issues based on detected stack:
```
/xfix
🔍 Detected: Python project
🔧 Running ruff check --fix...
🔧 Running ruff format...
🔧 Running mypy --strict...
🧪 Running pytest...
✅ All checks passed!
```

For different stacks:
- Python: ruff, mypy, pytest
- JavaScript/TypeScript: eslint, prettier, jest
- Flutter: flutter analyze, flutter test

### `/xverify`
Checks current project against its stack standards:
```
/xverify
🔍 Detected: Next.js project
✅ ESLint: passed
✅ TypeScript: no errors
✅ Tests: 42 passing
❌ Build: 1 error
📊 Coverage: 87%
```

## Refactoring Commands

### `/xrefactor`
Analyzes codebase against stack-specific standards:
```
/xrefactor
📂 Analyzing Next.js project...
📚 Checking against FRONTEND_STANDARDS.md...
🔍 Reviewing patterns...
📡 Fetching latest Next.js docs from context7...

Suggestions:
1. Upgrade: Next.js 13 → 14 (App Router improvements)
2. Pattern: Move to Server Components in 5 files
3. Performance: Add suspense boundaries
4. Type safety: 3 components missing proper types
5. Structure: Consider extracting shared hooks

Generate refactoring plan? [y/n]
```

### `/xsync`
Updates packages based on detected stack:
```
/xsync
🔍 Detected: Python FastAPI project
📦 Checking updates with uv...
  
Updates available:
  fastapi 0.109.0 → 0.115.0 ⚠️
  pydantic 2.5.0 → 2.10.0
  
Update all compatible? [y/n/selective]
```

## Stack Detection

Automatically detects from:
- `pyproject.toml` → Python
- `package.json` + next → Next.js
- `package.json` + react → React
- `pubspec.yaml` → Flutter
- `requirements.txt` → Python (legacy)
- `Cargo.toml` → Rust
- `go.mod` → Go

## Adding New Stacks

To support new stack `/xinit <newstack>`:
1. Create `<NEWSTACK>_STANDARDS.md` in standards/
2. Define project structure
3. List standard dependencies
4. Specify linting/testing tools

Example for a chat app:
```
/xinit chat
> Frontend framework? [nextjs/react/vue]
> Backend framework? [fastapi/express/django]  
> Database? [postgres/mongo/firebase]
> Create in subfolder? y
✅ Created chat app at ./mychat/ with Next.js + FastAPI
```

## Context-Aware Features

All commands:
1. **Auto-detect stack** from current directory
2. **Load appropriate standards** document
3. **Respect .gitignore** patterns
4. **Work from current directory**
5. **Check context7** for latest docs

## Quick Reference

```bash
/xinit <stack>     # New project of any stack
/xfix              # Fix issues for current stack
/xverify           # Verify against standards
/xrefactor         # Analyze and suggest improvements
/xsync             # Update dependencies safely
```

## Extensibility

Standards location: `/docs/standards/`
- PYTHON_STANDARDS.md
- FRONTEND_STANDARDS.md  
- ML_PROJECT_STANDARDS.md
- CHAT_APP_STANDARDS.md
- (add more as needed)

Each standard defines:
- File structure
- Naming conventions
- Required tools
- Verification commands
- Best practices