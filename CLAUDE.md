# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🏆 Golden Standards Repository

This is the central standards repository. When working on ANY project, refer to the appropriate standards document from this folder based on the project type.

## 📚 Quick Reference Guide

### General guidelines

- Choose mermaid diagram than ascii diagrams
- Don't modify the files in ~/dev/claudify without double confirmation from me
- Prefer editing existing files over creating new ones
- Never create documentation files unless explicitly requested

### By Technology Stack

- **Python Projects** → [PYTHON_STANDARDS.md](./PYTHON_STANDARDS.md) + [PYTHON_STACK.md](./PYTHON_STACK.md) + [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md) + [PYTHON_MUST.md](./PYTHON_MUST.md)
- **Next.js/React Projects** → [NEXTJS_STANDARDS.md](./NEXTJS_STANDARDS.md) + [NEXTJS_STACK.md](./NEXTJS_STACK.md)
- **Build Automation** → [BUILD_AUTOMATION.md](./BUILD_AUTOMATION.md)
- **Infrastructure Setup** → [INFRA_STACK.md](./INFRA_STACK.md)
- **UX/UI Design** → [UX_PRINCIPLES.md](./UX_PRINCIPLES.md)
- **Custom Commands** → [CLAUDE_COMMANDS.md](./CLAUDE_COMMANDS.md)
- **Architecture Patterns** → [features/XARCH.md](./features/XARCH.md)

### Stack Detection

Look for these files to determine which standards apply:

- `pyproject.toml` or `requirements.txt` → Python standards
- `package.json` with "next" → Next.js standards
- `package.json` with "react" → Next.js standards (we use Next.js for all React)
- `docker-compose.yml` or infrastructure files → Infrastructure standards
- Any project with UI → UX principles apply

## 🎯 How to Use These Standards

When starting work on any project:

1. **Identify the Stack** - Check the project files to determine technology
2. **Load Standards** - Reference the appropriate standards from this repository
3. **Apply Commands** - Use `/x*` commands defined in CLAUDE_COMMANDS.md
4. **Maintain Compliance** - Ensure all code follows the loaded standards

### Referencing in Other Projects

When creating CLAUDE.md in other projects, include:

```markdown
## Standards Reference

This project follows standards from: /Users/rrp/dev/git/claudify/

- Technology Stack: [Specify which .md files apply]
- Commands: See CLAUDE_COMMANDS.md for /x\* commands
```

## 🔧 Command System

All projects can use the `/x*` command system:

- `/xinit <stack>` - Initialize new project with standards
- `/xfix` - Auto-fix code quality issues
- `/xverify` - Verify against standards
- `/xrefactor` - Analyze and improve code
- `/xsync` - Update dependencies

See [CLAUDE_COMMANDS.md](./CLAUDE_COMMANDS.md) for full command documentation.

## 📋 Standards Priority

When standards conflict:

1. Language/Framework specific standards (PYTHON_STANDARDS, NEXTJS_STANDARDS)
2. Stack specifications (PYTHON_STACK, NEXTJS_STACK)
3. Infrastructure standards (INFRA_STACK)
4. UX principles (UX_PRINCIPLES)

## 🚀 Quick Start

For new projects:

```bash
/xinit [python|nextjs|fastapi|django|react]
```

For existing projects:

```bash
/xverify  # Check compliance
/xfix     # Fix issues
```

## 💻 Common Development Commands

### Python Projects (using uv)
```bash
# Setup
uv venv && uv pip install -e .

# Development
uv run dev        # Start dev server
uv run test       # Run all tests
uv run lint       # Auto-fix linting (ruff check . --fix)
uv run format     # Format code (ruff format .)
uv run typecheck  # Type checking (mypy src)
uv run quality    # All quality checks
uv run check      # Quality + tests

# Testing
pytest -x         # Stop on first failure
pytest -k test_name  # Run specific tests
pytest -n auto    # Run tests in parallel

# Database (if applicable)
uv run db-upgrade # Apply migrations
uv run db-migrate # Create new migration
uv run db-reset   # Reset database
```

### Next.js/React Projects
```bash
# Development
npm run dev       # Start dev server
npm run build     # Production build
npm run lint      # ESLint
npm run type-check # TypeScript checking
npm run test      # Run tests
```

## 🏗️ Architecture Patterns

Key architectural components defined in [features/XARCH.md](./features/XARCH.md):

- **XObjPrototype**: Base model abstraction with validation
- **XResource**: Connection factory for multiple data sources
- **XInspector**: Schema discovery and model generation
- **XRepository**: Data access with ACL and audit
- **XRegistry**: Dynamic model registry with UI mappings

## 🔒 Security Requirements

All projects must implement:
- Mandatory security headers for APIs
- Input validation with Pydantic (Python) or Zod (TypeScript)
- 80% minimum test coverage
- Rate limiting on public endpoints
- Error standardization

## 📝 Important Notes

- This folder is the **source of truth** for all standards
- Always check here first before implementing patterns
- When creating CLAUDE.md in other projects, reference this repository
- Use `/x*` commands for automated workflows

Remember: This folder is the source of truth. Always refer here first.
