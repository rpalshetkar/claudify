# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🏆 Golden Standards Repository

This is the central standards repository. When working on ANY project, refer to the appropriate standards document from this folder based on the project type.

## 📚 Quick Reference Guide

### By Technology Stack
- **Python Projects** → [PYTHON_STANDARDS.md](./PYTHON_STANDARDS.md) + [PYTHON_STACK.md](./PYTHON_STACK.md) + [PYTHON_SNIPPETS.md](./PYTHON_SNIPPETS.md)
- **Next.js/React Projects** → [NEXTJS_STANDARDS.md](./NEXTJS_STANDARDS.md) + [NEXTJS_STACK.md](./NEXTJS_STACK.md)
- **Build Automation** → [BUILD_AUTOMATION.md](./BUILD_AUTOMATION.md)
- **Infrastructure Setup** → [INFRA_STACK.md](./INFRA_STACK.md)
- **UX/UI Design** → [UX_PRINCIPLES.md](./UX_PRINCIPLES.md)
- **Custom Commands** → [CLAUDE_COMMANDS.md](./CLAUDE_COMMANDS.md)

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
- Commands: See CLAUDE_COMMANDS.md for /x* commands
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

## Development Commands

```bash
# Install dependencies
uv venv
uv pip install -e .

# Run tests
pytest
pytest -x  # Stop on first failure
pytest -k test_name  # Run specific tests

# Type checking
mypy src

# Linting and formatting
ruff check . --fix
ruff format .
```

Remember: This folder is the source of truth. Always refer here first.