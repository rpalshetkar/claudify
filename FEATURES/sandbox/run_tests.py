#!/usr/bin/env python3
"""Test runner script with ruff checks and formatting."""

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    with console.status(f"[bold blue]{description}..."):
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
            if result.stdout:
                console.print(f"[dim]{result.stdout}[/dim]")
            console.print(f"[bold green]✅ {description} passed[/bold green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]❌ {description} failed[/bold red]")
            if e.stdout:
                console.print(Panel(e.stdout, title="STDOUT", style="red"))
            if e.stderr:
                console.print(Panel(e.stderr, title="STDERR", style="red"))
            return False


def main() -> int:
    """Run all checks and tests."""
    console.print(
        Panel.fit(
            "[bold cyan]🚀 XArchitecture Test Suite[/bold cyan]", border_style="cyan"
        )
    )

    # Change to project directory
    project_dir = Path(__file__).parent
    console.print(f"📁 Working directory: [blue]{project_dir}[/blue]")

    # List of checks to run
    venv_bin = project_dir / ".venv" / "bin"
    checks = [
        {
            "cmd": [str(venv_bin / "ruff"), "check", ".", "--fix"],
            "description": "Ruff linting with autofix",
        },
        {
            "cmd": [str(venv_bin / "ruff"), "format", "."],
            "description": "Ruff formatting",
        },
        {
            "cmd": [str(venv_bin / "mypy"), "sandbox/"],
            "description": "Type checking",
        },
        {
            "cmd": [str(venv_bin / "pytest"), "sandbox/tests/", "-v"],
            "description": "Unit tests",
        },
        {
            "cmd": [
                str(venv_bin / "pytest"),
                "sandbox/tests/",
                "--cov=sandbox",
                "--cov-report=term-missing",
            ],
            "description": "Coverage analysis",
        },
    ]

    # Run all checks
    failed_checks = []
    for check in checks:
        if not run_command(check["cmd"], check["description"]):
            failed_checks.append(check["description"])

    # Summary
    console.print()
    if failed_checks:
        # Create failure summary table
        table = Table(title="[bold red]Failed Checks[/bold red]", show_header=False)
        table.add_column("Check", style="red")
        for check in failed_checks:
            table.add_row(f"❌ {check}")
        console.print(table)
        console.print(f"\n[bold red]Total failures: {len(failed_checks)}[/bold red]")
        return 1
    console.print(
        Panel.fit(
            "[bold green]🎉 All checks passed![/bold green]", border_style="green"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
