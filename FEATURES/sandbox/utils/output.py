"""Rich console output utilities for testing."""

from typing import Any

from icecream import ic
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

# Global console instance
console = Console()

# Configure icecream
ic.configureOutput(
    prefix="🍦 |> ",
    includeContext=True,
)


class TestOutput:
    """Rich output helpers for tests."""

    @staticmethod
    def section(title: str, style: str = "bold cyan") -> None:
        """Print a section header."""
        console.print(f"\n[{style}]{title}[/{style}]")
        console.print("─" * len(title), style="cyan")

    @staticmethod
    def success(message: str) -> None:
        """Print success message."""
        console.print(f"[bold green]✅ {message}[/bold green]")

    @staticmethod
    def error(message: str) -> None:
        """Print error message."""
        console.print(f"[bold red]❌ {message}[/bold red]")

    @staticmethod
    def warning(message: str) -> None:
        """Print warning message."""
        console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")

    @staticmethod
    def info(message: str) -> None:
        """Print info message."""
        console.print(f"[bold blue]ℹ️  {message}[/bold blue]")  # noqa: RUF001

    @staticmethod
    def code(code: str, language: str = "python") -> None:
        """Print syntax-highlighted code."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        console.print(syntax)

    @staticmethod
    def panel(content: str, title: str | None = None, style: str = "cyan") -> None:
        """Print content in a panel."""
        console.print(Panel(content, title=title, border_style=style))

    @staticmethod
    def table(data: list[dict[str, Any]], title: str | None = None) -> None:
        """Print data as a table."""
        if not data:
            return

        table = Table(title=title, show_header=True, header_style="bold magenta")

        # Add columns from first row
        for key in data[0]:
            table.add_column(key.replace("_", " ").title())

        # Add rows
        for row in data:
            table.add_row(*[str(v) for v in row.values()])

        console.print(table)

    @staticmethod
    def tree(data: dict[str, Any], title: str = "Tree") -> None:
        """Print data as a tree."""
        tree = Tree(f"[bold]{title}[/bold]")

        def add_branch(node: Tree, data: Any, key: str | None = None) -> None:
            if isinstance(data, dict):
                branch = node.add(f"[cyan]{key}[/cyan]" if key else "")
                for k, v in data.items():
                    add_branch(branch, v, k)
            elif isinstance(data, list):
                branch = node.add(f"[cyan]{key}[/cyan] [dim]({len(data)} items)[/dim]")
                for i, item in enumerate(data):
                    add_branch(branch, item, f"[{i}]")
            else:
                text = f"[cyan]{key}[/cyan]: {data}" if key else str(data)
                node.add(text)

        add_branch(tree, data)
        console.print(tree)

    @staticmethod
    def debug(obj: Any, label: str | None = None) -> None:
        """Debug print with icecream."""
        if label:
            ic(label, obj)
        else:
            ic(obj)


# Export shortcuts
section = TestOutput.section
success = TestOutput.success
error = TestOutput.error
warning = TestOutput.warning
info = TestOutput.info
code = TestOutput.code
panel = TestOutput.panel
table = TestOutput.table
tree = TestOutput.tree
debug = TestOutput.debug
