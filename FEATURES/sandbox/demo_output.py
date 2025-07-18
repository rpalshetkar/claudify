#!/usr/bin/env python3
"""Demo script showing Rich console output capabilities."""

from pathlib import Path

from icecream import ic
from rich.console import Console
from rich.progress import Progress

from sandbox.utils.output import (
    code,
    debug,
    error,
    info,
    panel,
    section,
    success,
    table,
    tree,
    warning,
)

console = Console()


def demo_basic_messages() -> None:
    """Demo basic message types."""
    section("Basic Messages")
    success("Test passed successfully")
    error("Test failed with error")
    warning("Deprecation warning")
    info("Running test suite")


def demo_icecream() -> None:
    """Demo icecream debugging."""
    section("IceCream Debugging")

    test_dict = {"name": "John", "age": 30, "active": True}
    test_list = [1, 2, 3, 4, 5]

    ic(test_dict)
    ic(test_list)

    # Using the debug wrapper
    debug(test_dict, "User data")
    debug(Path.cwd(), "Current directory")


def demo_code_display() -> None:
    """Demo syntax highlighting."""
    section("Code Display")

    sample_code = '''def factorial(n: int) -> int:
    """Calculate factorial recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)'''

    code(sample_code)


def demo_panels() -> None:
    """Demo panel displays."""
    section("Panels")

    panel("This is important information", title="Notice", style="yellow")
    panel(
        "Test Summary:\n✅ 10 passed\n❌ 2 failed\n⏳ 1 skipped",
        title="Results",
        style="green",
    )


def demo_tables() -> None:
    """Demo table display."""
    section("Tables")

    test_results = [
        {"test_name": "test_login", "status": "passed", "duration": "0.23s"},
        {"test_name": "test_logout", "status": "passed", "duration": "0.15s"},
        {"test_name": "test_auth", "status": "failed", "duration": "1.02s"},
        {"test_name": "test_db", "status": "skipped", "duration": "0.00s"},
    ]

    table(test_results, title="Test Results")


def demo_tree() -> None:
    """Demo tree display."""
    section("Tree Display")

    project_structure = {
        "sandbox": {
            "mock": ["base.py", "factories.py"],
            "fixtures": ["models.py", "data.py"],
            "tests": {
                "unit": ["test_base.py", "test_models.py"],
                "integration": ["test_api.py", "test_db.py"],
            },
            "utils": ["output.py", "helpers.py"],
        }
    }

    tree(project_structure, "Project Structure")


def demo_progress() -> None:
    """Demo progress bars."""
    section("Progress Tracking")

    import time

    with Progress() as progress:
        task = progress.add_task("[cyan]Running tests...", total=100)

        while not progress.finished:
            progress.update(task, advance=10)
            time.sleep(0.1)

    success("All tests completed!")


def main() -> None:
    """Run all demos."""
    console.print("[bold cyan]Rich Console Output Demo[/bold cyan]\n")

    demo_basic_messages()
    console.print()

    demo_icecream()
    console.print()

    demo_code_display()
    console.print()

    demo_panels()
    console.print()

    demo_tables()
    console.print()

    demo_tree()
    console.print()

    demo_progress()
    console.print()

    panel(
        "[bold green]Demo completed![/bold green]\n"
        "Rich console and IceCream are now integrated.",
        title="🎉 Success",
        style="green",
    )


if __name__ == "__main__":
    main()
