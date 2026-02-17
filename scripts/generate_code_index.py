#!/usr/bin/env python3
"""
Generate code index documentation for the Plexe codebase.

This script creates CODE_INDEX.md files that document the structure and public
interfaces of the plexe package and optionally the test suite. These indexes help
coding agents quickly understand the codebase structure.

Usage:
    python scripts/generate_code_index.py [--include-tests]

Output:
    - plexe/CODE_INDEX.md: Main package code documentation
    - tests/CODE_INDEX.md: Test suite overview (if --include-tests flag is used)
"""

import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# === Configuration ===
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    ".git",
    "dist",
    "build",
    "coverage",
    ".DS_Store",
}


# === Data Models ===
@dataclass
class FunctionInfo:
    """Information about a function."""

    name: str
    signature: str
    docstring: str
    is_async: bool = False


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    docstring: str
    init_signature: str | None = None
    methods: list[FunctionInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    """Code information extracted from a Python module."""

    file_path: Path
    module_docstring: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)


# === Python Code Extraction ===
class PythonCodeExtractor:
    """Extracts public code information from Python files using AST."""

    def extract_module_info(self, file: Path, include_private: bool = False) -> ModuleInfo | None:
        """Extract public code information from a Python module.

        Args:
            file: Path to Python file
            include_private: If True, include private (underscore-prefixed) items
        """
        try:
            content = file.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Extract module docstring
            module_docstring = ast.get_docstring(tree) or "No description"
            if module_docstring:
                module_docstring = module_docstring.strip().split("\n")[0].strip()

            # Extract classes and functions
            classes = []
            functions = []

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    if include_private or not node.name.startswith("_"):
                        classes.append(self._extract_class(node, include_private))
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    if include_private or not node.name.startswith("_"):
                        functions.append(self._extract_function(node))

            return ModuleInfo(
                file_path=file,
                module_docstring=module_docstring,
                classes=classes,
                functions=functions,
            )
        except Exception as e:
            print(f"Warning: Could not parse {file}: {e}")
            return None

    def _extract_class(self, node: ast.ClassDef, include_private: bool = False) -> ClassInfo:
        """Extract information from a class definition."""
        docstring = ast.get_docstring(node) or "No description"
        if docstring:
            docstring = docstring.strip().split("\n")[0].strip()

        # Find __init__ method
        init_signature = None
        methods = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                if item.name == "__init__":
                    init_signature = self._get_signature(item)
                elif include_private or not item.name.startswith("_"):
                    # Public method (or include private if requested)
                    methods.append(self._extract_function(item))

        return ClassInfo(
            name=node.name,
            docstring=docstring,
            init_signature=init_signature,
            methods=methods,
        )

    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
        """Extract information from a function definition."""
        docstring = ast.get_docstring(node) or "No description"
        if docstring:
            docstring = docstring.strip().split("\n")[0].strip()

        signature = self._get_signature(node)
        is_async = isinstance(node, ast.AsyncFunctionDef)

        return FunctionInfo(
            name=node.name,
            signature=signature,
            docstring=docstring,
            is_async=is_async,
        )

    def _get_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Generate function signature string."""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass  # Complex type annotations may fail to unparse; skip gracefully
            args.append(arg_str)

        signature = f"{node.name}({', '.join(args)})"

        if node.returns:
            try:
                signature += f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass  # Complex return annotations may fail to unparse; skip gracefully

        return signature


# === File Collection ===
class FileCollector:
    """Collects Python files from a directory."""

    def __init__(self, exclude_dirs: set[str] = EXCLUDE_DIRS):
        self.exclude_dirs = exclude_dirs

    def collect_python_files(self, base_path: Path, exclude_tests: bool = True) -> list[Path]:
        """Collect all Python files in directory, excluding test files if requested."""
        files = []
        for file in base_path.rglob("*.py"):
            if self._should_include(file, exclude_tests):
                files.append(file)
        return sorted(files)

    def _should_include(self, file: Path, exclude_tests: bool) -> bool:
        """Check if file should be included."""
        # Check if any parent directory is in EXCLUDE_DIRS
        for part in file.parts:
            if part in self.exclude_dirs:
                return False

        # Exclude test files if requested
        if exclude_tests:
            file_name = file.name
            if file_name.startswith("test_") or file_name.endswith("_test.py"):
                return False
            if "tests" in file.parts or "test" in file.parts:
                return False

        return True


# === Index Generator ===
class CodeIndexGenerator:
    """Generates CODE_INDEX.md files."""

    def __init__(self):
        self.extractor = PythonCodeExtractor()
        self.collector = FileCollector()

    def generate_package_index(self, package_path: Path, output_path: Path) -> None:
        """Generate code index for the main package."""
        print(f"\n📚 Generating code index for {package_path.name}/...")

        # Collect Python files (exclude tests)
        files = self.collector.collect_python_files(package_path, exclude_tests=True)
        print(f"   Found {len(files)} Python files")

        # Generate content
        content = self._generate_index_content(
            title=f"Code Index: {package_path.name}",
            description=f"Code structure and public interface documentation for the **{package_path.name}** package.",
            files=files,
            base_path=package_path,
            include_private=False,
        )

        # Write output
        output_path.write_text(content, encoding="utf-8")
        print(f"   ✅ Written to {output_path}")

    def generate_test_index(self, test_path: Path, output_path: Path) -> None:
        """Generate code index for the test suite."""
        print(f"\n🧪 Generating code index for {test_path.name}/...")

        # Collect Python files (include all test files)
        files = self.collector.collect_python_files(test_path, exclude_tests=False)
        print(f"   Found {len(files)} test files")

        # Generate content
        content = self._generate_index_content(
            title=f"Code Index: {test_path.name}",
            description="Test suite structure and test case documentation.",
            files=files,
            base_path=test_path,
            include_private=False,  # Still exclude private helpers
        )

        # Write output
        output_path.write_text(content, encoding="utf-8")
        print(f"   ✅ Written to {output_path}")

    def _generate_index_content(
        self,
        title: str,
        description: str,
        files: list[Path],
        base_path: Path,
        include_private: bool,
    ) -> str:
        """Generate the actual index content."""
        output = [
            f"# {title}",
            "",
            f"> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            description,
            "",
        ]

        # Process each file
        for file in files:
            module_info = self.extractor.extract_module_info(file, include_private=include_private)
            if module_info and (module_info.classes or module_info.functions):
                rel_path = file.relative_to(base_path)
                output.extend(self._format_module(rel_path, module_info))

        if len(output) <= 6:  # Only header, no content
            output.append("*No public code found.*")
            output.append("")

        return "\n".join(output)

    def _format_module(self, rel_path: Path, module_info: ModuleInfo) -> list[str]:
        """Format a module's code documentation."""
        # Start with module heading and description (no blank line between)
        lines = [
            f"## `{rel_path}`",
            f"{module_info.module_docstring}",
            "",
        ]

        # Classes
        if module_info.classes:
            for cls in module_info.classes:
                lines.extend(self._format_class(cls))

        # Functions
        if module_info.functions:
            lines.append("**Functions:**")
            for func in module_info.functions:
                lines.extend(self._format_function(func))
            lines.append("")

        lines.append("---")
        return lines

    def _format_class(self, cls: ClassInfo) -> list[str]:
        """Format a class's documentation."""
        # Compact format: **ClassName** - Description
        lines = [f"**`{cls.name}`** - {cls.docstring}"]

        # Add __init__ as first method if present
        if cls.init_signature:
            lines.append(f"- `{cls.init_signature}`")

        # Add all other methods
        if cls.methods:
            for method in cls.methods:
                async_prefix = "async " if method.is_async else ""
                lines.append(f"- `{async_prefix}{method.signature}` - {method.docstring}")

        lines.append("")
        return lines

    def _format_function(self, func: FunctionInfo) -> list[str]:
        """Format a function's documentation."""
        async_prefix = "async " if func.is_async else ""
        return [f"- `{async_prefix}{func.signature}` - {func.docstring}"]


# === Main ===
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate code index documentation for Plexe")
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Also generate test code index (tests/CODE_INDEX.md)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    generator = CodeIndexGenerator()

    print("🔍 Generating code index documentation for Plexe...")

    # Generate main package index
    package_path = root / "plexe"
    if package_path.exists():
        output_path = package_path / "CODE_INDEX.md"
        generator.generate_package_index(package_path, output_path)
    else:
        print(f"❌ Error: {package_path} not found!")
        return 1

    # Generate test index if requested
    if args.include_tests:
        test_path = root / "tests"
        if test_path.exists():
            output_path = test_path / "CODE_INDEX.md"
            generator.generate_test_index(test_path, output_path)
        else:
            print(f"⚠️  Warning: {test_path} not found, skipping test index")

    print("\n✨ Code index generation complete!")
    print("\n💡 Tip: Add plexe/CODE_INDEX.md to your .gitignore if you want to")
    print("   regenerate it dynamically, or commit it for agent reference.")
    return 0


if __name__ == "__main__":
    exit(main())
