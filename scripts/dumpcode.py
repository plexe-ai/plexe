"""
This script collects all code files from the project directory and writes it to a single output file.

The purpose of this script is to enable easily passing the entire codebase as context to a language model
with large context window, such as the Google Gemini models.
"""

from pathlib import Path

# === Config ===
EXTENSIONS = {".py", ".md", ".jinja"}
TARGET_DIRS = {"plexe"}
ROOT_FILES = {"README.md"}  # Loose files to include from root
OUTPUT_FILE = "plexe-full-codebase.txt"


def collect_files(base: Path):
    files = []
    for target in TARGET_DIRS:
        target_path = base / target
        if target_path.is_dir():
            files.extend(f for f in target_path.rglob("*") if f.suffix in EXTENSIONS and f.is_file())
    # Include specified root files if they exist and match extension filter
    files.extend(base / f for f in ROOT_FILES if (base / f).is_file() and (base / f).suffix in EXTENSIONS)
    return files


def format_entry(rel_path: Path, content: str) -> str:
    return f"## {rel_path}\n```\n{content}```\n\n\n"


def main():
    base = Path.cwd()
    files = collect_files(base)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write(f"# Full Codebase for {Path.cwd().name}\n\n")
        for file in files:
            rel_path = file.relative_to(base)
            content = file.read_text(encoding="utf-8")
            out.write(format_entry(rel_path, content))


if __name__ == "__main__":
    main()
