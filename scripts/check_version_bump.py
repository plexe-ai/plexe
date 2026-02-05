#!/usr/bin/env python3
"""
Pre-commit hook to enforce version bumps for plexe.

This script checks if any files in plexe/ have changed compared to origin/main.
If changes exist, it verifies that the version in pyproject.toml has been properly
incremented.

Exit codes:
    0: Success (no changes or version properly bumped)
    1: Failure (changes detected but version not bumped)
"""

import subprocess
import sys
import tomllib
from pathlib import Path
from packaging import version as pkg_version


def run_command(cmd: list[str]) -> tuple[int, str]:
    """Run a shell command and return the exit code and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout.strip()
    except Exception as e:
        print(f"Error running command {' '.join(cmd)}: {e}", file=sys.stderr)
        return 1, ""


def get_version_from_file(file_path: Path) -> str | None:
    """Extract version from pyproject.toml file."""
    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
        # Try old Poetry format first, then new PEP 621 format
        if "tool" in data and "poetry" in data["tool"] and "version" in data["tool"]["poetry"]:
            return data["tool"]["poetry"]["version"]
        elif "project" in data and "version" in data["project"]:
            return data["project"]["version"]
        else:
            print(f"Error: Could not find version in {file_path}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error reading version from {file_path}: {e}", file=sys.stderr)
        return None


def get_version_from_main() -> str | None:
    """Extract version from pyproject.toml on origin/main."""
    cmd = ["git", "show", "origin/main:pyproject.toml"]
    returncode, output = run_command(cmd)

    if returncode != 0:
        print("Error: Could not read pyproject.toml from origin/main", file=sys.stderr)
        print("Make sure origin/main exists and is up to date (run: git fetch origin main)", file=sys.stderr)
        return None

    try:
        data = tomllib.loads(output)
        # Try old Poetry format first, then new PEP 621 format
        if "tool" in data and "poetry" in data["tool"] and "version" in data["tool"]["poetry"]:
            return data["tool"]["poetry"]["version"]
        elif "project" in data and "version" in data["project"]:
            return data["project"]["version"]
        else:
            print("Error: Could not find version in origin/main pyproject.toml", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error parsing version from origin/main: {e}", file=sys.stderr)
        return None


def has_changes_in_plexe() -> bool:
    """Check if there are any changes in plexe/ vs origin/main."""
    # First, try to fetch origin/main
    print("Fetching origin/main to ensure fresh comparison...")
    run_command(["git", "fetch", "origin", "main"])

    # Get diff between current branch and origin/main
    cmd = ["git", "diff", "--name-only", "origin/main...HEAD"]
    returncode, output = run_command(cmd)

    if returncode != 0:
        print("Error: Could not get diff from origin/main", file=sys.stderr)
        print("Make sure origin/main exists (run: git fetch origin main)", file=sys.stderr)
        raise RuntimeError("Failed to get diff from origin/main")

    # Check if any changed files are in plexe/ directory (excluding auto-generated files)
    changed_files = output.split("\n") if output else []
    relevant_changes = [f for f in changed_files if f.startswith("plexe/") and not f.endswith("CODE_INDEX.md")]

    if relevant_changes:
        print(f"\n🔍 Found {len(relevant_changes)} changed file(s) in plexe/:")
        for f in relevant_changes[:5]:  # Show first 5
            print(f"   - {f}")
        if len(relevant_changes) > 5:
            print(f"   ... and {len(relevant_changes) - 5} more")
        return True

    return False


def compare_versions(current: str, main: str) -> bool:
    """Compare two semantic versions. Returns True if current > main."""
    current_v = pkg_version.parse(current)
    main_v = pkg_version.parse(main)
    return current_v > main_v


def main() -> int:
    """Main entry point for the version check script."""
    print("=" * 60)
    print("Plexe Version Check")
    print("=" * 60)

    # Check if there are changes in plexe/
    try:
        if not has_changes_in_plexe():
            print("\n✅ No relevant changes detected in plexe/")
            print("   Version check skipped.")
            return 0
    except RuntimeError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1

    # Get current version
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print(f"\n❌ Error: {pyproject_path} not found!", file=sys.stderr)
        return 1

    current_version = get_version_from_file(pyproject_path)
    if not current_version:
        print("\n❌ Error: Could not read current version", file=sys.stderr)
        return 1

    # Get main version
    main_version = get_version_from_main()
    if not main_version:
        print("\n❌ Error: Could not read version from origin/main", file=sys.stderr)
        return 1

    print(f"\n📦 Current branch version: {current_version}")
    print(f"📦 Origin/main version:    {main_version}")

    # Compare versions
    if not compare_versions(current_version, main_version):
        print("\n" + "=" * 60)
        print("❌ ERROR: Version must be incremented!")
        print("=" * 60)
        print("\nYou have made changes to plexe/ but the version in")
        print("pyproject.toml has not been bumped.")
        print("\nPlease increment the version in:")
        print(f"  {pyproject_path}")
        print(f"\nCurrent version: {current_version}")
        print(f"Required: > {main_version}")
        print("\nExample version bumps:")
        # Use packaging library to safely parse version and strip pre-release identifiers
        parsed = pkg_version.parse(main_version)
        base_parts = str(parsed.base_version).split(".")
        if len(base_parts) == 3:
            major, minor, patch = base_parts
            print(f"  - Patch: {major}.{minor}.{int(patch) + 1}")
            print(f"  - Minor: {major}.{int(minor) + 1}.0")
            print(f"  - Major: {int(major) + 1}.0.0")
        print("=" * 60)
        return 1

    print("\n✅ Version properly incremented!")
    print(f"   {main_version} → {current_version}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
