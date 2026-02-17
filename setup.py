"""
Setup script for the plexe project.

Installs core dependencies and configures pre-commit hooks.
"""

import subprocess
import sys
import shlex


def run_command(command):
    try:
        # shell=False is safer; command should be a list of arguments
        if isinstance(command, str):
            command = shlex.split(command)

        subprocess.run(command, check=True, shell=False)
    except subprocess.CalledProcessError:
        # Command is always a list here due to normalization above
        cmd_str = " ".join(map(str, command))
        print(f"Failed to run: {cmd_str}")
        sys.exit(1)


def main():
    print("Configuring pre-commit hooks...")
    print("(Note: Ensure you've run 'poetry install' first to install dependencies)")
    run_command(["poetry", "run", "pre-commit", "install"])

    print("Setup complete!")


if __name__ == "__main__":
    main()
