import subprocess
import sys


def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError:
        print(f"Failed to run: {command}")
        sys.exit(1)


def main():
    print("Installing dependencies...")
    print("Note: This installs the lightweight version of plexe by default.")
    print("Available installation options:")
    print("  poetry install                    # Default lightweight installation")
    print("  poetry install -E lightweight     # Explicitly install lightweight version")
    print("  poetry install -E all             # Full installation with deep learning support")
    print("  poetry install -E deep-learning   # Only deep learning dependencies")
    run_command("poetry install")

    print("Installing pre-commit hooks...")
    run_command("pre-commit install")

    print("Setup complete!")


if __name__ == "__main__":
    main()
