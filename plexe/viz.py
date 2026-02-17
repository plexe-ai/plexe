"""
Streamlit dashboard for plexe.

Usage:
    python -m source.viz --work-dir ./workdir
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Model-builder-slim dashboard")
    parser.add_argument("--work-dir", type=Path, required=True, help="Working directory containing experiments")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    parser.add_argument("--refresh", type=int, default=2, help="Refresh interval (seconds)")
    args = parser.parse_args()

    if not args.work_dir.exists():
        print(f"Error: Work dir does not exist: {args.work_dir}", file=sys.stderr)
        sys.exit(1)

    # Pass args via environment (streamlit limitation)
    os.environ["MBS_WORK_DIR"] = str(args.work_dir.absolute())
    os.environ["MBS_REFRESH_INTERVAL"] = str(args.refresh)

    print("Model Builder Dashboard")
    print(f"Work dir: {args.work_dir}")
    print(f"URL: http://localhost:{args.port}")
    print(f"Refresh: {args.refresh}s\n")

    # Launch streamlit
    import streamlit.web.bootstrap

    streamlit.web.bootstrap.run(
        f"{Path(__file__).parent}/utils/dashboard/app.py",
        False,
        [f"--server.port={args.port}", "--server.headless=true"],
        {},
    )


if __name__ == "__main__":
    main()
