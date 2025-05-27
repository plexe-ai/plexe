"""
Application entry point for using the plexe package as a conversational agent.
"""

import threading
import time
import webbrowser
import logging

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Launch the Plexe assistant with a web UI."""
    host = "127.0.0.1"
    port = 8000

    # Configure uvicorn to run in a thread
    config = uvicorn.Config("plexe.server:app", host=host, port=port, log_level="info", reload=False)
    server = uvicorn.Server(config)

    # Start server in a background thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Give the server a moment to start
    time.sleep(4)

    # Open the browser
    url = f"http://{host}:{port}"
    logger.info(f"Opening browser at {url}")
    webbrowser.open(url)

    # Keep the main thread alive
    try:
        logger.info("Plexe Assistant is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down Plexe Assistant...")
        server.should_exit = True


if __name__ == "__main__":
    main()
