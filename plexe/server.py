"""
FastAPI server for the Plexe conversational agent.

This module provides a lightweight WebSocket API for the conversational agent
and serves the assistant-ui frontend for local execution.
"""

import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from plexe.agents.conversational import ConversationalAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="Plexe Assistant", version="1.0.0")

# Serve static files from the ui directory
ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = ui_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"error": "Frontend not found. Please ensure plexe/ui/index.html exists."}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"New WebSocket connection: {session_id}")

    # Create a new agent instance for this session
    agent = ConversationalAgent()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                user_message = message_data.get("content", "")

                # Process the message with the agent
                logger.debug(f"Processing message: {user_message[:100]}...")
                response = agent.agent.run(user_message, reset=False)

                # Send response back to client
                await websocket.send_json({"role": "assistant", "content": response, "id": str(uuid.uuid4())})

            except json.JSONDecodeError:
                # Handle plain text messages for compatibility
                response = agent.agent.run(data, reset=False)
                await websocket.send_json({"role": "assistant", "content": response, "id": str(uuid.uuid4())})

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json(
                    {
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}. Please try again.",
                        "id": str(uuid.uuid4()),
                        "error": True,
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.close()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "plexe-assistant"}
