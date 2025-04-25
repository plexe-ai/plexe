# Plexe UI

This module provides custom UI implementations for the Plexe platform.

## GradioUI

The `gradio_ui.py` module implements a custom Gradio-based UI for Plexe with the following features:

1. A chat panel on the left where users can interact with the agent
2. A "Chain of Thought" panel on the right that displays the reasoning process of the multi-agent system

### Key Components

- `GradioEmitter`: A chain of thought emitter that captures and displays agent thoughts in the UI
- `PlexeUI`: The main UI class that creates and manages the Gradio interface

### Usage

```python
from plexe.internal.chat_agents import ChatPlexeAgent
from plexe.internal.ui.gradio_ui import PlexeUI

# Initialize the agent
agent = ChatPlexeAgent("openai/gpt-4o-mini")

# Create and launch the UI
ui = PlexeUI(agent.agent)
ui.launch()
```

The UI is designed to be framework-agnostic and can work with any agent that follows common patterns for step callbacks or observation systems.