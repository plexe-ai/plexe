"""
Custom Gradio UI for the Plexe chat interface.

This module implements a custom Gradio UI with side-by-side chat and chain of thought panels.
"""

from threading import Lock
from typing import List, Dict, Callable

import gradio as gr

from plexe.internal.common.utils.chain_of_thought.callable import ChainOfThoughtCallable
from plexe.internal.common.utils.chain_of_thought.emitters import ChainOfThoughtEmitter


class GradioEmitter(ChainOfThoughtEmitter):
    """Emitter that captures chain of thought for display in the Gradio UI."""

    def __init__(self):
        """Initialize the Gradio emitter."""
        self.logs = []
        self.lock = Lock()
        self.subscribers = []
        self.emitters = []  # Additional emitters to forward thoughts to

    def emit_thought(self, agent_name: str, message: str) -> None:
        """
        Emit a thought and notify subscribers.
        
        Args:
            agent_name: The name of the agent emitting the thought
            message: The thought message
        """
        with self.lock:
            log_entry = f"[{agent_name}] {message}"
            self.logs.append(log_entry)

            # Notify subscribers
            for callback in self.subscribers:
                callback("\n".join(self.logs))

            # Forward to any additional emitters while still inside the lock
            # Make a copy to avoid any potential modification during iteration
            emitters_copy = list(self.emitters)

        # Now call the external emitters outside the lock to avoid deadlocks
        # but after we've ensured a consistent state
        for emitter in emitters_copy:
            emitter.emit_thought(agent_name, message)

    def subscribe(self, callback: Callable[[str], None]) -> None:
        """
        Add a subscriber to be notified of new logs.
        
        Args:
            callback: Function to call with updated logs
        """
        self.subscribers.append(callback)

    def get_logs(self) -> str:
        """
        Get all logs as a single string.
        
        Returns:
            All logs joined by newlines
        """
        with self.lock:
            return "\n".join(self.logs)

    def clear(self) -> None:
        """Clear all logs."""
        with self.lock:
            self.logs = []

            # Notify subscribers of empty logs
            for callback in self.subscribers:
                callback("")


class PlexeUI:
    """Custom Gradio UI for Plexe with chat interface and chain of thought panel."""

    def __init__(self, agent):
        """
        Initialize the Plexe UI.
        
        Args:
            agent: The agent to use for chat
        """
        self.agent = agent

        # Create a Gradio emitter for chain of thought tracking
        self.emitter = GradioEmitter()
        self.cot_callable = ChainOfThoughtCallable(emitter=self.emitter)

        # These will be set in the create_blocks method
        self.chatbot = None
        self.cot_output = None

        # Hook into the chain of thought tracking system
        # This is important to make it framework-agnostic
        self._setup_chain_of_thought_tracking()

    def _setup_chain_of_thought_tracking(self):
        """
        Set up chain of thought tracking for the agent.
        
        This method uses a framework-agnostic approach to hook into the agent's
        chain of thought system, regardless of the underlying agent framework.
        """
        # Method 1: Direct hook if the agent has a standard interface
        if hasattr(self.agent, "add_observer"):
            self.agent.add_observer(self.cot_callable)
            return

        # Method 2: Hook into runner if available (common pattern)
        if hasattr(self.agent, "runner") and hasattr(self.agent.runner, "add_observer"):
            self.agent.runner.add_observer(self.cot_callable)
            return

        # Method 3: Hook into step_callbacks if available (SmoLAgents pattern)
        if hasattr(self.agent, "step_callbacks"):
            if isinstance(self.agent.step_callbacks, list):
                self.agent.step_callbacks.append(self.cot_callable)
            else:
                self.agent.step_callbacks = [self.cot_callable]
            return

        # Log that we weren't able to hook into the chain of thought system
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "Unable to hook into agent's chain of thought system. "
            "Chain of thought panel may not display real-time updates."
        )

    def create_blocks(self) -> gr.Blocks:
        """
        Create the Gradio blocks for the UI.
        
        Returns:
            Gradio Blocks instance
        """
        with gr.Blocks(theme=gr.themes.Soft(), title="Plexe Chat") as demo:
            with gr.Row():
                gr.Markdown("# Plexe: Build ML Models with Chat")

            with gr.Row(equal_height=True):
                # Column 1: Chat Interface
                with gr.Column(scale=2):  # Make chat wider
                    gr.Markdown("### 💬 Chat")
                    self.chatbot = gr.Chatbot(
                        label="",
                        bubble_full_width=False,
                        height=550,
                        type="messages"  # Fix the deprecation warning
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Message",
                            placeholder="Describe the model you want to build or ask a question...",
                            scale=7  # Make textbox take most width in its row
                        )
                        send_btn = gr.Button("Send", scale=1)

                    # Add clear button for chat
                    with gr.Row():
                        clear_btn = gr.ClearButton([msg_input, self.chatbot], value="Clear Chat")
                        # Only clear the CoT when we have the reference to cot_output

                # Column 2: Chain of Thought
                with gr.Column(scale=1):  # Make CoT narrower
                    gr.Markdown("### 🧠 Agent Chain of Thought")
                    self.cot_output = gr.Textbox(
                        label="",
                        interactive=False,
                        lines=30,  # Adjust height as needed
                        max_lines=30,
                        autoscroll=True
                    )

                    with gr.Row():
                        clear_cot_btn = gr.Button("Clear Chain of Thought")
                        clear_cot_btn.click(self.clear_cot, inputs=None, outputs=[self.cot_output])

            # Now add the callback for the clear chat button after cot_output is defined
            clear_btn.click(self.clear_cot, inputs=None, outputs=[self.cot_output])

            # Subscribe the CoT output to the emitter
            self.emitter.subscribe(self.update_cot)

            # Set up event handlers
            send_btn.click(
                self.handle_message,
                inputs=[msg_input, self.chatbot],
                outputs=[msg_input, self.chatbot, self.cot_output],
                show_progress=True,
            )

            msg_input.submit(
                self.handle_message,
                inputs=[msg_input, self.chatbot],
                outputs=[msg_input, self.chatbot, self.cot_output],
                show_progress=True,
            )

        return demo

    def handle_message(self, message: str, history: List[Dict[str, str]]) -> tuple:
        """
        Handle a user message.
        
        Args:
            message: The user message
            history: The chat history in the message format (list of dicts with 'role' and 'content')
            
        Returns:
            Tuple of (empty message, updated history, updated CoT logs)
        """
        if not message.strip():
            return "", history, self.emitter.get_logs()

        # Add user message to history immediately using the messages format
        # Check if history is None and initialize if needed
        if history is None:
            history = []

        # Add user message
        history.append({"role": "user", "content": message})

        # Get response from agent
        try:
            # Clear CoT logs for this new conversation turn
            self.emitter.clear()

            # Process the message with the agent to get the result
            step_result = self.agent.run(message)

            history.append({"role": "assistant", "content": step_result})

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()

            # Add error response in the messages format
            error_message = f"Error: {str(e)}"
            history.append({"role": "assistant", "content": error_message})

            # Log the error to the chain of thought, but keep it concise
            self.emitter.emit_thought("System", f"Error: {str(e)}")
            # Log just the first few lines of the traceback to avoid overwhelming the UI
            short_traceback = "\n".join(error_traceback.split("\n")[:10])
            if len(error_traceback.split("\n")) > 10:
                short_traceback += "\n... (traceback truncated)"
            self.emitter.emit_thought("System", f"Details: {short_traceback}")
            # Full traceback to console
            print(f"Error in agent: {error_traceback}")

        return "", history, self.emitter.get_logs()

    @staticmethod
    def update_cot(logs: str) -> str:
        """
        Update the CoT output (used as callback).
        
        Args:
            logs: The logs to display
            
        Returns:
            The updated logs
        """
        # Always return the logs - the Gradio components will handle None cases
        return logs

    def clear_cot(self) -> str:
        """
        Clear the chain of thought logs.
        
        Returns:
            Empty string for the UI update
        """
        self.emitter.clear()
        return ""

    def launch(self, **kwargs):
        """
        Launch the Gradio UI.
        
        Args:
            **kwargs: Keyword arguments to pass to gr.Blocks.launch()
            
        Returns:
            Result of launching the UI
        """
        demo = self.create_blocks()
        return demo.queue().launch(**kwargs)
