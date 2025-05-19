"""
This module defines Emitters for outputting chain of thought information.

The emitters are responsible for formatting and outputting the chain of thought of the agents to output
locations such as the console or a logging system. The emitters can be used in various contexts, such as
logging agent actions, debugging, or providing user feedback during agent execution.
"""

import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, TextIO

logger = logging.getLogger(__name__)


class ChainOfThoughtEmitter(ABC):
    """
    Abstract base class for chain of thought emitters.

    Emitters are responsible for outputting chain of thought
    information in a user-friendly format.
    """

    @abstractmethod
    def emit_thought(self, agent_name: str, message: str) -> None:
        """
        Emit a thought from an agent.

        Args:
            agent_name: The name of the agent emitting the thought
            message: The thought message
        """
        pass


class ConsoleEmitter(ChainOfThoughtEmitter):
    """
    Emitter that outputs chain of thought to the console with rich formatting.
    """

    def __init__(self, output: TextIO = sys.stdout):
        """
        Initialize the console emitter with Rich support.

        Args:
            output: The text IO to write to
        """
        self.output = output

        # Initialize Rich components
        try:
            from rich.console import Console

            # Force colorization even in environments like WSL where auto-detection might fail
            self.console = Console(file=output, force_terminal=True, color_system="auto", highlight=False)
            self.step_count = 0
            self.has_rich = True
        except ImportError:
            # Fall back to basic output if Rich isn't available
            self.has_rich = False

    def emit_thought(self, agent_name: str, message: str) -> None:
        """
        Emit a thought to the console using Rich tree visualization.

        Args:
            agent_name: The name of the agent emitting the thought
            message: The thought message
        """
        if not self.has_rich:
            # Fall back to basic output
            self.output.write(f"[{agent_name}] {message}\n")
            self.output.flush()
            return

        try:
            # Import Rich components for type annotations

            # Track step count for timeline
            self.step_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Create a separate tree for each output item, but track agent names
            # This simplifies the display and avoids issues with nested trees
            agent_color = self._get_agent_color(agent_name)

            # Add agent and step information in a cleaner format
            # Use plain print with styling - make it more compact
            step_info = f"Step {self.step_count}"

            # Add a blank line before new agent name for better readability
            # but only if this isn't the first step
            if self.step_count > 1:
                self.console.print("")

            self.console.print(
                f"[bold {agent_color}]{agent_name}[/bold {agent_color}] · [dim]{step_info} · {timestamp}[/dim]"
            )

            # Process the message based on format
            if message.startswith("💡"):
                # Friendly format with title and summary
                parts = message.split("\n", 1)
                title = parts[0]
                content = parts[1] if len(parts) > 1 else ""

                # Extract emoji if present
                emoji = ""
                if title.startswith("💡"):
                    emoji = "💡 "
                    title = title[2:].strip()

                # Display title and content with proper formatting
                self.console.print(f"[bold green]{emoji}{title}[/bold green]")
                if content:
                    self.console.print(f"[dim]{content}[/dim]")
            else:
                # Standard format - print directly
                self.console.print(f"{message}")

        except Exception as e:
            # Log the error at debug level
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Error in Rich formatting: {str(e)}")

            # Fall back to plain output
            self.output.write(f"[{agent_name}] {message}\n")
            self.output.flush()

    @staticmethod
    def _get_agent_color(agent_name: str) -> str:
        """Get the color for an agent based on its role."""
        agent_colors = {
            "System": "bright_blue",
            "MLResearchScientist": "green",
            "MLEngineer": "yellow",
            "MLOperationsEngineer": "magenta",
            "Orchestrator": "cyan",
            "DatasetAnalyser": "red",
            "SchemaResolver": "orange",
            "DatasetSplitter": "purple",
            # Default color
            "default": "blue",
        }

        # Match partial agent names (e.g. "Engineer" should match "ML Engineer")
        for role, color in agent_colors.items():
            if role in agent_name:
                return color

        return agent_colors["default"]


class LoggingEmitter(ChainOfThoughtEmitter):
    """
    Emitter that outputs chain of thought to the logging system.
    """

    def __init__(self, level: int = logging.INFO):
        """
        Initialize the logging emitter.

        Args:
            level: The logging level to use
        """
        self.logger = logging.getLogger("plexe.chain_of_thought")
        self.level = level

    def emit_thought(self, agent_name: str, message: str) -> None:
        """
        Emit a thought to the logger.

        Args:
            agent_name: The name of the agent emitting the thought
            message: The thought message
        """
        self.logger.log(self.level, f"[{agent_name}] {message}")


class MultiEmitter(ChainOfThoughtEmitter):
    """
    Emitter that outputs chain of thought to multiple emitters.
    """

    def __init__(self, emitters: List[ChainOfThoughtEmitter]):
        """
        Initialize the multi emitter.

        Args:
            emitters: The emitters to output to
        """
        self.emitters = emitters

    def emit_thought(self, agent_name: str, message: str) -> None:
        """
        Emit a thought to all configured emitters.

        Args:
            agent_name: The name of the agent emitting the thought
            message: The thought message
        """
        for emitter in self.emitters:
            emitter.emit_thought(agent_name, message)
