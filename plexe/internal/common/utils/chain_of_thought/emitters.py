"""
This module defines Emitters for outputting chain of thought information.

The emitters are responsible for formatting and outputting the chain of thought of the agents to output
locations such as the console or a logging system. The emitters can be used in various contexts, such as
logging agent actions, debugging, or providing user feedback during agent execution.
"""

import logging
import sys
from abc import ABC, abstractmethod
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
    Emitter that outputs chain of thought to the console.
    """

    def __init__(self, output: TextIO = sys.stdout):
        """
        Initialize the console emitter.
        
        Args:
            output: The text IO to write to
        """
        self.output = output

    def emit_thought(self, agent_name: str, message: str) -> None:
        """
        Emit a thought to the console.
        
        Args:
            agent_name: The name of the agent emitting the thought
            message: The thought message
        """
        self.output.write(f"[{agent_name}] {message}\n")
        self.output.flush()


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
