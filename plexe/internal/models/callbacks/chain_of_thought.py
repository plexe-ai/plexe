"""
Chain of Thought model callback for emitting chain of thought information

This module provides a callback class that captures and formats the chain of thought of the agents. It is
useful for understanding at a glance the steps taken by the model during the building process.
"""

from typing import List

from plexe.callbacks import Callback, BuildStateInfo
from plexe.internal.common.utils.chain_of_thought.callable import ChainOfThoughtCallable
from plexe.internal.common.utils.chain_of_thought.emitters import ConsoleEmitter


class ChainOfThoughtModelCallback(Callback):
    """
    Callback that captures and formats the chain of thought for model building.

    This callback bridges between the Plexe callback system and the
    chain of thought callback system.
    """

    def __init__(self, emitter=None):
        """
        Initialize the chain of thought model callback.

        Args:
            emitter: The emitter to use for chain of thought output
        """

        self.cot_callable = ChainOfThoughtCallable(emitter=emitter or ConsoleEmitter())

    def on_build_start(self, info: BuildStateInfo) -> None:
        """
        Reset the chain of thought at the beginning of the build process.
        """
        self.cot_callable.clear()
        self.cot_callable.emitter.emit_thought("System", f"🚀 Starting model build for intent: {info.intent[:40]}...")

    def on_build_end(self, info: BuildStateInfo) -> None:
        """
        Emit completion message at the end of the build process.
        """
        self.cot_callable.emitter.emit_thought("System", "✅ Model build completed")

    def on_iteration_start(self, info: BuildStateInfo) -> None:
        """
        Emit iteration start message.
        """
        self.cot_callable.emitter.emit_thought("System", f"📊 Starting iteration {info.iteration + 1}")

    def on_iteration_end(self, info: BuildStateInfo) -> None:
        """
        Emit iteration end message with performance metrics.
        """
        if info.node and info.node.performance:
            self.cot_callable.emitter.emit_thought(
                "System",
                f"📋 Iteration {info.iteration + 1} completed: {info.node.performance.name}={info.node.performance.value}",
            )
        else:
            self.cot_callable.emitter.emit_thought(
                "System", f"📋 Iteration {info.iteration + 1} failed: No performance metrics available"
            )

    def get_chain_of_thought_callable(self):
        """
        Get the underlying chain of thought callable.

        Returns:
            The chain of thought callable used by this model callback
        """
        return self.cot_callable

    def get_full_chain_of_thought(self) -> List:
        """
        Get the full chain of thought captured during model building.

        Returns:
            The list of steps in the chain of thought
        """
        return self.cot_callable.get_full_chain_of_thought()
