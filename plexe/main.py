"""
Application entry point for using the plexe package as a conversational agent.
"""

import argparse
from smolagents import GradioUI

from plexe.internal.chat_agents import ChatPlexeAgent


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Plexe conversational agent with Gradio UI.")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model ID to use for the agent (default: openai/gpt-4o-mini)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    """Main function to run the Gradio UI for the Plexe conversational agent."""
    args = parse_arguments()
    agent = ChatPlexeAgent(args.model, verbose=args.verbose)
    GradioUI(agent.agent).launch()


if __name__ == "__main__":
    main()
