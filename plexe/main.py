"""
Application entry point for using the plexe package as a conversational agent.
"""

import argparse

from plexe.internal.chat_agents import ChatPlexeAgent
from plexe.internal.common.utils.chain_of_thought.emitters import ConsoleEmitter
from plexe.internal.ui.gradio_ui import PlexeUI


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
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the Gradio UI on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio UI on")
    parser.add_argument("--share", action="store_true", help="Whether to create a public link for the UI")

    return parser.parse_args()


def main():
    """Main function to run the Gradio UI for the Plexe conversational agent."""
    args = parse_arguments()

    # Initialize the ChatPlexeAgent
    agent = ChatPlexeAgent(args.model, verbose=args.verbose)

    # Initialize a PlexeUI with the agent
    ui = PlexeUI(agent.agent)

    # Add console emitter for terminal output if verbose mode is enabled
    if args.verbose:
        # Create a console emitter and add it to the UI emitter
        console_emitter = ConsoleEmitter()
        ui.emitter.emitters.append(console_emitter)

    # Launch the UI
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
