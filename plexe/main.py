"""
Application entry point for using the plexe package as a conversational agent.
"""

from smolagents import GradioUI
from plexe.agents.conversational import ConversationalAgent


def main():
    ui = GradioUI(ConversationalAgent().agent)
    ui.launch()


if __name__ == "__main__":
    main()
