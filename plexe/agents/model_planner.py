import logging

from smolagents import ToolCallingAgent, LiteLLMModel

from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import get_dataset_preview, get_latest_datasets, get_eda_reports
from plexe.tools.schemas import get_model_schemas

logger = logging.getLogger(__name__)


class ModelPlannerAgent:
    """
    Agent responsible for planning ML model solutions based on provided requirements.

    This agent acts as an ML research scientist that develops detailed solution ideas
    and plans for ML use cases. It analyzes the dataset and requirements to propose
    appropriate modeling approaches.

    Attributes:
        verbosity (int): The verbosity level for agent output (0 for quiet, 1 for verbose)
        agent (ToolCallingAgent): The underlying tool-calling agent implementation
    """

    def __init__(
        self,
        model_id: str,
        verbose: bool = False,
        chain_of_thought_callable: callable = None,
    ):
        """
        Initialize the ModelPlannerAgent.

        Args:
            model_id (str): The identifier for the language model to use
            verbose (bool): Whether to enable verbose output from the agent
            chain_of_thought_callable (callable, optional): Callback function for
                intercepting and processing chain-of-thought outputs
        """
        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create solution planner agent - plans ML approaches
        self.agent = ToolCallingAgent(
            name="MLResearchScientist",
            description=(
                "Expert ML researcher that develops detailed solution ideas and plans for ML use cases. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- the ML task definition (i.e. 'intent')"
                "- input schema for the model"
                "- output schema for the model"
                "- the name and comparison method of the metric to optimise"
                "- the name of the dataset to use for training"
            ),
            model=LiteLLMModel(model_id=model_id),
            tools=[
                get_dataset_preview,
                get_latest_datasets,
                get_eda_reports,
                get_model_schemas,
            ],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            prompt_templates=get_prompt_templates("toolcalling_agent.yaml", "mls_prompt_templates.yaml"),
            step_callbacks=[chain_of_thought_callable],
        )
