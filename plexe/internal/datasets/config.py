"""
This module provides configuration for the data generation service.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Config:
    def __init__(self):
        pass

    # global configuration
    GENERATOR: Literal["simple", "combined"] = "combined"
    VALIDATOR: Literal["eda"] = "eda"

    # logging configuration
    LEVEL: str = "INFO"
    FORMAT: str = "[%(asctime)s - %(name)s - %(levelname)s - (%(threadName)-10s)]: - %(message)s"

    # combined generator configuration
    BATCH_SIZE: int = 50
    MAX_N_LLM_SAMPLES: int = 1000
    BASE_INSTRUCTION = (
        "Provide expert-level data science assistance. Communicate concisely and directly. "
        "When returning data or code, provide only the raw output without explanations. "
        "Keep the problem domain in mind. "
    )
    GENERATOR_INSTRUCTION = (
        "Generate exactly the requested number of samples for a machine learning problem, "
        "adhering to the schema and representing the real-world data distribution. "
        "Output only JSON-formatted text without additional text. "
    )
    FILTER_INSTRUCTION = (
        "Filter samples that violate the problem's constraints, including schema breaches or "
        "logical impossibilities. Return the filtered batch in the same format, possibly with samples removed. "
    )
    DENSITY_ESTIMATOR_INSTRUCTION = (
        "Assign a probability density score to each sample based on its likelihood in the real-world "
        "data distribution. Return the batch with an added column for the density scores. "
    )
    LABELLER_INSTRUCTION = (
        "Assign appropriate labels to each sample based on the problem definition. "
        "Return the batch with an added column named 'output' containing the labels. "
    )
    REVIEWER_INSTRUCTION = (
        "Review the batch and mark samples for removal if they are not representative of the problem. "
        "Provide a concise reason for each removal. Return the batch with added columns 'removal' and "
        "'removal_reason'. A row to remove should have 'removal' set to a boolean true (else false) and 'removal_reason' "
        "should contain the reason for removal. "
    )


config = Config()
