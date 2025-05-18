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
    GENERATOR: Literal["simple"] = "simple"
    VALIDATOR: Literal["eda"] = "eda"

    # logging configuration
    LEVEL: str = "INFO"
    FORMAT: str = "[%(asctime)s - %(name)s - %(levelname)s - (%(threadName)-10s)]: - %(message)s"

    # generator configuration
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
    # Additional instructions removed as they were only used by CombinedDataGenerator


config = Config()
