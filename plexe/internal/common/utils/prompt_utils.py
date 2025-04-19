import json
import logging
from typing import Type

from pydantic import BaseModel

from plexe.internal.common.utils.pydantic_utils import convert_schema_to_type_dict

logger = logging.getLogger(__name__)


def join_task_statement(intent: str, input_schema: Type[BaseModel], output_schema: Type[BaseModel]) -> str:
    """Join the problem statement into a single string."""
    problem_statement: str = (
        "# Problem Statement"
        "\n\n"
        f"{intent}"
        "\n\n"
        "# Expected Model Input Schema"
        "\n\n"
        f"{json.dumps(convert_schema_to_type_dict(input_schema), indent=4, default=str)}"
        "\n\n"
        "# Expected Model Output Schema"
        "\n\n"
        f"{json.dumps(convert_schema_to_type_dict(output_schema), indent=4, default=str)}"
    )
    return problem_statement
