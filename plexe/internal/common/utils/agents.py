"""
This module provides utilities for working with agents defined using the smolagents library.
"""

import yaml
import importlib
from plexe.config import config


def get_prompt_templates(base_template_name: str, override_template_name: str) -> dict:
    """
    Given the name of a smolagents prompt template (the 'base template') and a plexe prompt template
    (the 'overriding template'), this function loads both templates and returns a merged template in which
    all keys from the overriding template overwrite the matching keys in the base template.
    """
    base_template: dict = yaml.safe_load(
        importlib.resources.files("smolagents.prompts").joinpath(base_template_name).read_text()
    )
    override_template: dict = yaml.safe_load(
        str(
            importlib.resources.files("plexe")
            .joinpath("templates/prompts/agent")
            .joinpath(override_template_name)
            .read_text()
        ).replace("{{allowed_packages}}", str(config.code_generation.allowed_packages))
    )

    # Recursively merge two dictionaries to ensure deep merging
    def merge_dicts(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = merge_dicts(base[key], value)
            else:
                base[key] = value
        return base

    return merge_dicts(base_template, override_template)
