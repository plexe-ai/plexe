"""
    Provides a way to read configurate file instead of hard code. 
    WARNING: This file has not been intergrated with other files. 
"""

from typing import Dict, List, Type, Optional

import json
from dataclasses import dataclass, field, fields


@dataclass
class ModelProviderJSONConfig:
    default_provider: str
    orchestrator_provider: Optional[str]
    research_provider: Optional[str]
    engineer_provider: Optional[str]
    ops_provider: Optional[str]
    tool_provider: Optional[str]


@dataclass
class ModelJSONConfig:
    provider: ModelProviderJSONConfig
    verbose: bool
    distributed: bool
    working_dir: Optional[str]


class ModelConfigFactory:

    def __init__(self, config_file: str = "./plexe.json"):
        try:
            data = None

            with open(config_file, "r") as f:
                data = json.load(f)

            if not data or len(data) == 0:
                raise Exception(f"Config Error: No data in {config_file}")

            # now the data consist of something
            model_provider = ModelProviderJSONConfig(**data["provider"])

            # check model_provider
            if model_provider.default_provider == "":
                raise Exception("Config Error: Default provider must be provided")

            for field in fields(ModelProviderJSONConfig):
                if getattr(model_provider, field.name) == None:
                    setattr(model_provider, field.name, model_provider.default_provider)

            model_config = ModelJSONConfig(
                provider=model_provider, **{k: v for k, v in data.items() if k != "provider"}
            )

            self.model_config = model_config
        except Exception as e:
            raise Exception(f"Parsing Config Error {e}")
