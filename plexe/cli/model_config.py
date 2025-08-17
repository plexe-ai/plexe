"""
Provides a way to read configurate file instead of hard code.
WARNING: This file has not been intergrated with other files.
"""

from typing import Optional
from plexe.model_builder import ModelBuilder
from plexe.internal.common.provider import ProviderConfig

import json
from dataclasses import dataclass, fields


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

            if not data:
                raise Exception(f"Config Error: No data in {config_file}")

            # now the data consist of something
            model_provider = ModelProviderJSONConfig(**data["provider"])

            # check model_provider
            if model_provider.default_provider == "":
                raise Exception("Config Error: Default provider must be provided")

            for field in fields(ModelProviderJSONConfig):
                if getattr(model_provider, field.name) is None:
                    setattr(model_provider, field.name, model_provider.default_provider)

            model_config = ModelJSONConfig(
                provider=model_provider, **{k: v for k, v in data.items() if k != "provider"}
            )

            self.model_config = model_config
        except Exception as e:
            raise Exception(f"Parsing Config Error {e}")

    def get_model_builder(self) -> ModelBuilder:
        if self.model_config is None:
            raise Exception("No config found")

        model_provider = self.model_config.provider

        _provider_config = ProviderConfig(
            default_provider=model_provider.default_provider,
            orchestrator_provider=model_provider.orchestrator_provider,
            research_provider=model_provider.research_provider,
            engineer_provider=model_provider.engineer_provider,
            ops_provider=model_provider.ops_provider,
            tool_provider=model_provider.tool_provider,
        )

        return ModelBuilder(
            _provider_config, self.model_config.verbose, self.model_config.distributed, self.model_config.working_dir
        )
