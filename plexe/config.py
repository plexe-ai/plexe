"""
Configuration for the plexe library.
"""

import importlib
import logging
import warnings
from dataclasses import dataclass, field
from importlib.resources import files
from typing import List
from functools import cached_property
from jinja2 import Environment, FileSystemLoader
import sys
from pathlib import Path

from plexe import templates as template_module


TEMPLATE_DIR = files("plexe").joinpath("templates/prompts")


# configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def is_package_available(package_name: str) -> bool:
    """Check if a Python package is available/installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class _Config:
    @dataclass(frozen=True)
    class _FileStorageConfig:
        model_cache_dir: str = field(default=".smolcache/")
        model_dir: str = field(default="model_files/")

    @dataclass(frozen=True)
    class _LoggingConfig:
        level: str = field(default="INFO")
        format: str = field(default="[%(asctime)s - %(name)s - %(levelname)s - (%(threadName)-10s)]: - %(message)s")

    @dataclass(frozen=True)
    class _ModelSearchConfig:
        initial_nodes: int = field(default=3)
        max_nodes: int = field(default=15)
        max_fixing_attempts_train: int = field(default=3)
        max_fixing_attempts_predict: int = field(default=10)
        max_time_elapsed: int = field(default=600)

    @dataclass(frozen=True)
    class _ExecutionConfig:
        runfile_name: str = field(default="execution_script.py")

    @dataclass(frozen=True)
    class _CodeGenerationConfig:
        # Base ML packages that are always available
        _base_packages: List[str] = field(
            default_factory=lambda: [
                "pandas",
                "numpy",
                "scikit-learn",
                "sklearn",
                "joblib",
                "mlxtend",
                "xgboost",
                "pyarrow",
                "statsmodels",
            ]
        )

        # Deep learning packages that are optional
        _deep_learning_packages: List[str] = field(
            default_factory=lambda: [
                "tensorflow-cpu",
                "torch",
                "transformers",
                "tokenizers",
                "accelerate",
                "safetensors",
            ]
        )

        # Additional standard library modules for agent execution
        _standard_lib_modules: List[str] = field(
            default_factory=lambda: [
                "pathlib",
                "typing",
                "dataclasses",
                "json",
                "io",
                "time",
                "datetime",
                "os",
                "sys",
                "math",
                "random",
                "itertools",
                "collections",
                "functools",
                "operator",
                "re",
                "copy",
                "warnings",
                "logging",
                "importlib",
                "types",
                "plexe",
            ]
        )

        @property
        def allowed_packages(self) -> List[str]:
            """Dynamically determine which packages are available and can be used."""
            available_packages = self._base_packages.copy()

            # Check if deep learning packages are installed and add them if they are
            for package in self._deep_learning_packages:
                if is_package_available(package):
                    available_packages.append(package)

            return available_packages

        @property
        def authorized_agent_imports(self) -> List[str]:
            """Return the combined list of allowed packages and standard library modules for agent execution."""
            # Start with allowed packages
            imports = self.allowed_packages.copy()

            # Add standard library modules
            imports.extend(self._standard_lib_modules)

            return imports

        @property
        def deep_learning_available(self) -> bool:
            """Check if deep learning packages are available."""
            return any(is_package_available(pkg) for pkg in self._deep_learning_packages)

        k_fold_validation: int = field(default=5)

    @dataclass(frozen=True)
    class _DataGenerationConfig:
        pass  # todo: implement

    @dataclass(frozen=True)
    class _RayConfig:
        address: str = field(default=None)  # None for local, Ray address for remote
        num_cpus: int = field(default=None)  # None for auto-detect
        num_gpus: int = field(default=None)  # None for auto-detect

    # configuration objects
    file_storage: _FileStorageConfig = field(default_factory=_FileStorageConfig)
    logging: _LoggingConfig = field(default_factory=_LoggingConfig)
    model_search: _ModelSearchConfig = field(default_factory=_ModelSearchConfig)
    code_generation: _CodeGenerationConfig = field(default_factory=_CodeGenerationConfig)
    execution: _ExecutionConfig = field(default_factory=_ExecutionConfig)
    data_generation: _DataGenerationConfig = field(default_factory=_DataGenerationConfig)
    ray: _RayConfig = field(default_factory=_RayConfig)


@dataclass(frozen=True)
class _CodeTemplates:
    predictor_interface: str = field(
        default=Path(importlib.import_module("plexe.internal.models.interfaces.predictor").__file__).read_text()
    )
    predictor_template: str = field(
        default=files(template_module).joinpath("models").joinpath("predictor.tmpl.py").read_text()
    )


@dataclass(frozen=True)
class _PromptTemplates:
    template_dir: str = field(default=TEMPLATE_DIR)

    @cached_property
    def env(self) -> Environment:
        return Environment(loader=FileSystemLoader(str(self.template_dir)))

    def _render(self, template_name: str, **kwargs) -> str:
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    def planning_system(self) -> str:
        return self._render("planning/system_prompt.jinja")

    def planning_select_metric(self, problem_statement) -> str:
        return self._render("planning/select_metric.jinja", problem_statement=problem_statement)

    def schema_base(self) -> str:
        return self._render("schemas/base.jinja")

    def schema_identify_target(self, columns, intent) -> str:
        return self._render("schemas/identify_target.jinja", columns=columns, intent=intent)

    def schema_generate_from_intent(self, intent, input_schema="input_schema", output_schema="output_schema") -> str:
        return self._render(
            "schemas/generate_from_intent.jinja", intent=intent, input_schema=input_schema, output_schema=output_schema
        )

    def schema_resolver_prompt(
        self, intent, datasets, input_schema=None, output_schema=None, has_input_schema=False, has_output_schema=False
    ) -> str:
        return self._render(
            "agent/schema_resolver_prompt.jinja",
            intent=intent,
            datasets=datasets,
            input_schema=input_schema,
            output_schema=output_schema,
            has_input_schema=has_input_schema,
            has_output_schema=has_output_schema,
        )

    def eda_agent_prompt(self, intent, datasets) -> str:
        return self._render(
            "agent/agent_data_analyser_prompt.jinja",
            intent=intent,
            datasets=datasets,
        )

    def training_system(self) -> str:
        return self._render("training/system_prompt.jinja")

    def training_generate(
        self, problem_statement, plan, history, allowed_packages, training_data_files, validation_data_files
    ) -> str:
        return self._render(
            "training/generate.jinja",
            problem_statement=problem_statement,
            plan=plan,
            history=history,
            allowed_packages=allowed_packages,
            training_data_files=training_data_files,
            validation_data_files=validation_data_files,
            use_validation_files=len(validation_data_files) > 0,
        )

    def training_fix(
        self, training_code, plan, review, problems, allowed_packages, training_data_files, validation_data_files
    ) -> str:
        return self._render(
            "training/fix.jinja",
            training_code=training_code,
            plan=plan,
            review=review,
            problems=problems,
            allowed_packages=allowed_packages,
            training_data_files=training_data_files,
            validation_data_files=validation_data_files,
            use_validation_files=len(validation_data_files) > 0,
        )

    def training_review(self, problem_statement, plan, training_code, problems, allowed_packages) -> str:
        return self._render(
            "training/review.jinja",
            problem_statement=problem_statement,
            plan=plan,
            training_code=training_code,
            problems=problems,
            allowed_packages=allowed_packages,
        )

    def review_system(self) -> str:
        return self._render("review/system_prompt.jinja")

    def review_model(
        self,
        intent: str,
        input_schema: str,
        output_schema: str,
        solution_plan: str,
        training_code: str,
        inference_code: str,
    ) -> str:
        return self._render(
            "review/model.jinja",
            intent=intent,
            input_schema=input_schema,
            output_schema=output_schema,
            solution_plan=solution_plan,
            training_code=training_code,
            inference_code=inference_code,
        )

    def cot_system(self) -> str:
        return self._render("utils/system_prompt.jinja")

    def cot_summarize(self, context: str) -> str:
        return self._render("utils/cot_summarize.jinja", context=context)

    def agent_builder_prompt(
        self,
        intent: str,
        input_schema: str,
        output_schema: str,
        datasets: List[str],
        working_dir: str,
        max_iterations: int = None,
        schema_reasoning: str = None,
    ) -> str:
        return self._render(
            "agent/agent_manager_prompt.jinja",
            intent=intent,
            input_schema=input_schema,
            output_schema=output_schema,
            datasets=datasets,
            working_dir=working_dir,
            max_iterations=max_iterations,
            schema_reasoning=schema_reasoning,
        )


# Instantiate configuration and templates
config: _Config = _Config()
code_templates: _CodeTemplates = _CodeTemplates()
prompt_templates: _PromptTemplates = _PromptTemplates()


# Default logging configuration
def configure_logging(level: str | int = logging.INFO, file: str = None) -> None:
    # Configure the library's root logger
    sm_root_logger = logging.getLogger("plexe")
    sm_root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    sm_root_logger.handlers = []

    # Define a common formatter
    formatter = logging.Formatter(config.logging.format)

    stream_handler = logging.StreamHandler()
    # Only apply reconfigure if the stream supports it
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    stream_handler.setFormatter(formatter)
    sm_root_logger.addHandler(stream_handler)

    if file:
        file_handler = logging.FileHandler(file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        sm_root_logger.addHandler(file_handler)


configure_logging(level=config.logging.level)
