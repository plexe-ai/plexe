"""Configuration management for MLE Bench runner."""

import os
import sys
from pathlib import Path
import yaml
from jinja2 import Environment, Template, meta


class ConfigManager:
    """Class to handle configuration loading and generation"""

    @staticmethod
    def load_config(config_path):
        """Load configuration from YAML file"""
        print(f"🔍 Loading test configuration from {config_path}...")
        if not os.path.exists(config_path):
            print(f"❌ Config file not found at: {config_path}")
            sys.exit(1)
        try:
            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file)
            print("✅ Configuration loaded successfully.")
            return config
        except yaml.YAMLError as e:
            print(f"❌ Error parsing config file: {e}")
            sys.exit(1)

    @staticmethod
    def ensure_config_exists(rebuild: bool = False):
        """Check if `mle-bench-config.yaml` exists, and if not, generate it from `mle-bench-config.yaml.jinja`"""
        config_path = Path("mle-bench-config.yaml")
        if config_path.exists() and not rebuild:
            print("✅ Configuration file 'mle-bench-config.yaml' already exists.")
            return

        # Get the script directory for finding the template
        script_dir = Path(__file__).parent.parent.parent.absolute()
        template_path = script_dir / "mle-bench-config.yaml.jinja"

        if not template_path.exists():
            print(f"❌ Template file '{template_path}' not found. Cannot proceed.")
            sys.exit(1)

        if rebuild:
            print(f"🔄 Rebuilding 'mle-bench-config.yaml' from '{template_path}'...")
        else:
            print(f"📝 'mle-bench-config.yaml' not found. Generating it from '{template_path}'...")

        # Load the template
        with open(template_path, "r") as template_file:
            template_content = template_file.read()

        env = Environment()
        ast = env.parse(template_content)
        template = Template(template_content)

        print(f"📝 Template loaded from {template_path}")

        # Set default values and gather user inputs for template variables
        variables = {
            "repo_dir": str(Path.home() / "mle-bench"),
            "provider": "openai/gpt-4o",
            "max_iterations": "3",
            "timeout": "3600",
        }

        # Allow user to override defaults
        for var in meta.find_undeclared_variables(ast):
            if not var.startswith("_"):
                if var in variables:
                    prompt = f"💡 Provide a value for '{var}' (default: {variables[var]}): "
                else:
                    prompt = f"💡 Provide a value for '{var}': "

                try:
                    user_input = input(prompt)
                    if user_input.strip():  # Only update if user provided a non-empty value
                        variables[var] = user_input
                except EOFError:
                    print(f"Using default value for '{var}': {variables.get(var, '')}")

        # Render and write the config file
        config_content = template.render(**variables)

        # Parse the rendered config
        config_yaml = yaml.safe_load(config_content)

        # Add plexe configurations to the config
        config_yaml["provider"] = variables["provider"]
        config_yaml["max_iterations"] = int(variables["max_iterations"])
        config_yaml["timeout"] = int(variables["timeout"])

        # Write the updated config
        with open(config_path, "w") as config_file:
            yaml.dump(config_yaml, config_file, default_flow_style=False)

        print("✅ 'mle-bench-config.yaml' generated successfully.")
