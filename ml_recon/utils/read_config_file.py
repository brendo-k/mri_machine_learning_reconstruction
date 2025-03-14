import yaml
import os
import re

def read_yaml_config(config_file):
    if not config_file:
        return {}

    with open(config_file, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    return config_data

def resolve_env_vars(value):
    """Replace environment variable references in a string."""
    if isinstance(value, str) and value.startswith('$'):
        # Extract the environment variable name (removing the $ prefix)
        env_var_name = value[1:]
        # Get the value from environment variables or return empty string if not found
        return os.environ.get(env_var_name, '')
    return value

def update_nested_args(config_data, args):
    for key, value in config_data.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            update_nested_args(value, args)
        elif hasattr(args, key):
            # Resolve environment variables before updating
            if isinstance(value, str) and value.startswith('$'):
                resolved_value = resolve_env_vars(value)
            else:
                resolved_value = value
            old_value = getattr(args, key)
            if old_value != resolved_value:
                setattr(args, key, resolved_value)
                print(f"Updated '{key}': {old_value} -> {resolved_value}")
        else:
            print(f"Warning: Key '{key}' in the config file is not a recognized argument.")

def replace_args_from_config(config_file, args):
    config_data = read_yaml_config(config_file)
    update_nested_args(config_data, args)
    return args
