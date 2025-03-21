import yaml
import os
import re
from argparse import Namespace

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

def update_nested_args(config_data, args, passed_args):
    for key, value in config_data.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            update_nested_args(value, args, passed_args)
        elif hasattr(args, key):
            if key in passed_args:
                continue 
            # Resolve environment variables before updating
            if isinstance(value, str) and value.startswith('$'):
                value = resolve_env_vars(value)
            
            old_value = getattr(args, key)
            if old_value != value:
                setattr(args, key, value)
                print(f"Updated '{key}': {old_value} -> {value}")
        else:
            print(f"Warning: Key '{key}' in the config file is not a recognized argument.")

def get_passed_arguments(args, parser):
    """
    Determines which arguments were explicitly passed by the user.
    
    Args:
        args (namespace): argparse namespace from parser.parse_args
        parser (argparse): argument parser to get default values from
        
    Returns:
        set: set of argument names that were explicitly passed
    """
    # Get all default values from the parser
    args = parser.parse_args()
    sentinel = object()

    # Make a copy of args where everything is the sentinel.
    sentinel_ns = Namespace(**{key:sentinel for key in vars(args)})
    parser.parse_args(namespace=sentinel_ns)

    # Now everything in sentinel_ns that is still the sentinel was not explicitly passed.
    explicit = set(key for key, value in vars(sentinel_ns).items() if value is not sentinel) 
    return explicit

def replace_args_from_config(config_file, args, parser):
    """Replaces arguments in argparse namespace from the yaml file. If the argument
    was explicitly passed, it is ignored from the file

    Args:
        config_file (str): string path to config file
        args (namespace): agrpase namespace from parser.parse_args
        parser (argparse): argument parser to get default values from

    Returns:
        args: updated argparse namespace
    """

    passed_args = get_passed_arguments(args, parser)
    config_data = read_yaml_config(config_file)
    update_nested_args(config_data, args, passed_args)
    return args
