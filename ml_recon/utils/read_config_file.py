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

def update_nested_args(config_data, args, passed_args):
    for key, value in config_data.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            update_nested_args(value, args, passed_args)
        elif hasattr(args, key) and key not in passed_args:
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
    defaults = {}
    for action in parser._actions:
        if action.dest != 'help':  # Skip the help action
            defaults[action.dest] = action.default
    
    # Identify which arguments differ from their defaults
    passed_args = set()
    for arg_name, arg_value in vars(args).items():
        if arg_name in defaults and arg_value != defaults[arg_name]:
            passed_args.add(arg_name)
            
    return passed_args

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
