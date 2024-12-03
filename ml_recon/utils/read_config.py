import yaml

def read_yaml_config(config_file):
    if not config_file:
        return {}

    with open(config_file, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    return config_data

def update_nested_args(config_data, args):
    for key, value in config_data.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            update_nested_args(value, args)
        elif hasattr(args, key):  # Adjust key mapping
            old_value = getattr(args, key)
            if old_value != value:
                setattr(args, key, value)
                print(f"Updated '{key}': {old_value} -> {value}")
        else:
            print(f"Warning: Key '{key}' in the config file is not a recognized argument.")

def replace_args_from_config(config_file, args):
    config_data = read_yaml_config(config_file)
    update_nested_args(config_data, args)
    return args