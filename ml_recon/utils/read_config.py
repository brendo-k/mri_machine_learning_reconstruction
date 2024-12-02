import yaml 

def read_yaml_config(config_file):
    if not config_file: 
        return {}

    with open(config_file, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    return config_data

def replace_args_from_config(config_file, args):
    config_data = read_yaml_config(config_file)
    for key, new_value in config_data.items():
        if hasattr(args, key):
            old_value = getattr(args, key)
            if old_value != new_value:
                setattr(args, key, new_value)
                print(f"Updated '{key}': {old_value} -> {new_value}")
        else:
            print(f"Warning: Key '{key}' in the config file is not a recognized argument.")
    return args