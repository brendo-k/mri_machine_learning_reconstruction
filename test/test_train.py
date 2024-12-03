import subprocess
import os
import pytest

# Directory where the config files are located
CONFIG_DIR = "./configs/train_learn_ssl_"

# List of YAML config files
config_files = [
    "pass_all_3.yaml",
    "pass_full.yaml",
    "pass_inverse.yaml",
    "pass_learn_ssl.yaml"
]

# Function to run the script with a given config file
def run_train_learn_ssl(config_file):
    config_path = CONFIG_DIR + config_file
    print(os.listdir('./configs'))
    
    # Run the Python script with the given config file
    result = subprocess.run(
        ["python", "train_learn_ssl_sampling.py", "--config", config_path],
        capture_output=True, text=True
    )
    
    return result

# Integration test for each config file
@pytest.mark.parametrize("config_file", config_files)
def test_run_train_learn_ssl_with_config(config_file):
    print(f"Running test with config: {config_file}")
    
    # Run the script
    result = run_train_learn_ssl(config_file)
    
    # Check if the script ran successfully
    assert result.returncode == 0, f"Error running script with config {config_file}. Error: {result.stderr}"
    
    # Optionally, assert that output contains expected strings
    # assert "some expected output" in result.stdout, f"Unexpected output with config {config_file}"
