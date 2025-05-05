import subprocess
import torch
import os
import pytest
from pathlib import Path

# Directory where the config files are located
CONFIG_DIR = Path("./configs/")

CONFIG_FILES = [
    os.path.join(CONFIG_DIR, f)
    for f in os.listdir(CONFIG_DIR) 
    if f.endswith(".yaml") 
]


# Function to run the script with a given config file
def run_train_learn_ssl(config_file):
    print(os.listdir('.'))
    
    # Run the Python script with the given config file
    result = subprocess.run(
        [
            "python", "train.py", 
            "--config", config_file, 
            "--chans", "10", 
            "--fast_dev_run"
        ],
        capture_output=True, text=True
    )
    
    return result


# Integration test for each config file
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
@pytest.mark.parametrize("config_file", CONFIG_FILES)
def test_run_train_learn_ssl_with_config(config_file):
    print(f"Running test with config: {config_file}")
    
    # Run the script
    result = run_train_learn_ssl(config_file)
    
    # Check if the script ran successfully
    assert result.returncode == 0, f"Error running script with config {config_file}. Error: {result.stderr}"
    
    # Optionally, assert that output contains expected strings
    # assert "some expected output" in result.stdout, f"Unexpected output with config {config_file}"
