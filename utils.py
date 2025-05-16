import yaml
import os


def config_yaml(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file {config_path} not found.") 
        
    return config
