import yaml
import os

def config_yaml(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    If the file doesn't exist, create a default configuration.
    """
    # Default configuration
    default_config = {
        "preprocessing": {
            "augmentation": True
        },
        "training": {
            "learning_rate": 0.001,
            "momentum": 0.9,
            "epochs": 5,
            "batch_size": 32
        }
    }
    
    # If config file exists, load it
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Create the default config file
        config = default_config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created default configuration file at {config_path}")
    
    return config
