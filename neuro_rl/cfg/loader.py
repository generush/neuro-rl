import os
import yaml
import argparse

from utils.common import dict_to_simplenamespace  # Adjust the import path based on your project structure

def load_configuration():
    # Initialize parser
    parser = argparse.ArgumentParser(description='Load YAML configuration.')

    # Adding argument
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')

    # Parsing arguments
    args = parser.parse_args()

    # Construct the full path to the configuration file
    config_path = os.path.join(os.getcwd(), args.config_path)

    # Load YAML config file using the full path
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Convert the loaded config dictionary to a SimpleNamespace for easier attribute access
    cfg = dict_to_simplenamespace(config)

    return cfg