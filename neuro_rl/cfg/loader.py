import os
import yaml
import argparse
import json

from utils.common import dict_to_simplenamespace  # Adjust the import path based on your project structure

def load_configuration():
    # Initialize parser
    parser = argparse.ArgumentParser(description='Load YAML configuration.')

    # Adding argument
    parser.add_argument('--config_path', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--model_path', type=str, help='Optional override for the input path')
    parser.add_argument('--data_path', type=str, help='Optional override for the input path')
    parser.add_argument('--output_path', type=str, help='Optional override for the output path')

    # Parsing arguments
    args = parser.parse_args()
    
    # Construct the full path to the configuration file
    config_path = os.path.join(os.getcwd(), args.config_path)

    # Load YAML config file using the full path
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Convert the loaded config dictionary to a SimpleNamespace for easier attribute access
    cfg = dict_to_simplenamespace(config)

    # Override input_path and output_path if provided in command line arguments
    if args.model_path is not None:
        cfg.model_path = args.model_path
        if cfg.model_path.startswith('['):
            cfg.model_path = json.loads(cfg.model_path)
    if args.data_path is not None:
        cfg.data_path = args.data_path
        if cfg.data_path.startswith('['):
            cfg.data_path = json.loads(cfg.data_path)
    if args.output_path is not None:
        cfg.output_path = args.output_path
        cfg.output_path = args.output_path
        if cfg.output_path.startswith('['):
            cfg.output_path = json.loads(cfg.output_path)
    return cfg