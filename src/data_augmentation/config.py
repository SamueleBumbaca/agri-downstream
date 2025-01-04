from dataclasses import dataclass
import os
import yaml

@dataclass
class Config:
    image_dir: str
    bbox_file: str
    output_dir: str
    latent_dim: int
    hidden_dim: int
    image_size: tuple
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    split: str

def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def save_config(config, output_dir):
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

def create_config(config_path):
    # Load VAE configuration
    config = load_config(config_path)
    # Load handcrafted configuration
    handcrafted_config_path = config['handcrafted_config']
    handcrafted_config = load_config(handcrafted_config_path)
    resolution = handcrafted_config['dataset']['resolution']
    min_dist_on_row = handcrafted_config['dataset']['min_dist_on_row']
    dist_tolerance = handcrafted_config['dataset']['dist_tolerance']
    # Calculate the image size
    box_size = (min_dist_on_row - dist_tolerance) / resolution
    image_size = (int(box_size), int(box_size))
    # Create the configuration dictionary
    config_dict = {
        'image_dir': config['image_dir'],
        'bbox_file': config['bbox_file'],
        'output_dir': config['output_dir'],
        'latent_dim': config['latent_dim'],
        'hidden_dim': config['hidden_dim'],
        'image_size': image_size,        
        'batch_size': int(config['batch_size']),
        'learning_rate': float(config['learning_rate']),
        'weight_decay': float(config['weight_decay']),
        'num_epochs': int(config['num_epochs']),
        'split': handcrafted_config['data']['split']
    }
    # Create the configuration object
    config = Config(**config_dict)
    # Create the output directory
    os.makedirs(config.output_dir, exist_ok=True)
    # Save the configuration to a YAML file
    save_config(config_dict, config.output_dir)
    return config