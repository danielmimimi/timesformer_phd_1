import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def write_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def update_freeze_layers(config, new_freeze_layers):
    if 'TRAIN' in config and 'FREEZE_LAYERS' in config['TRAIN']:
        config['TRAIN']['FREEZE_LAYERS'] = new_freeze_layers
    return config


def main(input_file, output_file, new_freeze_layers):
    # Read the YAML file
    config = read_yaml(input_file)
    
    # Update the FREEZE_LAYERS
    updated_config = update_freeze_layers(config, new_freeze_layers)
    
    # Write the updated configuration to a new YAML file
    write_yaml(updated_config, output_file)
    print(f"Updated configuration written to {output_file}")



if __name__ == "__main__":
    # File paths
    input_file = 'config.yaml'
    output_file = 'updated_config.yaml'
    
    # New FREEZE_LAYERS to be set
    new_freeze_layers = ["blocks.2", "blocks.3"]
    
    main(input_file, output_file, new_freeze_layers)