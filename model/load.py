from args import ModelArgs, MOEModelArgs
import torch.nn as nn
import torch
import os
import json

def from_pretrained(path: str) -> nn.Module:
    # Load config.json from the path
    config_path = os.path.join(path, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get the 'model_type' from the config and apply the right ModelArgs
    model_type = config.get('model_type', None)
    if model_type is None:
        raise ValueError("model_type not found in config file")

    # Load the right ModelArgs based on the model_type
    if model_type == 'KANaMoEv1':
        model_args = MOEModelArgs.from_config(config)
    else:
        model_args = ModelArgs.from_config(config)

    # Create the model architecture using the loaded configuration
    model_class = globals().get(model_args.model_class, None)  # Ensure that model class is defined in the current scope
    if model_class is None:
        raise ValueError(f"Model class {model_args.model_class} not found in the scope")

    # Initialize the model with the loaded configuration
    model = model_class(model_args)

    # Load the model weights from the model.pth file
    model_weights_path = os.path.join(path, "pytorch_model.pth")
    
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
    
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    
    return model

def save_pretrained(path_to_save: str, model: nn.Module):
    # Ensure the save directory exists
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # Get the args from the model (assuming the model has an 'args' attribute)
    model_args = model.args
    model_type = getattr(model, 'model_type', None)
    
    if model_type is None:
        raise ValueError("The model does not have a 'model_type' attribute")
    
    # Prepare config data to save, including model_type and model_args
    config = {
        "model_type": model_type,
        **model_args.to_dict()  # Assuming model_args has a to_dict method
    }

    # Save the config as a JSON file
    config_path = os.path.join(path_to_save, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Save the model weights
    model_weights_path = os.path.join(path_to_save, "model.pth")
    torch.save(model.state_dict(), model_weights_path)
    
    print(f"Model saved successfully to {path_to_save}")
    
    return path_to_save