from .KANamav1 import KANamav1
from .KANamav2 import KANamav2
from .KANamav3 import KANamav3
from .KANamav4 import KANamav4
from .KANaMoEv1 import KANaMoEv1
from .args import ModelArgs, MOEModelArgs
import torch.nn as nn
import torch
import os
import json

MODEL_CLASS_MAPPING = {
    "KANamav1": KANamav1,
    "KANamav2": KANamav2,
    "KANamav3": KANamav3,
    "KANamav4": KANamav4,
    "KANaMoEv1": KANaMoEv1,
}

def from_pretrained(path: str, device : str = "cpu") -> nn.Module:
    # Load config.json from the path
    config_path = os.path.join(path, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get the 'model_type' from the config
    model_type = config.get('model_type', None)
    if model_type is None:
        raise ValueError("model_type not found in config file")

    # Load the right ModelArgs based on the model_type
    if model_type == 'KANaMoEv1':
        model_args = MOEModelArgs(**config)  # Initialize MOEModelArgs using kwargs
    else:
        model_args = ModelArgs(**config)  # Initialize ModelArgs using kwargs

    # Ensure the model class is in the model class mapping
    model_class = MODEL_CLASS_MAPPING.get(model_type, None)
    if model_class is None:
        raise ValueError(f"Model class {model_type} not found in the model class mapping")

    # Initialize the model with the loaded configuration
    model = model_class(model_args)

    # Load the model weights from the model.pth file
    model_weights_path = os.path.join(path, "model.pth")
    
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
    
    # Use weights_only=True to avoid the warning and future-proof your code
    state_dict = torch.load(model_weights_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(state_dict)

    print("[INFO] Model and configuration loaded successfully")
    
    return model


def save_pretrained(path_to_save: str, model: nn.Module):
    # Ensure the save directory exists
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # Get the args from the model (assuming the model has an 'args' attribute)
    model_args = model.args

    # Use vars() to extract only the attributes (no need to manually specify them)
    config = {key: value for key, value in vars(model_args).items() if not key.startswith('__')}

    # Save the config as a JSON file
    config_path = os.path.join(path_to_save, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Save the model weights
    model_weights_path = os.path.join(path_to_save, "model.pth")
    torch.save(model.state_dict(), model_weights_path)

    print(f"[INFO] Model and configuration saved successfully to {path_to_save}")
    
    return path_to_save


def quick_inference(model: torch.nn.Module, tokens: torch.Tensor, max_new_tokens: int, tokenizer):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        for _ in range(max_new_tokens):
            # Take the last 'max_seq_len' tokens as input to the model
            tokens_conditioned = tokens[:, -model.args.max_seq_len:]
            
            # Get the model's predictions (logits)
            logits, _ = model(tokens_conditioned)
            
            # Apply softmax to the last token's logits to get probabilities
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            
            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1)
            
            # Append the predicted token to the input sequence
            tokens = torch.cat((tokens, next_token), dim=1)
            
            # Decode and print the token (convert from token ID to string)
            decoded_token = tokenizer.decode(next_token.squeeze(dim=1).tolist(), skip_special_tokens=True)
            print(decoded_token, end="", flush=True)
    
    # Return the final generated sequence (both as tokens and decoded text)
    return tokens, tokenizer.decode(tokens.squeeze(dim=0).tolist(), skip_special_tokens=True)