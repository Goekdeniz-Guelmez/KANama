from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence

from model.kan import KANLinear

def train_old(model, dataloader, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            _, loss = model(inputs, targets=targets)
            
            loss.backward()
            optimizer.step()

            # Update grid points at the end of each epoch
            for layer in model.layers:
                if isinstance(layer.mlp, KANLinear):
                    with torch.no_grad():
                        for batch in dataloader:
                            inputs, _ = batch
                            layer.mlp.update_grid(inputs)
            
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


def get_batch_in_text_file(split, train_data, val_data, model, device: str):
    data = train_data if split == 'train' else val_data
    if len(data[0]) <= model.args.max_seq_len:
        raise ValueError(f"Data length ({len(data[0])}) is not sufficient for the sequence length ({model.args.max_seq_len}).")
    ix = torch.randint(len(data[0]) - model.args.max_seq_len, (model.args.max_batch_size,))
    x = torch.stack([data[0, i:i+model.args.max_seq_len] for i in ix])
    y = torch.stack([data[0, i+1:i+model.args.max_seq_len+1] for i in ix])
    return x.to(device), y.to(device)

def get_batch_in_jsonl(split, train_data, val_data, model, device: str):
    # Select the appropriate dataset
    data = train_data if split == 'train' else val_data
    
    # Ensure the data has enough tokens for the required sequence length
    if data.size(1) <= model.args.max_seq_len:
        raise ValueError(f"Data length ({data.size(1)}) is not sufficient for the sequence length ({model.args.max_seq_len}).")

    # Sample starting indices for each batch element
    max_index = data.size(1) - model.args.max_seq_len
    ix = torch.randint(0, max_index, (model.args.max_batch_size,))
    
    # Create batch sequences based on the sampled indices
    x = torch.stack([data[0, i:i + model.args.max_seq_len] for i in ix])
    y = torch.stack([data[0, i + 1:i + model.args.max_seq_len + 1] for i in ix])

    return x.to(device), y.to(device)

def get_batch(split, train_data, val_data, model, device: str):
    # Select the appropriate dataset
    data = train_data if split == 'train' else val_data
    num_sequences = data.size(0)
    seq_len = data.size(1)
    pad_token_id = model.args.pad_id
    
    # Randomly sample batch indices
    batch_indices = torch.randint(0, num_sequences, (model.args.max_batch_size,))
    
    # Generate random start positions and lengths
    max_seq_len = model.args.max_seq_len
    min_seq_len = model.args.min_seq_len if hasattr(model.args, 'min_seq_len') else (model.args.max_seq_len // 2)
    
    x = []
    y = []
    
    for i in batch_indices:
        while True:
            start = torch.randint(0, seq_len - min_seq_len, (1,)).item()
            length = torch.randint(min_seq_len, max_seq_len + 1, (1,)).item()
            end = min(start + length, seq_len)
            
            x_sequence = data[i, start:end]
            y_sequence = data[i, start+1:end+1]
            
            if not torch.all(x_sequence == pad_token_id):
                x.append(x_sequence)
                y.append(y_sequence)
                break  # Move to the next sequence if this one is valid
    
    # Pad the sequences to the length of the longest sequence in the batch
    x_padded = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
    y_padded = pad_sequence(y, batch_first=True, padding_value=pad_token_id)

    return x_padded.to(device), y_padded.to(device)

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, eval_steps, train_data, val_data, device: str):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            X, Y = get_batch(split, train_data=train_data, val_data=val_data, model=model, device=device)
            _, loss = model(X, start_pos=0, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train(
        model: torch.nn.Module,
        optimizer,
        train_data,
        val_data,
        scheduler=None,
        max_steps=100,
        loss_interval: int = 10,
        eval_interval: int = 10,
        eval_steps: int = 10,
        save_model_name: str = None,
        device: str = "cpu"
    ):
    torch.autograd.set_detect_anomaly(True)
    model.to(device)
    steps = []
    train_losses = []
    val_losses = []
    temperature_exists = False

    # Check if the first layer has the current_softmax_temp attribute
    temperature_exists = hasattr(model.layers[0].attention, 'current_softmax_temp') if model.layers else False

    print(f"\nTraining for {max_steps} steps")

    # Initialize tqdm progress bar
    with tqdm(total=max_steps, desc="Training Progress", unit="step") as pbar:
        for step in range(max_steps):
            # Every once in a while, evaluate the loss on train and val sets
            if step % eval_interval == 0 or step == max_steps - 1:
                losses = estimate_loss(model=model, eval_steps=eval_steps, train_data=train_data, val_data=val_data, device=device)
                steps.append(step)
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                
                # Update tqdm postfix
                postfix = {
                    "train_loss": f"{losses['train']:.4f}",
                    "val_loss": f"{losses['val']:.4f}"
                }
                
                if temperature_exists:
                    # Add temperature values for each layer if the attribute exists
                    for layer_id, layer in enumerate(model.layers):
                        postfix[f"layer_{layer_id}_temp"] = f"{layer.attention.current_softmax_temp:.2f}"
                
                pbar.set_postfix(postfix)

            # Sample a batch of data
            xb, yb = get_batch(split='train', train_data=train_data, val_data=val_data, model=model, device=device)

            # Evaluate the loss
            logits, loss = model(xb, start_pos=0, targets=yb)

            if step % loss_interval == 0 or step == max_steps - 1:
                pbar.set_postfix_str(f"step {step}, train loss: {loss.item():.4f}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()

            for layer in model.layers:
                if isinstance(layer.mlp, KANLinear):
                    with torch.no_grad():
                        layer.mlp.update_grid(xb)

            # Update the progress bar
            pbar.update(1)

    if save_model_name is not None:
        # Save the trained model
        torch.save(model.state_dict(), f"{save_model_name}.pth")

    return model