import torch

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


def get_batch(split, train_data, val_data, model, device: str):
    data = train_data if split == 'train' else val_data
    if len(data[0]) <= model.args.max_seq_len:
        raise ValueError(f"Data length ({len(data[0])}) is not sufficient for the sequence length ({model.args.max_seq_len}).")
    ix = torch.randint(len(data[0]) - model.args.max_seq_len, (model.args.max_batch_size,))
    x = torch.stack([data[0, i:i+model.args.max_seq_len] for i in ix])
    y = torch.stack([data[0, i+1:i+model.args.max_seq_len+1] for i in ix])
    return x.to(device), y.to(device)

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
        save: bool = False,
        print_softmax_temp: bool = True,
        device: str = "cpu"
    ):
    torch.autograd.set_detect_anomaly(True)
    model.to(device)
    steps = []
    train_losses = []
    val_losses = []
    temperature_exists = False

    if print_softmax_temp:
        # Check if the first layer has the current_softmax_temp attribute
        temperature_exists = hasattr(model.layers[0].attention, 'current_softmax_temp') if model.layers else False


    print(f"Training for {max_steps} steps")

    for step in range(max_steps - 1):
        # Every once in a while, evaluate the loss on train and val sets
        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model=model, eval_steps=eval_steps, train_data=train_data, val_data=val_data, device=device)
            steps.append(step)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            print(f"step {step}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

            if temperature_exists:
                # Print temperature values for each layer if the attribute exists
                for layer_id, layer in enumerate(model.layers):
                    print(f"Layer {layer_id} temperature: {layer.attention.current_softmax_temp}")

        # Sample a batch of data
        xb, yb = get_batch(split='train', train_data=train_data, val_data=val_data, model=model, device=device)

        # Evaluate the loss
        logits, loss = model(xb, start_pos=0, targets=yb)

        if step % loss_interval == 0 or step == max_steps - 1:
            print(f"step {step}, train loss: {loss.item():.4f}")

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

    if save:
        # Save the trained model
        torch.save(model.state_dict(), "trained_KANama_model.pth")

    return model