import torch

from kan import KANLinear

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


def get_batch(split, train_data, val_data, model):
    data = train_data if split == 'train' else val_data
    if len(data[0]) <= model.args.max_seq_len:
        raise ValueError(f"Data length ({len(data[0])}) is not sufficient for the sequence length ({model.args.max_seq_len}).")
    ix = torch.randint(len(data[0]) - model.args.max_seq_len, (model.args.max_batch_size,))
    x = torch.stack([data[0, i:i+model.args.max_seq_len] for i in ix])
    y = torch.stack([data[0, i+1:i+model.args.max_seq_len+1] for i in ix])
    return x.to("cpu"), y.to("cpu")

@torch.no_grad()
def estimate_loss(model, eval_steps, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            X, Y = get_batch(split, train_data=train_data, val_data=val_data, model=model)
            _, loss = model(X, start_pos=0, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, optimizer, train_data, val_data, max_steps=100, loss_intervall=10, eval_interval=10, eval_steps=10, save=False):
    steps = []
    train_losses = []
    val_losses = []

    print(f"Training for {max_steps} steps")

    for step in range(max_steps - 1):
    # every once in a while evaluate the loss on train and val sets
        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model=model, eval_steps=eval_steps, train_data=train_data, val_data=val_data)
            steps.append(step)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(split='train', train_data=train_data, val_data=val_data, model=model)

        # evaluate the loss
        logits, loss = model(xb, start_pos=0, targets=yb)

        if step % loss_intervall == 0 or step == max_steps - 1:
            print(f"step {step}: train loss {loss.item():.4f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for layer in model.layers:
            if isinstance(layer.mlp, KANLinear):
                with torch.no_grad():
                    layer.mlp.update_grid(xb)

    if save:
        # Save the trained model
        torch.save(model.state_dict(), "trained_KANama_model.pth")

    return model