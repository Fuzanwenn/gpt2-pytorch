import torch
import torch.nn.functional as F
from model import Transformer
from data import get_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
block_size = 64
batch_size = 12
embed_dim = 128
n_layer = 4
n_head = 4
learning_rate = 6e-4
max_epochs = 2000        # here: 2000 training steps (not full passes over data)
eval_interval = 500

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Load data
train_loader, test_loader, vocab_size, stoi, itos = get_dataloaders(
    "shakespeare.txt", block_size, batch_size, train_split=0.9
)

# Initialize model
model = Transformer(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_dim=embed_dim,
    n_layer=n_layer
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.1,
    betas=(0.9, 0.95)
)

# Training loop: run for max_epochs steps
model.train()
loss_history = []

train_iter = iter(train_loader)

for step in tqdm(range(1, max_epochs + 1), desc="Training"):
    # recycle dataloader if we exhaust it
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    x, y = x.to(device), y.to(device)

    # Forward
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log loss
    loss_val = loss.item()
    loss_history.append(loss_val)

    if step % eval_interval == 0 or step == 1:
        print(f"Step {step}/{max_epochs}, loss = {loss_val:.4f}")

# Save model
torch.save(model, "model.pt")

# Plot loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training loss over time")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
print("Saved loss curve to loss_curve.png")
