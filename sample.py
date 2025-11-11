import torch
from model import Transformer
from data import get_dataloaders

# Hyperparameters
block_size = 64
batch_size = 12
embed_dim = 128
n_layer = 4
n_head = 4
learning_rate = 6e-4
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

model = Transformer(vocab_size=vocab_size, block_size=block_size, embed_dim=embed_dim, n_layer=n_layer).to(device)
# model = torch.load("model.pt", weights_only=False)

model.eval() 
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with empty context
generated = model.sample(context, max_tokens=500)
generated_text = ''.join([itos[i] for i in generated[0].tolist()])
print("\n" + "="*50)
print("Generated text:")
print("="*50)
print(generated_text)

