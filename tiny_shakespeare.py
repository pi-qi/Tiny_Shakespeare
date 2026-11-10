##Import Liberies

from typing import Text
import torch
import torch.nn as nn
from torch.nn import functional as F


## Download the tiny shakespeare dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("input.txt", "r", encoding="utf-8") as f:
  text = f.read()

print("Number of Characters in Dataset: ", len(text))
print(text[:1000])

"""Vocab Size"""

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


"""Tokenization"""
# Character Level tokenizartion

## Create character-to-index mapping (encoding)
token = {}
for i, ch in enumerate(chars):
    token[ch] = i

## Create index-to-character mapping (decoding)
itoken = {}
for i, ch in enumerate(chars):
    itoken[i] = ch

def encode(text):
    tokens = []

    for character in text:
        token_id = token[character]
        tokens.append(token_id)

    return tokens

def decode(token_ids):
    characters = []

    for token_id in token_ids:
        character = itoken[token_id]
        characters.append(character)

    text = ''.join(characters)

    return text

print(encode('hello world'))
print(decode(encode('hello world')))

## Tokenization

data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
print(data[:100])

"""Split Dataset"""
# Training data 90%
# Validation data 10%

n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]

block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
  context = x[:t+1]
  target = y[t]
  print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)

batch_size = 4 # Number of independent sequence process in parallel
block_size = 8 # Maximum length of context for predicting the next token

## generate a small batch of data of input x and target y
def get_batch(split):

    # Choose which dataset to use
    if split == 'train':
        data = train_data
    else:
        data = val_data

    # Choose random starting positions
    starting_positions = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    # Create empty lists for inputs and targets
    x_batch = []
    y_batch = []

    # Create one input-target pair for each starting position
    for position in starting_positions:

        # Get a sequence of block_size tokens
        x = data[position : position + block_size]

        # Get the same sequence, shifted one token forward
        y = data[position + 1 : position + block_size + 1]

        # Add them to our batches
        x_batch.append(x)
        y_batch.append(y)

    # Convert the lists into PyTorch tensors
    x_batch = torch.stack(x_batch)
    y_batch = torch.stack(y_batch)

    return x_batch, y_batch

xb, yb = get_batch('train')
print('input: ')
print(xb.shape)
print(xb)
print('target: ')
print(yb.shape)
print(yb)


for b in range(batch_size): # batch dimention
  for t in range(block_size): # time dimention
    context = xb[b, :t + 1]
    target = yb[b, t]
    print(f"when input is {context.tolist()} the target: {target.item()}")

"""Bigram Language Model"""

# Pperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 100
learning_rate = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 264
n_head = 4
n_layer = 4
dropout = 0.2

chars = sorted(list(set(text)))
vocab_size = len(chars)

"""Characters to integers Mapping"""

# Create dictionaries to map characters to numbers and numbers to characters
st_to_in = {} # String to Integer
in_to_st = {} # Integer to String

# Go through the list of characters one by one
for i, ch in enumerate(chars):
    st_to_in[ch] = i      # Character to Integer
    in_to_st[i] = ch      # Integer to Character


# Encoder: take a string, output a list of integers
def encode(text):
    encoded_list = []
    for char in text:
        encoded_list.append(st_to_in[char])
    return encoded_list


# Decoder: take a list of integers, output a string
def decode(numbers):
    decoded_chars = []
    for num in numbers:
        decoded_chars.append(in_to_st[num])
    return "".join(decoded_chars)


"""Load data"""

def get_batch(split):

    # Choose whether to use training data or validation data
    if split == 'train':
        data = train_data
    else:
        data = val_data

    # Subtract "block_size" so the chunks don't go past the end of the text.
    max_index = len(data) - block_size
    ix = torch.randint(max_index, (batch_size,))

    # Empty lists to hold input chunks and target chunks
    x_list = []
    y_list = []

    # Loop through each random starting position
    for i in ix:
        # Grab a chunk of text for the input (x)
        chunk_x = data[i : i + block_size]

        # Grab the exact same chunk, but shifted one character to the right, for the target (y)
        chunk_y = data[i + 1 : i + block_size + 1]

        x_list.append(chunk_x)
        y_list.append(chunk_y)

    # Combine the lists into PyTorch Tensors
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    # Move the data to device (CPU or GPU)
    x = x.to(device)
    y = y.to(device)

    return x, y


"""Loss Function"""

@torch.no_grad()

def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      x, y = get_batch(split)
      logits, loss = model(x,y)
      losses[k] = loss.item()
      out[split] = losses.mean()
  model.train()
  return out

"""Self-Attention Head"""

## Single head Self Attention module
class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
  # compute attention score
    weight = q @ k.transpose(-2, -1) * C**-0.5
    weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    weight = F.softmax(weight, dim= -1)
    weight = self.dropout(weight)
  # perform the weighted aggregation of the value
    v = self.value(x)
    out = weight @ v
    return out

"""Multi head self attention module"""
## Self-Attention in Parallel

class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.projection = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.projection(out)
    return out

"""Feed Forward"""

class FeedForward(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout)
        )

  def forward(self, x):
    return self.net(x)

"""Block"""
## Communication followed by computation

class Block(nn.Module):

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.feed = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.feed(self.ln2(x))
    return x

"""Bigral Model"""

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embd_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head)
        for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.feed = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embd = self.token_embedding_table(idx) # (B,T,C)
        position_embd = self.position_embd_table(torch.arange(T, device = device))
        x = token_embd + position_embd
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)


"""Optimizer"""

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


"""Training Loop"""
""""I use my frive to save all the checkpoints, trining log, and model weights.
Chenge the following code, according to your file dir."""


from google.colab import files
import os
from google.colab import drive

drive.mount('/content/drive')
save_dir = '/file_path'

os.makedirs(save_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
log_path = os.path.join(save_dir, 'training_log.csv')

start_iter = 0
checkpoint_interval = 500

# Checking if a previous crashed session left a checkpoint behind
if os.path.exists(checkpoint_path):
    print(f"Found previous checkpoint. Resuming training from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter'] + 1
    print(f"Resumed successfully at iteration {start_iter}")

    # Open log file in append mode to avoid overwriting the previous history
    log_file = open(log_path, "a")
else:
    print("No previous checkpoint found")
    log_file = open(log_path, "w")
    log_file.write("step,train_loss,val_loss\n")
    log_file.flush()

# Main Training Loop
try:
    for iter in range(start_iter, max_iters):

        # Evaluate loss periodically
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            log_file.write(f"{iter},{losses['train']},{losses['val']}\n")
            log_file.flush() # Forces write to Google Drive immediately

        # Sample data, forward pass, backward pass, update weights
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save a checkpoint periodically
        if iter > 0 and iter % checkpoint_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
            }, checkpoint_path)
            print(f"Checkpoint backed up to Google Drive at iteration {iter}")

finally:
    log_file.close()

## Final model weight
torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))


"""Losses (Training and Validation)"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/file_path/training_log.csv')

plt.figure(figsize=(8, 4))

plt.plot(df['step'], df['train_loss'], label='Train Loss')
plt.plot(df['step'], df['val_loss'], label='Validation Loss')

# Get final loss values
final_train_loss = df['train_loss'].iloc[-1]
final_val_loss = df['val_loss'].iloc[-1]
final_step = df['step'].iloc[-1]

# Annotate final losses
plt.annotate(f'Train Loss: {final_train_loss:.3f}',
             (final_step, final_train_loss))

plt.annotate(f'Val Loss: {final_val_loss:.3f}',
             (final_step, final_val_loss))

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


"""Text Generation"""

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
    super().__init__()
    self.block_size = block_size
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embd_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
        *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
    )
    self.ln_f = nn.LayerNorm(n_embd)
    self.feed = FeedForward(n_embd)

    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embd = self.token_embedding_table(idx)  # (B,T,C)
    position_embd = self.position_embd_table(
        torch.arange(T, device=idx.device)
    )
    x = token_embd + position_embd
    x = self.blocks(x)

    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size :]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size)

weights_path = '/file_path/model_weights.pth'
model.load_state_dict(torch.load(weights_path, map_location=device))

m = model.to(device)
m.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))