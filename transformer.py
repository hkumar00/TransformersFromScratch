import os
import urllib.request as request
import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy
import torch.nn as nn


GPT_MODEL_124M = {
    "vocab_size": 50257,
    "embed_size": 768,
    "context_len": 256,
    "num_layers": 12,
    "num_heads": 12,
    "dropout": 0.1,
    "qkv_bias": True
}

class ConfigurationDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not found")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not found")

cfg = ConfigurationDict(GPT_MODEL_124M)

class CustomDataset(Dataset):
    def __init__(self, text, tokenizer, context_len, stride=1):
        self.input_tokens = []
        self.target_tokens = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids)-context_len, stride):
            # Input and target would be shifted by 1 for next word prediction
            input = token_ids[i:i+context_len]
            target = token_ids[i+1:i+context_len+1]

            # Data loader would expect tensors from the dataset
            self.input_tokens.append(torch.tensor(input))
            self.target_tokens.append(torch.tensor(target))
    
    def __len__(self):
        # Length of dataset
        return len(self.input_tokens)
    
    def __getitem__(self, index):
        # Return next item in dataset
        return self.input_tokens[index], self.target_tokens[index]
    
def create_verdict_dataloader(text, context_len=256, stride=128, 
                            batch_size=4, shuffle=True, drop_last=True, 
                            num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = CustomDataset(text, tokenizer, context_len, stride)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader

class MultiHeadAttentionParallel(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_len, dropout=0.0, qkv_bias=False):
        super(MultiHeadAttentionParallel, self).__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))
        self.out_proj = nn.Linear(d_out, d_out)
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape   # batch size, input tokens, embedding dimension
        keys = self.W_key(x)    # b, input_tokens, d_out
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) # b, num_heads, input_tokens, head_dim
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        ctx_vecs = torch.matmul(attn_weights, values).transpose(1, 2)
        
        ctx_vecs = ctx_vecs.contiguous().view(b, num_tokens, self.d_out)
        ctx_vecs = self.out_proj(ctx_vecs)

        return ctx_vecs

class LayerNorm(nn.Module):
    """"
        Layer Normalization
            x_i' = (x_i - mean) / sqrt(var + eps)
            x_i' = scale * x_i' + shift
            where x_i is the ith input, mean and var are the mean and variance of the input sample
            scale and shift are learnable parameters.
            eps is a small value to avoid division by zero (numerical stability)

        For layer normalization, we normalize across features for each input sample
        instead of across the batch as in batch normalization.
    """
    def __init__(self, embed_size):
        super(LayerNorm, self).__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_size))
        self.shift = nn.Parameter(torch.zeros(embed_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# Gaussian error linear units
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    
    def forward(self, x):
        return 0.5*x*(1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi))*(x + 0.044715*torch.pow(x, 3))))

class FeedForwardNN(nn.Module):
    def __init__(self, cfg):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embed_size, 4*cfg.embed_size),
            GELU(),
            nn.Linear(4*cfg.embed_size, cfg.embed_size)
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    """
        -> LayerNorm1 -> Masked MutiHeadAttention -> Dropout  + ->  LayerNorm2 -> FeedForward --> Dropout + ->
        |                                                     | |                                     |
        -------------------------------------------------------  -------------------------------------- 
    """
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.ln_1 = LayerNorm(cfg.embed_size)
        self.attention = MultiHeadAttentionParallel(cfg.embed_size, cfg.embed_size, 
                                                    cfg.num_heads, cfg.context_len,
                                                    cfg.dropout, cfg.qkv_bias)

        self.fc = FeedForwardNN(cfg)
        self.ln_2 = LayerNorm(cfg.embed_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x_residual = x
        x = self.ln_1(x)
        x = self.attention(x)
        x = x_residual + self.dropout(x)
        
        x_residual = x

        x = self.ln_2(x)
        x = self.fc(x)
        x = x_residual + self.dropout(x)
        return x

## LLM architecture
"""
Tokenizedtext -> tokenEmdlayer -> PosEmbLayer -> Dropout -> [Transformer Block]*12
                                                                                |
                                                        LinearLayer <- LayerNorm<
"""
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super(GPTModel, self).__init__()
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.position_embeddings = nn.Embedding(cfg.context_len, cfg.embed_size)
        self.dropout = nn.Dropout(cfg.dropout)

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.num_layers)])

        self.ln_f = LayerNorm(cfg.embed_size)
        self.out_head = nn.Linear(cfg.embed_size, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        token_embeddings = self.token_embeddings(in_idx)
        position_embeddings = self.position_embeddings(torch.arange(seq_len, device=in_idx.device))
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.ln_f(x)
        logits = self.out_head(x)
        return logits


def text_to_token_ids(text, tokenizer):
    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    token_ids = torch.tensor(tokens).unsqueeze(0)   # adds the batch dimension
    return token_ids

def token_ids_to_text(ids, tokenizer):
    ids = ids.squeeze(0).tolist()
    return tokenizer.decode(ids)

def generate_text(model, idx, max_tokens, context_len):
    for _ in range(max_tokens):
        idx_cond = idx[:, -context_len:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def loss_for_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def loss_across_dataset(loader, model, device, nbatches=None):
    total_loss = 0.
    if len(loader) == 0:
        return float("nan")
    elif nbatches is None:
        nbatches = len(loader)
    else:
        nbatches = min(nbatches, len(loader))

    for i, (input_batch, target_batch) in enumerate(loader):
        if i < nbatches:
            loss = loss_for_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / nbatches


def evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = loss_across_dataset(train_dataloader, model, device, nbatches=eval_iter)
        val_loss = loss_across_dataset(val_dataloader, model, device, nbatches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_text(model, tokenizer, device, context):
    model.eval()
    context_len = model.position_embeddings.weight.shape[0]
    token_ids = text_to_token_ids(context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model, token_ids, max_tokens=50, 
                                  context_len=context_len)
    text = token_ids_to_text(token_ids, tokenizer)
    print(f"{text}".replace("\n", ""))
    model.train()

def training_loop(model, train_dataloader, val_dataloader, optimizer,
                  device, epochs, eval_after, eval_iter, start_context,
                  tokenizer):
    """
    Training Loop
        for each epoch -> (loop over entire dataset)
            for each batch supplied by loader
                reset loss gradients
                find loss for current batch
                loss.backward()
                optim.step()
                print train / val losses for batch
            generate sample text for inspection after each epoch
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = loss_for_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_after == 0:
                train_loss, val_loss = evaluate_model(model, train_dataloader,
                                                      val_dataloader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)

                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            global_step +=  1

        generate_and_print_text(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

### Training and Validation losses

with open("verdict.txt", "r") as fh:
    verdict_text = fh.read()

print(f"Characters in dataset : {len(verdict_text)}")
print(f"Tokens in dataset : {len(tokenizer.encode(verdict_text))}")

train_ratio = 0.9
split_idx = int(train_ratio * len(verdict_text))
train_data = verdict_text[:split_idx]
val_data = verdict_text[split_idx:]

train_dataloader = create_verdict_dataloader(train_data, context_len=cfg.context_len,
                                           stride=cfg.context_len, batch_size=2,
                                           shuffle=True, drop_last=True,
                                           num_workers=0)

val_dataloader = create_verdict_dataloader(val_data, context_len=cfg.context_len,
                                           stride=cfg.context_len, batch_size=2,
                                           shuffle=False, drop_last=False,
                                           num_workers=0)

torch.manual_seed(123)

model = GPTModel(cfg)
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

train_losses, val_losses, tokens_seen = training_loop(model, train_dataloader, val_dataloader, optim,
                  device, epochs=10, eval_after=5, eval_iter=5,
                  start_context="Every effort moves you", tokenizer=tokenizer)
