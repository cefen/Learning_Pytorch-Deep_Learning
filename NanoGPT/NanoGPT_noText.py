# 少注释版本

import torch
import torch.nn as nn
from torch.nn import functional as F
import os 
import glob
import numpy as np
import pickle
import time

torch.set_float32_matmul_precision('high')

# ======== 超参数 =========
batch_size = 64
block_size = 256  
max_iters = 15000
eval_interval = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.2
temperature = 0.7
manual_seed = 1337
# ==================

torch.manual_seed(manual_seed) 

# ======== 数据处理 =========
meta_path = 'meta.pkl'
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f"成功加载词表，大小为: {vocab_size}")
else:
    raise FileNotFoundError("未找到 meta.pkl")

train_data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('data/val.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# =====================================

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =========== 模型组件 ==========

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.qkv_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=dropout if self.training else 0
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# =============================

# ========== 顶层容器 ==========

class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ================================

# ================保存训练进度=================
def save_checkpoint(model, optimizer, iter, loss, filename="ckpt.pt"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter,
        'loss': loss,
        'vocab_size': vocab_size,
    }
    # 1. 保存一个最新的（用于断点续传）
    torch.save(checkpoint, "CheckPoint/ckpt_last.pt")
    
    # 2. 每隔一定步数，保存一个永久备份
    if iter % 1000 == 0:
        torch.save(checkpoint, f"CheckPoint/ckpt_iter_{iter}.pt")
        
    # 3. 只有当 Loss 创下新低时，保存一个“最佳”版本
    # (这需要你在外部维护一个 best_loss 变量)
    print(f"--> 已保存步数: {iter}")

def load_checkpoint(model, optimizer, filename="CheckPoint/ckpt.pt"):
    if os.path.exists(filename):
        print(f"恢复存档 {filename}...")
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['iter']
    return 0

# ==========================================


# =================== 主训练循环 =================
if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = NanoGPT()
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    # ===============
    current_iter = load_checkpoint(model, optimizer=optimizer, filename="CheckPoint/ckpt_last.pt")
    # ===============
    start_time = time.time()
    try:
        best_val_loss = float('inf')
        scaler = torch.amp.GradScaler('cuda')
        for iter in range(current_iter, max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss()
                print(f"步数 {iter}: 训练集 Loss {losses['train']:.4f}, 验证集 Loss {losses['val']:.4f}")
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    save_checkpoint(model, optimizer, iter, losses['val'])
                    print("发现更优模型，更新best_model.pt")

            xb, yb = get_batch('train')
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(xb, yb)   

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    except KeyboardInterrupt:
        current_loss = loss.item() if 'loss' in locals() else 0.0
        save_checkpoint(model, optimizer, iter, current_loss)

    total_time = time.time() - start_time
    print(f"\n耗时 {total_time // 60:.0f} min, 生成测试:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=1500)[0].tolist()))