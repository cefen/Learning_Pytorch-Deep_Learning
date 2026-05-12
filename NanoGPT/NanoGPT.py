import torch
import torch.nn as nn
from torch.nn import functional as F

# === 1. 超参数设置 ===
batch_size = 64      # 每次并行处理多少个序列
block_size = 256     # 句子的最大长度 (Context Window)
max_iters = 5000     # 总训练步数
eval_interval = 500  # 每隔多少步评估一次
learning_rate = 3e-4 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU的强项: 进行海量的矩阵运算
eval_iters = 200     # 在评估损失的estimate_loss中使用，表示评估平均损失时的采样样本数
n_embd = 384         # 词向量维度
n_head = 6           # 多头注意力的头数
n_layer = 6          # Transformer Block 的层数
dropout = 0.2        # 防止过拟合的丢弃率：在每一次训练步骤中，程序会随机选中 20% 的神经元，并强行把它们的输出变成 0。
# =====================

torch.manual_seed(1337) # 设定随机种子，使得程序每次运行时生成的初始随机数都是一样的，使得在不同设备上能得到相同的结果

# === 2. 数据读取与预处理 ===
with open("NanoGPT\input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# set给整篇文章去重，enumerate 给每个字符打上索引
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }    # 从字符到数字的映射表
itos = { i:ch for i,ch in enumerate(chars) }    # 从数字到字符的映射表
encode = lambda s: [stoi[c] for c in s]         # 编码函数
decode = lambda l: ''.join([itos[i] for i in l])# 解码函数 将列表里的字符连接成一个完整的字符串。'' 表示字符之间不留空格。结果就是 "ABA"。

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))    # 划分训练集和验证集，这里 训练:验证 = 9:1
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data

    #随机生成batch_size个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 提取输入
    x = torch.stack([data[i:i+block_size] for i in ix])
    # 提取标准输出（答案）， 即正确的预测结果
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)   # 如果有GPU, 就把数据从CPU的内存丢到显卡的显存里
    return x, y

# 本质: 
# model.eval()
# with torch.no_grad(): ...

@torch.no_grad() # 装饰器的作用：在进入函数前先关闭梯度记录，退出时又重新开启
def estimate_loss():
    '''评估损失'''
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

# === 3. 模型组件定义 ===

class Head(nn.Module):
    '''单头注意力'''
    def __init__(self, head_size):
        super().__init__() # 调用父类初始化，激活Pytorch的管理功能
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 在nn.Module的__init__中定义线性层时，它们的参数会被注册为Parameters, 意味着优化器在训练时会更新它们
        # self.register_buffer : 专门存放"非参数，但属于模型一部分"的数据，不会更新，但也会随着模型保存、搬到显卡上
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # 创建一个具体的层，用来执行丢弃任务
        # nn.Dropout(dropout_rate)是nn.Module提供的功能层,
        # 调用model.train()时：nn.Module会通知所有dropout层：开始工作
        # 调用model.eval()时：会停止丢弃，准备预测。
        self.dropout = nn.Dropout(dropout)
        # 底层逻辑：1. 生成掩码，如对于(1, 1, 1, 1), 丢弃率0.25, 则产生掩码(随机) (0, 1, 1, 1)
        # 2. 丢弃 ： (1, 1, 1, 1) * (0, 1, 1, 1) = (0, 1, 1, 1)
        # 3. 缩放(使得模长不变) 所有数值除以(1 - p)
 
    def forward(self, x):
        B, T, C = x.shape # Batch, Time(Tokens, 有多少单词), Channels(向量维度)
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # k.transpose(-2, -1) : (B, head_size, T), 只转置最后两个维度进行矩阵相乘
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # wei : (B, T, T)
        wei = self.dropout(wei)      # 随机丢弃
        v = self.value(x)
        out = wei @ v # (B, T, T) * (B, T, head_size) = (B, T, head_size) 
        
        # wei (B, T, T)：这是一个权重矩阵。对于 batch 中的每一行，它告诉我们：为了理解当前位置的词，我们需要对句子中其他位置的词付出多少“注意力” 。
        # v (B, T, head_size)：这是每个词的“特征向量”
        # 得到的out就是学习了上下文之后，表示了每个词的特征的矩阵
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # self.proj : 将多头注意力分别处理数据并拼接得到的矩阵再进行一次处理，从而融合多头的数据特征
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 拼接 n_embd // head_size 个 (B, T, head_size) 得到 (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    '''前馈网络'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        '''前向传播'''
        return self.net(x)

class Block(nn.Module):
    '''Transformer 块'''
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 先经过自注意力机制(Q/K/V)，获取词与词之间的联系。详细过程：
        # 1. 自注意力机制通过三个线性变换(矩阵乘法),将每个词投射到Q,K,V三个空间
        # 2. 算匹配度: weight = Q @ K^T 得到关系矩阵, weight(i, j) 表示"处理第i个词时，第j个词的重要性"。
        # 复习前面所说的，为了避免"偷看"，会将上三角的内容变为-inf，使得第i个词只能关注到前i个词。
        # 3. 归一化
        x = x + self.sa(self.ln1(x))
        # 再通过前馈网络，其实就是一组线性层加上一个激活函数，来处理得到的信息
        x = x + self.ffwd(self.ln2(x))
        return x

# === 4. 主模型类 ===

class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # 拿到(batch_size, tokens, n_embd),即原词表中提取的部分特征向量
        # torch.arange(T) ：生成一个从 0 到 T-1 的等差数列张量
        # device = device : 在Pytorch中，两个张量必须处在同一个设备上才能进行运算
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (tokens, n_embd)
        x = tok_emb + pos_emb # 加上位置张量，给输入增添了位置信息(运算采用广播机制)
        x = self.blocks(x)
        x = self.ln_f(x)
        # 解码，输出的logits(batch_size, tokens, vocab_size)
        # 对每一批(batch)的每个词(tokens), 长度为vocab_size 的向量表示词表中"下一个词"出现的概率
        logits = self.lm_head(x)

        if targets is None: # target 标准答案 (B, T)
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # F.cross_entropy的输入是二维的：(样本数，类别数)
            loss = F.cross_entropy(logits, targets) # 计算交叉熵

        return logits, loss

    def generate(self, idx, max_new_tokens):
        '''idx : (B, T), B个起始点与其后面的共T个字符'''
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # 只选取最后block_size个词来预测下一个词
            # 好习惯：self(idx_cond)而不是forward(idx_cond), 虽然是实际上是调用了forward
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, 1, n_embd)， 取最后一个词的预测结果，也就是预测出来的下一个词的可能性
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # 为每个词根据概率分布抽取下一个词(而不是直接选择概率最高的词)
            idx = torch.cat((idx, idx_next), dim=1) # 把抽中的新词拼接在旧序列的后面
        return idx # (B, T + max_new_tokens)， 因为循环了max_new_tokens次，每次拼接一个词

# === 5. 实例化与训练 ===

model = NanoGPT()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"开始训练，设备：{device}")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"步数 {iter}: 训练集 Loss {losses['train']:.4f}, 验证集 Loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# === 6. 生成结果测试 ===
print("\n--- 训练完成，正在生成样本文本 ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))