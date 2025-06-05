# Transformer Scratch

这是一个从零开始实现 Transformer 架构的项目，基于论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。本项目旨在帮助读者深入理解 Transformer 的工作原理，并通过实际编码来掌握其实现细节。

## 背景介绍

### Transformer 的诞生

在 Transformer 出现之前，序列建模和转换任务主要依赖于 RNN（循环神经网络）和 LSTM（长短期记忆网络）。这些模型虽然在某些任务上表现不错，但存在以下问题：

1. **并行化困难**：RNN 需要按顺序处理序列，难以充分利用现代硬件的并行计算能力。
2. **长程依赖问题**：当序列较长时，RNN 难以捕捉远距离的依赖关系。
3. **训练效率低**：由于序列处理的顺序性，训练过程较慢。

2017 年，Google 的研究团队提出了 Transformer 架构，它完全基于注意力机制，解决了上述问题：

1. **并行计算**：可以同时处理整个序列，充分利用 GPU/TPU 的并行能力。
2. **全局依赖**：通过自注意力机制，直接建立任意位置之间的关联。
3. **训练效率**：并行计算显著提升了训练速度。

### 核心创新

Transformer 的核心创新在于：

1. **自注意力机制**：允许模型直接关注序列中的任何位置，计算它们之间的关联度。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用 mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用 softmax 获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)
```

2. **多头注意力**：将注意力分成多个头，每个头关注不同的特征子空间。

```python
class MultiHeadAttention(nn.Module):
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用 softmax 获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        return output
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 对每个头分别计算注意力
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)
```

3. **位置编码**：通过正弦位置编码，为模型提供序列位置信息。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与反向传播）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 添加位置编码
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

4. **残差连接和层归一化**：帮助训练更深的网络。

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力层
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接和层归一化
        
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

5. **掩码机制**：用于处理变长序列和防止信息泄露。

```python
def create_padding_mask(seq):
    # 创建填充掩码
    mask = (seq == 0).unsqueeze(1).unsqueeze(2)
    return mask

def create_look_ahead_mask(size):
    # 创建前瞻掩码
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

def create_masks(src, tgt):
    # 编码器掩码
    enc_padding_mask = create_padding_mask(src)
    
    # 解码器掩码
    dec_padding_mask = create_padding_mask(src)
    look_ahead_mask = create_look_ahead_mask(tgt.size(1))
    dec_target_padding_mask = create_padding_mask(tgt)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask
```

6. **学习率调度**：使用 warmup 和余弦退火策略。

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # 线性预热
            lr = self.current_step / self.warmup_steps
        else:
            # 余弦退火
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * (1 + math.cos(math.pi * progress))
        
        # 更新学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['initial_lr']
```

这些核心创新点共同构成了 Transformer 的强大能力：

1. **自注意力机制**使模型能够直接建模任意位置之间的关系，解决了 RNN 的长程依赖问题。
2. **多头注意力**允许模型同时关注不同的特征子空间，增强了模型的表达能力。
3. **位置编码**为模型提供了序列位置信息，弥补了自注意力机制对位置不敏感的缺点。
4. **残差连接和层归一化**帮助训练更深的网络，提高了模型的性能。
5. **掩码机制**确保了模型在处理变长序列和生成任务时的正确性。
6. **学习率调度**策略帮助模型更好地收敛。

## 项目结构

```
.
├── src/
│   ├── models/        # Transformer模型相关代码
│   ├── utils/         # 工具函数
│   ├── data/          # 数据处理相关代码
│   └── train.py       # 训练脚本
├── requirements.txt   # 项目依赖
└── README.md         # 项目说明
```

## 环境配置

1. 创建并激活conda环境：
```bash
conda create -n transformer python=3.8
conda activate transformer
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 主要创新点

1. Self-Attention机制：允许模型直接建模序列中任意位置之间的关系
2. Multi-Head Attention：通过多个注意力头捕获不同类型的关系
3. Positional Encoding：为模型提供序列位置信息
4. 残差连接和层归一化：帮助训练更深的网络
5. 并行计算能力：相比RNN，可以并行处理整个序列

## 使用说明

待补充... 