# Transformer 架构

本文档详细介绍了 Transformer 模型的架构设计和实现细节。

## 整体架构

Transformer 模型由编码器和解码器两个主要部分组成，每个部分都包含多个相同的层。模型的主要特点包括：

1. 自注意力机制
2. 多头注意力
3. 位置编码
4. 残差连接
5. 层归一化
6. 前馈神经网络

## 编码器

编码器由 N 个相同的层组成，每层包含两个子层：

1. 多头自注意力层
2. 前馈神经网络层

每个子层都使用残差连接和层归一化。

### 多头自注意力

多头自注意力机制允许模型同时关注输入序列的不同位置，每个头可以学习不同的特征表示。具体实现包括：

- 缩放点积注意力
- 多头投影
- 残差连接
- 层归一化

### 前馈神经网络

前馈神经网络是一个两层的全连接网络，使用 ReLU 激活函数：

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

## 解码器

解码器也由 N 个相同的层组成，每层包含三个子层：

1. 掩码多头自注意力层
2. 多头编码器-解码器注意力层
3. 前馈神经网络层

每个子层同样使用残差连接和层归一化。

### 掩码多头自注意力

解码器的自注意力层使用掩码来防止模型看到未来的信息，确保预测时只能使用已知的信息。

### 编码器-解码器注意力

编码器-解码器注意力层允许解码器关注输入序列的相关部分，实现源语言和目标语言之间的对齐。

## 位置编码

由于 Transformer 不包含循环或卷积结构，需要额外的位置信息。我们使用正弦和余弦函数来生成位置编码：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## 模型配置

模型的主要超参数包括：

- `d_model`: 模型维度（默认：512）
- `num_heads`: 注意力头数（默认：8）
- `num_encoder_layers`: 编码器层数（默认：6）
- `num_decoder_layers`: 解码器层数（默认：6）
- `d_ff`: 前馈网络维度（默认：2048）
- `dropout`: Dropout 比率（默认：0.1）
- `max_seq_length`: 最大序列长度（默认：512）

## 创新特性

1. 可配置的模型维度、头数和层数
   - 支持不同规模的模型配置
   - 可以根据任务需求调整模型容量

2. 支持不同的位置编码方式
   - 正弦位置编码
   - 可扩展支持其他编码方式

3. 灵活的注意力机制
   - 可配置的注意力头数
   - 支持不同的注意力变体

## 实现细节

### 注意力计算

```python
def attention(query, key, value, mask=None):
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 应用 softmax
    attn = torch.softmax(scores, dim=-1)
    
    # 应用 dropout
    attn = self.dropout(attn)
    
    # 计算输出
    output = torch.matmul(attn, value)
    
    return output
```

### 多头注意力

```python
def multi_head_attention(query, key, value, mask=None):
    # 线性投影
    q = self.w_q(query)
    k = self.w_k(key)
    v = self.w_v(value)
    
    # 分割头
    q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    # 计算注意力
    output = self.attention(q, k, v, mask)
    
    # 合并头
    output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    # 线性投影
    output = self.w_o(output)
    
    return output
```

### 位置编码

```python
def positional_encoding(position, d_model):
    # 创建位置编码矩阵
    pos_encoding = torch.zeros(position, d_model)
    position = torch.arange(0, position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # 计算正弦和余弦
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding
```

## 性能优化

1. 批处理优化
   - 动态批处理大小
   - 序列长度填充优化

2. 内存优化
   - 梯度检查点
   - 混合精度训练

3. 计算优化
   - 矩阵运算优化
   - 注意力计算优化

## 扩展性

模型设计支持以下扩展：

1. 新的注意力机制
2. 不同的位置编码方式
3. 自定义的层结构
4. 不同的优化策略


## 代码结构

1. **整体架构** (`src/models/transformer.py`)
```python
class Transformer:
    def __init__():
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
    
    def forward():  # 训练过程
        # 1. 编码器处理源序列
        enc_output = self.encoder(src, src_mask)
        # 2. 解码器处理目标序列
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
    
    def generate():  # 推理过程
        # 1. 编码器处理源序列
        enc_output = self.encoder(src, src_mask)
        # 2. 初始化解码器输入
        tgt = torch.ones(batch_size, 1)  # 起始标记
        # 3. 自回归生成
        for _ in range(max_length - 1):
            output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
            next_word = sample(output)
            tgt = torch.cat([tgt, next_word], dim=1)
```

2. **编码器结构** (`src/models/encoder.py`)
```python
class Encoder:
    def __init__():
        # 1. 词嵌入层
        self.embedding = nn.Embedding(...)
        # 2. 位置编码层
        self.pos_encoding = PositionalEncoding(...)
        # 3. 编码器层堆叠
        self.layers = nn.ModuleList([EncoderLayer(...) for _ in range(num_layers)])
    
    def forward():
        # 1. 词嵌入
        x = self.embedding(x)
        # 2. 位置编码
        x = self.pos_encoding(x)
        # 3. 编码器层
        for layer in self.layers:
            x = layer(x, mask)
```

3. **解码器结构** (`src/models/decoder.py`)
```python
class Decoder:
    def __init__():
        # 1. 词嵌入层
        self.embedding = nn.Embedding(...)
        # 2. 位置编码层
        self.pos_encoding = PositionalEncoding(...)
        # 3. 解码器层堆叠
        self.layers = nn.ModuleList([DecoderLayer(...) for _ in range(num_layers)])
        # 4. 输出层
        self.fc_out = nn.Linear(...)
    
    def forward():
        # 1. 词嵌入
        x = self.embedding(x)
        # 2. 位置编码
        x = self.pos_encoding(x)
        # 3. 解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        # 4. 输出层
        x = self.fc_out(x)
```

4. **编码器层** (`src/models/encoder_layer.py`)
```python
class EncoderLayer:
    def __init__():
        # 1. 自注意力层
        self.self_attention = MultiHeadAttention(...)
        # 2. 前馈网络
        self.feed_forward = FeedForward(...)
        # 3. 层归一化
        self.norm1 = LayerNorm(...)
        self.norm2 = LayerNorm(...)
    
    def forward():
        # 1. 自注意力
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
```

5. **解码器层** (`src/models/decoder_layer.py`)
```python
class DecoderLayer:
    def __init__():
        # 1. 掩码自注意力层
        self.self_attention = MultiHeadAttention(...)
        # 2. 交叉注意力层
        self.cross_attention = MultiHeadAttention(...)
        # 3. 前馈网络
        self.feed_forward = FeedForward(...)
        # 4. 层归一化
        self.norm1 = LayerNorm(...)
        self.norm2 = LayerNorm(...)
        self.norm3 = LayerNorm(...)
    
    def forward():
        # 1. 掩码自注意力
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + attn_output)
        # 2. 交叉注意力
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + attn_output)
        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
```

6. **注意力机制** (`src/models/multi_head_attention.py`)
```python
class MultiHeadAttention:
    def __init__():
        # 1. 线性变换层
        self.W_q = nn.Linear(...)
        self.W_k = nn.Linear(...)
        self.W_v = nn.Linear(...)
        self.W_o = nn.Linear(...)
    
    def forward():
        # 1. 线性变换
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        # 2. 分割多头
        Q = Q.view(batch_size, -1, num_heads, d_k)
        K = K.view(batch_size, -1, num_heads, d_k)
        V = V.view(batch_size, -1, num_heads, d_k)
        # 3. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
        attn = softmax(scores)
        output = torch.matmul(attn, V)
        # 4. 合并多头
        output = output.view(batch_size, -1, d_model)
        # 5. 输出变换
        output = self.W_o(output)
```

**训练和推理的主要区别**：

1. **训练过程**：
   - 编码器和解码器同步执行
   - 输入完整的源序列和目标序列
   - 使用教师强制（Teacher Forcing）
   - 可以并行计算所有时间步

2. **推理过程**：
   - 编码器先执行，得到编码表示
   - 解码器自回归生成，每次生成一个词
   - 使用掩码确保只能看到当前位置之前的信息
   - 需要串行计算每个时间步

**数据流向**：

1. **训练时**：
```
输入序列 -> 编码器 -> 编码表示
目标序列 -> 解码器 -> 输出序列
```

2. **推理时**：
```
输入序列 -> 编码器 -> 编码表示
起始标记 -> 解码器 -> 第一个词
[起始标记, 第一个词] -> 解码器 -> 第二个词
[起始标记, 第一个词, 第二个词] -> 解码器 -> 第三个词
...
```

结构方法论：
1. 理解模型的层次结构
2. 了解各个组件的作用
3. 掌握训练和推理的区别
4. 理解数据流向

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) 