import torch
import torch.nn as nn
from .components import PositionalEncoding, MultiHeadAttention, FeedForward


class EncoderLayer(nn.Module):
    """编码器层
    
    实现 Transformer 编码器的一个层。
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络维度
        dropout: Dropout 比率
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    """编码器
    
    实现 Transformer 编码器。
    
    Args:
        src_vocab_size: 源语言词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: 编码器层数
        d_ff: 前馈网络维度
        max_seq_length: 最大序列长度
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 词嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        # 编码器层
        for layer in self.layers:
            x = layer(x, mask)
        return x 