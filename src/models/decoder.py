import torch
import torch.nn as nn
from .components import MultiHeadAttention, FeedForward, PositionalEncoding


class DecoderLayer(nn.Module):
    """解码器层
    
    实现 Transformer 解码器的一个层。
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络维度
        dropout: Dropout 比率
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            enc_output: 编码器输出张量，形状为 [batch_size, seq_length, d_model]
            src_mask: 源序列掩码，形状为 [batch_size, seq_length, seq_length]
            tgt_mask: 目标序列掩码，形状为 [batch_size, seq_length, seq_length]
            
        Returns:
            输出张量，形状为 [batch_size, seq_length, d_model]
        """
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Decoder(nn.Module):
    """解码器
    
    实现 Transformer 解码器。
    
    Args:
        tgt_vocab_size: 目标语言词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: 解码器层数
        d_ff: 前馈网络维度
        max_seq_length: 最大序列长度
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length]
            enc_output: 编码器输出张量，形状为 [batch_size, seq_length, d_model]
            src_mask: 源序列掩码，形状为 [batch_size, seq_length, seq_length]
            tgt_mask: 目标序列掩码，形状为 [batch_size, seq_length, seq_length]
            
        Returns:
            输出张量，形状为 [batch_size, seq_length, tgt_vocab_size]
        """
        # 词嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        # 输出层
        return self.fc_out(x) 