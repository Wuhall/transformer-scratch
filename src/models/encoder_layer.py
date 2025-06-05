import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

class EncoderLayer(nn.Module):
    """
    编码器层
    
    创新点：
    1. 使用多头自注意力机制
    2. 使用残差连接和层归一化
    3. 使用前馈神经网络进行特征转换
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        初始化编码器层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            dropout: dropout比率
        """
        super().__init__()
        
        # 多头自注意力
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            mask: 掩码张量，用于自注意力
            
        Returns:
            编码器层的输出
        """
        # 多头自注意力
        attn_output = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x 