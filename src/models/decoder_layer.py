import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

class DecoderLayer(nn.Module):
    """
    解码器层
    
    创新点：
    1. 使用掩码多头自注意力机制
    2. 使用交叉注意力机制连接编码器和解码器
    3. 使用残差连接和层归一化
    4. 使用前馈神经网络进行特征转换
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        初始化解码器层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            dropout: dropout比率
        """
        super().__init__()
        
        # 掩码多头自注意力
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 交叉注意力
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                self_attn_mask: torch.Tensor = None, 
                cross_attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            enc_output: 编码器输出，形状为 [batch_size, seq_length, d_model]
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码
            
        Returns:
            解码器层的输出
        """
        # 掩码多头自注意力
        attn_output = self.self_attention(x, x, x, self_attn_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 交叉注意力
        attn_output = self.cross_attention(x, enc_output, enc_output, cross_attn_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x 