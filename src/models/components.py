import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码层
    
    使用正弦和余弦函数生成位置编码。
    
    Args:
        d_model: 模型维度
        max_seq_length: 最大序列长度
        dropout: Dropout 比率
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与反向传播）
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力层
    
    实现多头自注意力机制。
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: Dropout 比率
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            q: 查询张量，形状为 [batch_size, seq_length, d_model]
            k: 键张量，形状为 [batch_size, seq_length, d_model]
            v: 值张量，形状为 [batch_size, seq_length, d_model]
            mask: 掩码张量，形状为 [batch_size, seq_length, seq_length]
            
        Returns:
            注意力输出张量，形状为 [batch_size, seq_length, d_model]
        """
        batch_size = q.size(0)
        
        # 线性变换
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用 softmax 和 dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(out)


class FeedForward(nn.Module):
    """前馈神经网络
    
    实现 Transformer 中的前馈神经网络。
    
    Args:
        d_model: 模型维度
        d_ff: 前馈网络维度
        dropout: Dropout 比率
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            
        Returns:
            输出张量，形状为 [batch_size, seq_length, d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x)))) 