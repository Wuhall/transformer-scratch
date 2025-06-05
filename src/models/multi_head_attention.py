import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    创新点：
    1. 将注意力机制扩展到多个头，每个头可以关注不同的特征
    2. 通过线性变换将输入投影到不同的子空间
    3. 使用缩放点积注意力机制，提高数值稳定性
    4. 支持自注意力和交叉注意力
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力模块
        
        Args:
            d_model: 模型的维度
            num_heads: 注意力头的数量
            dropout: dropout比率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算缩放点积注意力
        
        Args:
            Q: 查询矩阵
            K: 键矩阵
            V: 值矩阵
            mask: 掩码矩阵
            
        Returns:
            注意力加权的值矩阵
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        return output
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            Q: 查询矩阵，形状为 [batch_size, seq_length, d_model]
            K: 键矩阵，形状为 [batch_size, seq_length, d_model]
            V: 值矩阵，形状为 [batch_size, seq_length, d_model]
            mask: 掩码矩阵
            
        Returns:
            多头注意力的输出
        """
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q)  # [batch_size, seq_length, d_model]
        K = self.W_k(K)  # [batch_size, seq_length, d_model]
        V = self.W_v(V)  # [batch_size, seq_length, d_model]
        
        # 将张量分割成多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终的线性变换
        output = self.W_o(output)
        
        return output 