import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    创新点：
    1. 使用正弦和余弦函数生成位置编码，而不是可学习的参数
    2. 不同频率的正弦和余弦函数可以捕获不同尺度的位置信息
    3. 位置编码与词嵌入相加，使模型能够同时考虑词义和位置信息
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码模块
        
        Args:
            d_model: 模型的维度
            max_seq_length: 最大序列长度
            dropout: dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为缓冲区（不参与反向传播）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            
        Returns:
            添加了位置编码的张量
        """
        # 将位置编码添加到输入中
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 