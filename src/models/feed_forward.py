import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    前馈神经网络
    
    创新点：
    1. 使用两层线性变换和ReLU激活函数
    2. 中间层的维度是输入维度的4倍，提供更大的模型容量
    3. 使用dropout进行正则化
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        """
        初始化前馈神经网络
        
        Args:
            d_model: 输入维度
            d_ff: 中间层维度，默认为4倍的d_model
            dropout: dropout比率
        """
        super().__init__()
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        
        # 定义两层线性变换
        self.linear1 = nn.Linear(d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, d_model)
        
        # 激活函数和dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            
        Returns:
            经过前馈网络处理后的张量
        """
        # 第一层线性变换和激活
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 第二层线性变换
        x = self.linear2(x)
        
        return x 