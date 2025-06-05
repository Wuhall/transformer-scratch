import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    层归一化模块
    
    创新点：
    1. 对每个样本的特征维度进行归一化，而不是批次维度
    2. 引入可学习的缩放和平移参数
    3. 帮助稳定深层网络的训练
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        初始化层归一化模块
        
        Args:
            d_model: 特征维度
            eps: 数值稳定性参数
        """
        super().__init__()
        self.eps = eps
        
        # 可学习的参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_length, d_model]
            
        Returns:
            归一化后的张量
        """
        # 计算均值
        mean = x.mean(dim=-1, keepdim=True)
        
        # 计算方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        x = self.gamma * x + self.beta
        
        return x 