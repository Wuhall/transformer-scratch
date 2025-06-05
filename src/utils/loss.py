import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    
    创新点：
    1. 使用标签平滑技术提高模型泛化能力
    2. 防止模型对训练数据过度自信
    3. 提高模型对未见数据的鲁棒性
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        """
        初始化标签平滑损失函数
        
        Args:
            smoothing: 平滑系数
            ignore_index: 需要忽略的索引（通常是padding token的索引）
        """
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            pred: 模型预测值，形状为 (batch_size * seq_len, vocab_size)
            target: 目标值，形状为 (batch_size * seq_len)
            
        Returns:
            损失值
        """
        # 获取词汇表大小
        vocab_size = pred.size(-1)
        
        # 创建平滑标签
        smooth_target = torch.zeros_like(pred).scatter_(
            1, target.unsqueeze(1), 1.0 - self.smoothing
        )
        smooth_target += self.smoothing / (vocab_size - 1)
        
        # 计算交叉熵损失
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_prob).sum(dim=-1)
        
        # 忽略padding位置
        mask = (target != self.ignore_index).float()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss 