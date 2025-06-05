import math
import torch
from torch.optim import Optimizer


class TransformerOptimizer(Optimizer):
    """Transformer 优化器，实现学习率预热和衰减"""
    
    def __init__(
        self,
        params,
        d_model: int,
        warmup_steps: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        """初始化优化器
        
        Args:
            params: 模型参数
            d_model: 模型维度
            warmup_steps: 预热步数
            learning_rate: 学习率
            weight_decay: 权重衰减
            max_grad_norm: 最大梯度范数
        """
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self._step = 0
        # 包装 Adam 优化器
        self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        # 兼容 Optimizer 父类
        super().__init__(params, defaults={"lr": learning_rate, "weight_decay": weight_decay})
    
    def step(self, closure=None):
        """执行优化步骤
        
        Args:
            closure: 闭包函数
            
        Returns:
            损失值
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # 更新学习率
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        # 梯度裁剪
        for param_group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group["params"], self.max_grad_norm)
        
        # 更新参数
        self.optimizer.step()
        
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def _get_lr(self) -> float:
        """计算当前学习率
        
        Returns:
            学习率
        """
        step = self._step
        warmup_steps = self.warmup_steps
        
        # 预热阶段
        if step < warmup_steps:
            return self.learning_rate * (step / warmup_steps)
        
        # 衰减阶段
        return self.learning_rate * math.sqrt(
            warmup_steps / step
        ) 