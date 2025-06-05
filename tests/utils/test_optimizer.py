import unittest
import torch
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from utils.optimizer import TransformerOptimizer
from models.transformer import Transformer

class TestTransformerOptimizer(unittest.TestCase):
    def setUp(self):
        """
        设置测试环境
        """
        # 创建模型
        self.model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            max_seq_length=512,
            dropout=0.1
        )
        
        # 创建优化器
        self.optimizer = TransformerOptimizer(
            model=self.model,
            d_model=512,
            warmup_steps=4000,
            learning_rate=1e-4,
            weight_decay=0.01,
            max_grad_norm=1.0
        )
    
    def test_learning_rate_schedule(self):
        """
        测试学习率调度
        """
        # 检查初始学习率
        initial_lr = self.optimizer.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(initial_lr, 0)
        
        # 模拟训练步骤
        for step in range(1, 4001):
            self.optimizer.step()
            current_lr = self.optimizer.optimizer.param_groups[0]['lr']
            
            # 检查学习率是否在预热阶段增加
            if step <= 4000:
                self.assertGreater(current_lr, 0)
                self.assertLessEqual(current_lr, 1e-4)
            else:
                # 预热后学习率应该开始下降
                self.assertLess(current_lr, 1e-4)
    
    def test_gradient_clipping(self):
        """
        测试梯度裁剪
        """
        # 创建一些梯度
        for param in self.model.parameters():
            param.grad = torch.randn_like(param) * 10  # 创建大梯度
        
        # 应用梯度裁剪
        self.optimizer.clip_gradients()
        
        # 检查梯度范数是否被裁剪
        total_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        self.assertLessEqual(total_norm, 1.0)
    
    def test_weight_decay(self):
        """
        测试权重衰减
        """
        # 获取初始权重
        initial_weights = {}
        for name, param in self.model.named_parameters():
            initial_weights[name] = param.data.clone()
        
        # 执行一步优化
        self.optimizer.step()
        
        # 检查权重是否被衰减
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # 只检查权重参数
                weight_diff = (param.data - initial_weights[name]).abs().mean()
                self.assertGreater(weight_diff, 0)
    
    def test_optimizer_state(self):
        """
        测试优化器状态
        """
        # 检查优化器状态
        self.assertIsNotNone(self.optimizer.optimizer)
        self.assertEqual(len(self.optimizer.optimizer.param_groups), 1)
        
        # 检查参数组
        param_group = self.optimizer.optimizer.param_groups[0]
        self.assertIn('lr', param_group)
        self.assertIn('weight_decay', param_group)
        self.assertEqual(param_group['weight_decay'], 0.01)
    
    def test_step(self):
        """
        测试优化步骤
        """
        # 创建一些梯度
        for param in self.model.parameters():
            param.grad = torch.randn_like(param)
        
        # 记录初始参数
        initial_params = {}
        for name, param in self.model.named_parameters():
            initial_params[name] = param.data.clone()
        
        # 执行优化步骤
        self.optimizer.step()
        
        # 检查参数是否更新
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.allclose(param.data, initial_params[name]))
    
    def test_zero_grad(self):
        """
        测试梯度清零
        """
        # 创建一些梯度
        for param in self.model.parameters():
            param.grad = torch.randn_like(param)
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 检查梯度是否为零
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.allclose(param.grad, torch.zeros_like(param.grad)))

if __name__ == '__main__':
    unittest.main() 