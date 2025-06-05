import unittest
import torch
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from utils.loss import LabelSmoothingLoss

class TestLabelSmoothingLoss(unittest.TestCase):
    def setUp(self):
        """
        设置测试环境
        """
        self.batch_size = 2
        self.seq_length = 10
        self.vocab_size = 1000
        self.smoothing = 0.1
        self.ignore_index = 0
        
        self.criterion = LabelSmoothingLoss(
            smoothing=self.smoothing,
            ignore_index=self.ignore_index
        )
    
    def test_forward(self):
        """
        测试前向传播
        """
        # 创建预测和目标张量
        pred = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        # 计算损失
        loss = self.criterion(pred, target)
        
        # 检查损失是否为标量
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
    
    def test_ignore_index(self):
        """
        测试忽略索引
        """
        # 创建预测和目标张量
        pred = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        # 将一些目标设置为忽略索引
        target[0, 0] = self.ignore_index
        target[1, -1] = self.ignore_index
        
        # 计算损失
        loss = self.criterion(pred, target)
        
        # 检查损失是否为标量
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
    
    def test_smoothing(self):
        """
        测试标签平滑
        """
        # 创建预测和目标张量
        pred = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        # 计算损失
        loss = self.criterion(pred, target)
        
        # 检查损失是否在合理范围内
        self.assertGreater(loss.item(), 0)
        self.assertLess(loss.item(), 10)  # 假设损失不会太大
    
    def test_different_smoothing_values(self):
        """
        测试不同的平滑值
        """
        # 创建预测和目标张量
        pred = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        # 测试不同的平滑值
        smoothing_values = [0.0, 0.1, 0.2, 0.5]
        losses = []
        
        for smoothing in smoothing_values:
            criterion = LabelSmoothingLoss(
                smoothing=smoothing,
                ignore_index=self.ignore_index
            )
            loss = criterion(pred, target)
            losses.append(loss.item())
        
        # 检查不同平滑值的损失是否不同
        self.assertGreater(len(set(losses)), 1)
    
    def test_gradient(self):
        """
        测试梯度
        """
        # 创建预测和目标张量
        pred = torch.randn(self.batch_size, self.seq_length, self.vocab_size, requires_grad=True)
        target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        # 计算损失
        loss = self.criterion(pred, target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否存在
        self.assertIsNotNone(pred.grad)
        self.assertEqual(pred.grad.shape, pred.shape)

if __name__ == '__main__':
    unittest.main() 