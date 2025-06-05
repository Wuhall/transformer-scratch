import unittest
import torch
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from models.transformer import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        """
        设置测试环境
        """
        self.batch_size = 2
        self.src_seq_length = 10
        self.tgt_seq_length = 8
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.d_model = 512
        self.num_heads = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.d_ff = 2048
        self.max_seq_length = 512
        self.dropout = 0.1
        
        self.model = Transformer(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            d_ff=self.d_ff,
            max_seq_length=self.max_seq_length,
            dropout=self.dropout
        )
    
    def test_forward(self):
        """
        测试前向传播
        """
        # 创建输入张量
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_length))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_length))
        
        # 前向传播
        output = self.model(src, tgt)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.tgt_seq_length, self.tgt_vocab_size)
        self.assertEqual(output.shape, expected_shape)
    
    def test_generate(self):
        """
        测试生成
        """
        # 创建输入张量
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_length))
        
        # 生成翻译
        output = self.model.generate(
            src,
            max_length=self.max_seq_length,
            beam_size=5,
            length_penalty=0.6
        )
        
        # 检查输出形状
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertLessEqual(output.shape[1], self.max_seq_length)
    
    def test_model_parameters(self):
        """
        测试模型参数
        """
        # 检查模型参数数量
        num_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(num_params, 0)
        
        # 检查模型参数是否可训练
        for name, param in self.model.named_parameters():
            self.assertTrue(param.requires_grad)
    
    def test_different_sequence_lengths(self):
        """
        测试不同序列长度
        """
        # 创建不同长度的输入
        src_lengths = [5, 10, 15]
        tgt_lengths = [3, 8, 12]
        
        for src_len, tgt_len in zip(src_lengths, tgt_lengths):
            src = torch.randint(0, self.src_vocab_size, (self.batch_size, src_len))
            tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, tgt_len))
            
            # 前向传播
            output = self.model(src, tgt)
            
            # 检查输出形状
            expected_shape = (self.batch_size, tgt_len, self.tgt_vocab_size)
            self.assertEqual(output.shape, expected_shape)
    
    def test_dropout(self):
        """
        测试dropout
        """
        # 训练模式
        self.model.train()
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_length))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_length))
        
        # 两次前向传播应该产生不同的结果
        output1 = self.model(src, tgt)
        output2 = self.model(src, tgt)
        
        # 检查输出是否不同
        self.assertFalse(torch.allclose(output1, output2))
        
        # 评估模式
        self.model.eval()
        output3 = self.model(src, tgt)
        output4 = self.model(src, tgt)
        
        # 检查输出是否相同
        self.assertTrue(torch.allclose(output3, output4))

if __name__ == '__main__':
    unittest.main() 