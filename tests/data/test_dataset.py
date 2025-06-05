import unittest
import torch
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from data.dataset import TranslationDataset, collate_fn
from data.vocab import Vocabulary

class TestTranslationDataset(unittest.TestCase):
    def setUp(self):
        """
        设置测试环境
        """
        # 创建词汇表
        self.src_vocab = Vocabulary(min_freq=1)
        self.tgt_vocab = Vocabulary(min_freq=1)
        
        # 构建词汇表
        self.src_sentences = [
            ['hello', 'world'],
            ['hello', 'world', 'test'],
            ['test', 'test', 'test']
        ]
        self.tgt_sentences = [
            ['你好', '世界'],
            ['你好', '世界', '测试'],
            ['测试', '测试', '测试']
        ]
        
        self.src_vocab.build_vocab(self.src_sentences)
        self.tgt_vocab.build_vocab(self.tgt_sentences)
        
        # 创建数据集
        self.dataset = TranslationDataset(
            src_sentences=self.src_sentences,
            tgt_sentences=self.tgt_sentences,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            max_seq_length=10
        )
    
    def test_len(self):
        """
        测试数据集长度
        """
        self.assertEqual(len(self.dataset), len(self.src_sentences))
    
    def test_getitem(self):
        """
        测试获取样本
        """
        # 获取第一个样本
        src, tgt = self.dataset[0]
        
        # 检查源语言
        self.assertIsInstance(src, torch.Tensor)
        self.assertEqual(src.dim(), 1)
        self.assertLessEqual(src.size(0), 10)  # 最大序列长度
        
        # 检查目标语言
        self.assertIsInstance(tgt, torch.Tensor)
        self.assertEqual(tgt.dim(), 1)
        self.assertLessEqual(tgt.size(0), 10)  # 最大序列长度
        
        # 检查特殊标记
        self.assertEqual(src[0].item(), self.src_vocab.bos_idx)
        self.assertEqual(src[-1].item(), self.src_vocab.eos_idx)
        self.assertEqual(tgt[0].item(), self.tgt_vocab.bos_idx)
        self.assertEqual(tgt[-1].item(), self.tgt_vocab.eos_idx)
    
    def test_collate_fn(self):
        """
        测试批处理函数
        """
        # 创建批次
        batch = [self.dataset[i] for i in range(3)]
        src_batch, tgt_batch = collate_fn(batch)
        
        # 检查批次形状
        self.assertEqual(src_batch.size(0), 3)  # 批次大小
        self.assertEqual(tgt_batch.size(0), 3)  # 批次大小
        
        # 检查填充
        self.assertEqual(src_batch.size(1), max(len(src) for src, _ in batch))
        self.assertEqual(tgt_batch.size(1), max(len(tgt) for _, tgt in batch))
    
    def test_max_seq_length(self):
        """
        测试最大序列长度
        """
        # 创建长序列
        long_src = ['word'] * 20
        long_tgt = ['词'] * 20
        
        # 创建数据集
        dataset = TranslationDataset(
            src_sentences=[long_src],
            tgt_sentences=[long_tgt],
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            max_seq_length=10
        )
        
        # 获取样本
        src, tgt = dataset[0]
        
        # 检查序列长度
        self.assertLessEqual(src.size(0), 10)
        self.assertLessEqual(tgt.size(0), 10)
    
    def test_special_tokens(self):
        """
        测试特殊标记
        """
        # 获取样本
        src, tgt = self.dataset[0]
        
        # 检查源语言特殊标记
        self.assertEqual(src[0].item(), self.src_vocab.bos_idx)
        self.assertEqual(src[-1].item(), self.src_vocab.eos_idx)
        
        # 检查目标语言特殊标记
        self.assertEqual(tgt[0].item(), self.tgt_vocab.bos_idx)
        self.assertEqual(tgt[-1].item(), self.tgt_vocab.eos_idx)
    
    def test_unknown_tokens(self):
        """
        测试未知标记
        """
        # 创建包含未知词的句子
        unknown_src = ['unknown_word']
        unknown_tgt = ['未知词']
        
        # 创建数据集
        dataset = TranslationDataset(
            src_sentences=[unknown_src],
            tgt_sentences=[unknown_tgt],
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            max_seq_length=10
        )
        
        # 获取样本
        src, tgt = dataset[0]
        
        # 检查未知标记
        self.assertEqual(src[1].item(), self.src_vocab.unk_idx)
        self.assertEqual(tgt[1].item(), self.tgt_vocab.unk_idx)

if __name__ == '__main__':
    unittest.main() 