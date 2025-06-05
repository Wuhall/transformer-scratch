import unittest
import sys
import os
import json
import tempfile

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from data.vocab import Vocabulary

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        """
        设置测试环境
        """
        self.min_freq = 2
        self.vocab = Vocabulary(min_freq=self.min_freq)
        
        # 测试数据
        self.sentences = [
            ['hello', 'world', 'hello', 'world'],
            ['hello', 'world', 'test', 'test'],
            ['hello', 'test', 'test', 'test']
        ]
    
    def test_build_vocab(self):
        """
        测试构建词汇表
        """
        # 构建词汇表
        self.vocab.build_vocab(self.sentences)
        
        # 检查特殊标记
        self.assertEqual(self.vocab.word2idx['<pad>'], 0)
        self.assertEqual(self.vocab.word2idx['<bos>'], 1)
        self.assertEqual(self.vocab.word2idx['<eos>'], 2)
        self.assertEqual(self.vocab.word2idx['<unk>'], 3)
        
        # 检查词汇表大小
        self.assertGreater(len(self.vocab), 4)  # 大于特殊标记数量
        
        # 检查词频
        self.assertGreaterEqual(self.vocab.word_freq['hello'], self.min_freq)
        self.assertGreaterEqual(self.vocab.word_freq['world'], self.min_freq)
        self.assertGreaterEqual(self.vocab.word_freq['test'], self.min_freq)
    
    def test_encode(self):
        """
        测试编码
        """
        # 构建词汇表
        self.vocab.build_vocab(self.sentences)
        
        # 测试编码
        sentence = ['hello', 'world', 'unknown']
        indices = self.vocab.encode(sentence)
        
        # 检查编码结果
        self.assertEqual(len(indices), len(sentence))
        self.assertEqual(indices[0], self.vocab.word2idx['hello'])
        self.assertEqual(indices[1], self.vocab.word2idx['world'])
        self.assertEqual(indices[2], self.vocab.unk_idx)
    
    def test_decode(self):
        """
        测试解码
        """
        # 构建词汇表
        self.vocab.build_vocab(self.sentences)
        
        # 测试解码
        indices = [self.vocab.word2idx['hello'], self.vocab.word2idx['world'], self.vocab.unk_idx]
        words = self.vocab.decode(indices)
        
        # 检查解码结果
        self.assertEqual(len(words), len(indices))
        self.assertEqual(words[0], 'hello')
        self.assertEqual(words[1], 'world')
        self.assertEqual(words[2], '<unk>')
    
    def test_save_load(self):
        """
        测试保存和加载
        """
        # 构建词汇表
        self.vocab.build_vocab(self.sentences)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存词汇表
            self.vocab.save(temp_path)
            
            # 加载词汇表
            loaded_vocab = Vocabulary.load(temp_path)
            
            # 检查词汇表是否相同
            self.assertEqual(self.vocab.word2idx, loaded_vocab.word2idx)
            self.assertEqual(self.vocab.idx2word, loaded_vocab.idx2word)
            self.assertEqual(self.vocab.word_freq, loaded_vocab.word_freq)
            self.assertEqual(self.vocab.min_freq, loaded_vocab.min_freq)
        
        finally:
            # 删除临时文件
            os.unlink(temp_path)
    
    def test_special_tokens(self):
        """
        测试特殊标记
        """
        # 检查特殊标记索引
        self.assertEqual(self.vocab.pad_idx, 0)
        self.assertEqual(self.vocab.bos_idx, 1)
        self.assertEqual(self.vocab.eos_idx, 2)
        self.assertEqual(self.vocab.unk_idx, 3)
        
        # 检查特殊标记词
        self.assertEqual(self.vocab.idx2word[0], '<pad>')
        self.assertEqual(self.vocab.idx2word[1], '<bos>')
        self.assertEqual(self.vocab.idx2word[2], '<eos>')
        self.assertEqual(self.vocab.idx2word[3], '<unk>')
    
    def test_empty_vocab(self):
        """
        测试空词汇表
        """
        # 检查空词汇表
        self.assertEqual(len(self.vocab), 4)  # 只有特殊标记
        self.assertEqual(self.vocab.encode(['unknown'])[0], self.vocab.unk_idx)
        self.assertEqual(self.vocab.decode([self.vocab.unk_idx])[0], '<unk>')

if __name__ == '__main__':
    unittest.main() 