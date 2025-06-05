from typing import List, Dict, Optional
import torch


class Vocabulary:
    """词汇表类，用于管理词汇和特殊标记"""
    
    def __init__(self, min_freq: int = 1):
        """初始化词汇表"""
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = {}
        
        # 添加特殊标记
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
        self.pad_idx = self.add_word(self.pad_token)
        self.unk_idx = self.add_word(self.unk_token)
        self.bos_idx = self.add_word(self.bos_token)
        self.eos_idx = self.add_word(self.eos_token)
    
    def add_word(self, word: str) -> int:
        """添加单词到词汇表
        
        Args:
            word: 要添加的单词
            
        Returns:
            单词的索引
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1
        return self.word2idx[word]
    
    def add_sentence(self, sentence: str) -> None:
        """添加句子中的所有单词到词汇表
        
        Args:
            sentence: 要添加的句子
        """
        words = sentence.split()
        for word in words:
            self.add_word(word)
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def encode(self, sentence: str) -> List[int]:
        """将句子编码为索引序列
        
        Args:
            sentence: 要编码的句子
            
        Returns:
            索引序列
        """
        words = sentence.split()
        return [self.word2idx.get(word, self.unk_idx) for word in words]
    
    def decode(self, indices: List[int]) -> str:
        """将索引序列解码为句子
        
        Args:
            indices: 要解码的索引序列
            
        Returns:
            解码后的句子
        """
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        return " ".join(words)
    
    def tokenize(self, sentence: str) -> List[int]:
        """将句子转换为模型输入格式
        
        Args:
            sentence: 要转换的句子
            
        Returns:
            包含特殊标记的索引序列
        """
        indices = self.encode(sentence)
        return [self.bos_idx] + indices + [self.eos_idx]
    
    def detokenize(self, indices: List[int]) -> str:
        """将模型输出转换为句子
        
        Args:
            indices: 要转换的索引序列
            
        Returns:
            转换后的句子
        """
        # 移除特殊标记
        if indices[0] == self.bos_idx:
            indices = indices[1:]
        if indices[-1] == self.eos_idx:
            indices = indices[:-1]
        return self.decode(indices)
    
    def save(self, path: str) -> None:
        """保存词汇表到文件
        
        Args:
            path: 保存路径
        """
        torch.save({
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "word_freq": self.word_freq
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """从文件加载词汇表
        
        Args:
            path: 加载路径
            
        Returns:
            加载的词汇表
        """
        vocab = cls()
        data = torch.load(path)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        vocab.word_freq = data["word_freq"]
        return vocab 

    def add_text(self, text: str):
        """将单句文本中的词加入词表（动态构建）"""
        for word in text.strip().split():
            self.word_freq[word] = self.word_freq.get(word, 0) + 1
            if word not in self.word2idx and self.word_freq[word] >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def add_texts(self, texts: list):
        """批量添加多句文本到词表"""
        for text in texts:
            self.add_text(text)

    def build_vocab(self, texts: list, min_freq: int = 1):
        """从文本列表构建词表，支持最小词频"""
        # 统计词频
        freq = {}
        for text in texts:
            for word in text.strip().split():
                freq[word] = freq.get(word, 0) + 1
        # 添加特殊标记
        self.word2idx = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        idx = 4
        for word, count in freq.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        self.word_freq = freq
        self.min_freq = min_freq

    def build(self, *args, **kwargs):
        return self.build_vocab(*args, **kwargs)

    def get_index(self, word: str) -> int:
        """获取词的索引，若不存在则返回 <unk> 的索引"""
        return self.word2idx.get(word, self.word2idx.get("<unk>", 3)) 