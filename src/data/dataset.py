from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from .vocab import Vocabulary


class TranslationDataset(Dataset):
    """翻译数据集类"""
    
    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_seq_length: int = 50
    ):
        """初始化数据集
        
        Args:
            src_sentences: 源语言句子列表
            tgt_sentences: 目标语言句子列表
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            max_seq_length: 最大序列长度
        """
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_seq_length = max_seq_length
        
        # 预处理数据
        self.data = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_tokens = src_vocab.tokenize(src)
            tgt_tokens = tgt_vocab.tokenize(tgt)
            
            # 截断过长的序列
            if len(src_tokens) > max_seq_length:
                src_tokens = src_tokens[:max_seq_length]
            if len(tgt_tokens) > max_seq_length:
                tgt_tokens = tgt_tokens[:max_seq_length]
            
            self.data.append({
                "src": src_tokens,
                "tgt": tgt_tokens
            })
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            数据样本
        """
        sample = self.data[idx]
        
        # 转换为张量
        src = torch.tensor(sample["src"], dtype=torch.long)
        tgt = torch.tensor(sample["tgt"], dtype=torch.long)
        
        # 创建掩码
        src_mask = (src != self.src_vocab.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != self.tgt_vocab.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 创建目标掩码
        seq_len = tgt.size(0)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        
        return {
            "src": src,
            "tgt": tgt,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask
        }
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """批处理函数
        
        Args:
            batch: 批次数据
            
        Returns:
            处理后的批次数据
        """
        # 获取批次大小
        batch_size = len(batch)
        
        # 获取最大序列长度
        src_max_len = max(x["src"].size(0) for x in batch)
        tgt_max_len = max(x["tgt"].size(0) for x in batch)
        
        # 创建填充后的张量
        src = torch.full((batch_size, src_max_len), self.src_vocab.pad_idx, dtype=torch.long)
        tgt = torch.full((batch_size, tgt_max_len), self.tgt_vocab.pad_idx, dtype=torch.long)
        
        # 填充数据
        for i, sample in enumerate(batch):
            src_len = sample["src"].size(0)
            tgt_len = sample["tgt"].size(0)
            
            src[i, :src_len] = sample["src"]
            tgt[i, :tgt_len] = sample["tgt"]
        
        # 创建掩码
        src_mask = (src != self.src_vocab.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != self.tgt_vocab.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 创建目标掩码
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        
        return {
            "src": src,
            "tgt": tgt,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask
        } 