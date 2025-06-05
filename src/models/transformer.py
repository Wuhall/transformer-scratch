import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .encoder import Encoder
from .decoder import Decoder

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """初始化位置编码
        
        Args:
            d_model: 模型维度
            max_seq_length: 最大序列长度
        """
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册缓冲区
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """初始化多头注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            q: 查询张量，形状为 (batch_size, seq_len, d_model)
            k: 键张量，形状为 (batch_size, seq_len, d_model)
            v: 值张量，形状为 (batch_size, seq_len, d_model)
            mask: 掩码张量，形状为 (batch_size, 1, seq_len, seq_len)
            
        Returns:
            注意力输出，形状为 (batch_size, seq_len, d_model)
        """
        batch_size = q.size(0)
        
        # 线性变换
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_linear(out)


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """初始化前馈神经网络
        
        Args:
            d_model: 模型维度
            d_ff: 前馈网络维度
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """初始化编码器层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 掩码张量，形状为 (batch_size, 1, seq_len, seq_len)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """初始化解码器层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            enc_output: 编码器输出，形状为 (batch_size, seq_len, d_model)
            src_mask: 源语言掩码，形状为 (batch_size, 1, seq_len, seq_len)
            tgt_mask: 目标语言掩码，形状为 (batch_size, 1, seq_len, seq_len)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """编码器"""
    
    def __init__(
        self,
        src_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.1
    ):
        """初始化编码器
        
        Args:
            src_vocab_size: 源语言词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            d_ff: 前馈网络维度
            max_seq_length: 最大序列长度
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len)
            mask: 掩码张量，形状为 (batch_size, 1, seq_len, seq_len)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 词嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class Decoder(nn.Module):
    """解码器"""
    
    def __init__(
        self,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.1
    ):
        """初始化解码器
        
        Args:
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 解码器层数
            d_ff: 前馈网络维度
            max_seq_length: 最大序列长度
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len)
            enc_output: 编码器输出，形状为 (batch_size, seq_len, d_model)
            src_mask: 源语言掩码，形状为 (batch_size, 1, seq_len, seq_len)
            tgt_mask: 目标语言掩码，形状为 (batch_size, 1, seq_len, seq_len)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, tgt_vocab_size)
        """
        # 词嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        # 输出层
        x = self.linear(x)
        
        return x


class Transformer(nn.Module):
    """Transformer 模型
    
    实现完整的 Transformer 模型。
    
    Args:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: 编码器和解码器层数
        d_ff: 前馈网络维度
        max_seq_length: 最大序列长度
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            src: 源序列张量，形状为 [batch_size, src_seq_length]
            tgt: 目标序列张量，形状为 [batch_size, tgt_seq_length]
            src_mask: 源序列掩码，形状为 [batch_size, src_seq_length, src_seq_length]
            tgt_mask: 目标序列掩码，形状为 [batch_size, tgt_seq_length, tgt_seq_length]
            
        Returns:
            输出张量，形状为 [batch_size, tgt_seq_length, tgt_vocab_size]
        """
        # 编码器
        enc_output = self.encoder(src, src_mask)
        
        # 解码器
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output
        
    def generate(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        max_length: int = 50,
        beam_size: int = 5,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """生成翻译
        
        使用束搜索生成翻译。
        
        Args:
            src: 源序列张量，形状为 [batch_size, src_seq_length]
            src_mask: 源序列掩码，形状为 [batch_size, src_seq_length, src_seq_length]
            max_length: 最大生成长度
            beam_size: 束搜索大小
            temperature: 采样温度
            
        Returns:
            生成的序列张量，形状为 [batch_size, max_length]
        """
        batch_size = src.size(0)
        device = src.device
        
        # 编码器
        enc_output = self.encoder(src, src_mask)
        
        # 初始化目标序列
        tgt = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        # 束搜索
        for _ in range(max_length - 1):
            # 创建目标掩码
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # 解码器
            output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
            
            # 获取最后一个时间步的输出
            output = output[:, -1, :] / temperature
            
            # 采样下一个词
            probs = torch.softmax(output, dim=-1)
            next_word = torch.multinomial(probs, num_samples=1)
            
            # 添加到目标序列
            tgt = torch.cat([tgt, next_word], dim=1)
            
            # 检查是否生成了结束标记
            if (next_word == 2).all():  # 假设 2 是结束标记
                break
                
        return tgt
        
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成方形后续掩码
        
        Args:
            sz: 序列长度
            
        Returns:
            掩码张量，形状为 [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask 