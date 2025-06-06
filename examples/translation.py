import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.transformer import Transformer
from src.utils.optimizer import TransformerOptimizer
from src.utils.loss import LabelSmoothingLoss
from src.data.vocab import Vocabulary
from src.data.dataset import TranslationDataset
import os
import json
from tqdm import tqdm

def create_vocab(texts, min_freq=1):
    """创建词汇表
    
    Args:
        texts: 文本列表
        min_freq: 最小词频
        
    Returns:
        词汇表对象
    """
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(texts, min_freq=min_freq)
    return vocab

"""
文本 > 词汇表 > 索引 > 模型输入
"""
def prepare_data():
    """准备示例数据
    
    Returns:
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        train_data: 训练数据
        val_data: 验证数据
    """
    # 示例数据
    train_data = [
        ("hello world", "你好世界"),
        ("how are you", "你好吗"),
        ("good morning", "早上好"),
        ("good night", "晚安"),
        ("thank you", "谢谢"),
        ("you are welcome", "不客气"),
        ("see you tomorrow", "明天见"),
        ("have a nice day", "祝你今天愉快"),
        ("happy birthday", "生日快乐"),
        ("merry christmas", "圣诞快乐")
    ]
    
    val_data = [
        ("hello", "你好"),
        ("goodbye", "再见")
    ]
    
    # 创建词汇表
    src_texts = [pair[0] for pair in train_data]
    tgt_texts = [pair[1] for pair in train_data]
    
    src_vocab = create_vocab(src_texts)
    tgt_vocab = create_vocab(tgt_texts)
    
    return src_vocab, tgt_vocab, train_data, val_data


def get_device():
    """获取可用的设备
    
    Returns:
        torch.device: 可用的设备
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 准备数据
    src_vocab, tgt_vocab, train_data, val_data = prepare_data()
    
    # 创建数据集
    train_dataset = TranslationDataset(
        [x[0] for x in train_data],
        [x[1] for x in train_data],
        src_vocab,
        tgt_vocab
    )
    val_dataset = TranslationDataset(
        [x[0] for x in val_data],
        [x[1] for x in val_data],
        src_vocab,
        tgt_vocab
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=train_dataset.collate_fn # 序列填充，将不同长度的序列填充到相同长度
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        num_layers=3,
        num_heads=4,
        d_ff=512,
        max_seq_length=50,
        dropout=0.1
    ).to(device)
    
    # 先将参数转为 list，避免多次迭代消耗
    params = list(model.parameters())
    
    # 检查模型参数
    print("\nModel parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
            total_params += param.numel()
    print(f"Total trainable parameters: {total_params:,}")
    
    # 创建优化器
    optimizer = TransformerOptimizer(
        params,
        d_model=256,
        warmup_steps=4000,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0
    )
    
    # 创建损失函数
    criterion = LabelSmoothingLoss(
        smoothing=0.1,
        ignore_index=tgt_vocab.get_index("<pad>")
    )
    
    # 训练模型
    num_epochs = 100
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练
        model.train()
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
        # 验证
        model.eval()
        val_loss = evaluate(model, val_dataloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model")
        
        # 测试翻译
        test_sentences = [
            "Hello world",
            "How are you",
            "I love you"
        ]
        
        print("\nTesting translations:")
        for src_text in test_sentences:
            tgt_text = translate(model, src_text, src_vocab, tgt_vocab, device)
            print(f"{src_text} -> {tgt_text}")


def create_masks(src, tgt_input, pad_idx):
    # src_mask: [batch, 1, 1, src_len]
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    # tgt_mask: [batch, 1, tgt_len, tgt_len]，包含下三角mask
    tgt_len = tgt_input.size(1)
    tgt_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(2)
    subsequent_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt_input.device), diagonal=1).bool()
    tgt_mask = tgt_mask & ~subsequent_mask.unsqueeze(0)
    return src_mask, tgt_mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    total_loss = 0
    num_batches = len(dataloader)
    pad_idx = 0  # 假设 <pad> 的 index 为 0
    for batch in tqdm(dataloader, desc="Training"):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
        # 调试输出
        print(f"src.shape: {src.shape}")
        print(f"tgt_input.shape: {tgt_input.shape}")
        print(f"src_mask.shape: {src_mask.shape}")
        print(f"tgt_mask.shape: {tgt_mask.shape}")
        try:
            output = model(src, tgt_input, src_mask, tgt_mask)
            print(f"output.shape: {output.shape}")
        except Exception as e:
            print(f"forward error: {e}")
            raise
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    total_loss = 0
    num_batches = len(dataloader)
    pad_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()
    return total_loss / num_batches


def translate(model, text, src_vocab, tgt_vocab, device):
    """翻译文本
    
    Args:
        model: Transformer 模型
        text: 源文本
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        device: 设备
        
    Returns:
        str: 翻译后的文本
    """
    # 将文本转换为索引
    src_tokens = src_vocab.tokenize(text)
    src = torch.tensor([src_tokens], dtype=torch.long).to(device)
    
    # 生成翻译
    with torch.no_grad():
        output = model.generate(src)
    
    # 将索引转换为文本
    tgt_tokens = output[0].cpu().numpy()
    tgt_text = tgt_vocab.detokenize(tgt_tokens)
    
    return tgt_text


if __name__ == "__main__":
    main() 