"""
文本摘要生成示例

这个示例展示了如何使用 Transformer 模型进行文本摘要生成任务。
我们使用 CNN/Daily Mail 数据集，这是一个新闻文章和摘要的数据集。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import CNNDM
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from src.models.transformer import Transformer
from src.utils.optimizer import AdamW
from src.utils.loss import LabelSmoothingCrossEntropy

# 设置随机种子
torch.manual_seed(42)

# 超参数
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 3e-4
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1
MAX_SRC_LEN = 1024
MAX_TGT_LEN = 128
VOCAB_SIZE = 50000

# 特殊标记
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

# 数据预处理
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for article, _ in data_iter:
        yield tokenizer(article)

# 加载数据集
train_iter = CNNDM(split='train')
test_iter = CNNDM(split='test')

# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_iter), max_tokens=VOCAB_SIZE)
vocab.set_default_index(vocab[UNK_TOKEN])

# 文本处理函数
def text_pipeline(text):
    return vocab(tokenizer(text))

# 数据集类
class SummarizationDataset(Dataset):
    def __init__(self, data_iter, max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN):
        self.data = []
        for article, summary in data_iter:
            src_tokens = text_pipeline(article)
            tgt_tokens = text_pipeline(summary)
            
            # 处理源文本
            if len(src_tokens) > max_src_len:
                src_tokens = src_tokens[:max_src_len]
            else:
                src_tokens = src_tokens + [vocab[PAD_TOKEN]] * (max_src_len - len(src_tokens))
            
            # 处理目标文本
            tgt_tokens = [vocab[BOS_TOKEN]] + tgt_tokens + [vocab[EOS_TOKEN]]
            if len(tgt_tokens) > max_tgt_len:
                tgt_tokens = tgt_tokens[:max_tgt_len]
            else:
                tgt_tokens = tgt_tokens + [vocab[PAD_TOKEN]] * (max_tgt_len - len(tgt_tokens))
            
            self.data.append((torch.tensor(src_tokens), torch.tensor(tgt_tokens)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 创建数据加载器
train_dataset = SummarizationDataset(train_iter)
test_dataset = SummarizationDataset(test_iter)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 创建模型
model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    max_len=MAX_SRC_LEN
)

# 损失函数和优化器
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc='Training'):
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        # 准备目标序列的输入和输出
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.view(-1, VOCAB_SIZE), tgt_output.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# 评估函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, VOCAB_SIZE), tgt_output.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print("开始训练...")
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model, test_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{EPOCHS}:')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'summarization_model.pt')

# 推理函数
def generate_summary(text, max_length=MAX_TGT_LEN):
    model.eval()
    tokens = text_pipeline(text)
    if len(tokens) > MAX_SRC_LEN:
        tokens = tokens[:MAX_SRC_LEN]
    else:
        tokens = tokens + [vocab[PAD_TOKEN]] * (MAX_SRC_LEN - len(tokens))
    
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # 初始化目标序列
    tgt = torch.tensor([[vocab[BOS_TOKEN]]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_length - 1):
            output = model(src, tgt)
            next_token = output[:, -1].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=-1)
            
            if next_token.item() == vocab[EOS_TOKEN]:
                break
    
    # 将生成的标记转换回文本
    id2token = {v: k for k, v in vocab.get_stoi().items()}
    summary_tokens = [id2token[t.item()] for t in tgt[0]]
    
    # 移除特殊标记
    summary_tokens = [t for t in summary_tokens if t not in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]]
    
    return ' '.join(summary_tokens)

# 测试一些文章
test_articles = [
    """
    Apple Inc. announced today that it will be launching a new line of products next month. 
    The company's CEO, Tim Cook, revealed the plans during a virtual event. The new products 
    are expected to revolutionize the tech industry with their innovative features and design. 
    Analysts predict strong sales for the upcoming launch.
    """,
    """
    Scientists at a leading research institute have made a breakthrough in renewable energy technology. 
    The new solar panel design achieves unprecedented efficiency levels while reducing manufacturing costs. 
    This development could accelerate the global transition to clean energy and help combat climate change.
    """,
    """
    The local football team won the championship after an intense final match that went into overtime. 
    The victory marks their first title in over a decade. Fans celebrated throughout the night, 
    and the team's captain dedicated the win to their loyal supporters.
    """
]

print("\n摘要生成示例:")
for article in test_articles:
    print(f"\n原文:\n{article.strip()}")
    summary = generate_summary(article)
    print(f"\n生成的摘要:\n{summary}") 