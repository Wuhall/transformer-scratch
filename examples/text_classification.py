"""
文本分类示例

这个示例展示了如何使用 Transformer 模型进行文本分类任务。
我们使用 IMDB 电影评论数据集，这是一个二分类任务（正面/负面评论）。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from src.models.transformer import Transformer
# from src.utils.loss import LabelSmoothingCrossEntropy

# 设置随机种子
torch.manual_seed(42)

# 超参数
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-4
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
D_FF = 1024
DROPOUT = 0.1
MAX_LEN = 512
VOCAB_SIZE = 50000
NUM_CLASSES = 2

# 数据预处理
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

# 加载数据集
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_iter), max_tokens=VOCAB_SIZE)
vocab.set_default_index(vocab['<unk>'])

# 文本处理函数
def text_pipeline(text):
    return vocab(tokenizer(text))

# 标签处理函数
def label_pipeline(label):
    return int(label) - 1  # 将 1,2 转换为 0,1

# 数据集类
class IMDBDataset(Dataset):
    def __init__(self, data_iter, max_len=MAX_LEN):
        self.data = []
        for text, label in data_iter:
            tokens = text_pipeline(text)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [vocab['<pad>']] * (max_len - len(tokens))
            self.data.append((torch.tensor(tokens), torch.tensor(label_pipeline(label))))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 创建数据加载器
train_dataset = IMDBDataset(train_iter)
test_dataset = IMDBDataset(test_iter)

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
    max_len=MAX_LEN,
    num_classes=NUM_CLASSES  # 添加分类头
)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc='Training'):
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=-1)
        correct += (pred == tgt).sum().item()
        total += tgt.size(0)
    
    return total_loss / len(loader), correct / total

# 评估函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src)
            loss = criterion(output, tgt)
            
            total_loss += loss.item()
            pred = output.argmax(dim=-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    
    return total_loss / len(loader), correct / total

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print("开始训练...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{EPOCHS}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# 保存模型
torch.save(model.state_dict(), 'text_classification_model.pt')

# 推理示例
def predict(text):
    model.eval()
    tokens = text_pipeline(text)
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
    else:
        tokens = tokens + [vocab['<pad>']] * (MAX_LEN - len(tokens))
    
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(src)
        pred = output.argmax(dim=-1)
    
    return "正面" if pred.item() == 1 else "负面"

# 测试一些评论
test_reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "What a waste of time. The plot was confusing and the acting was terrible.",
    "A masterpiece of modern cinema. The director's vision is perfectly executed.",
    "I couldn't wait for it to end. The characters were poorly developed."
]

print("\n预测示例:")
for review in test_reviews:
    sentiment = predict(review)
    print(f"评论: {review}")
    print(f"情感: {sentiment}\n") 