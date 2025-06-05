"""
命名实体识别 (NER) 示例

这个示例展示了如何使用 Transformer 模型进行命名实体识别任务。
我们使用 CoNLL-2003 数据集，这是一个包含人名、地名、组织名等实体的标注数据集。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import CoNLL2003
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from src.models.transformer import Transformer
from src.utils.optimizer import AdamW
from src.utils.loss import LabelSmoothingCrossEntropy

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
MAX_LEN = 128
VOCAB_SIZE = 50000

# 标签映射
NER_TAGS = {
    'O': 0,  # 非实体
    'B-PER': 1,  # 人名开始
    'I-PER': 2,  # 人名中间
    'B-ORG': 3,  # 组织名开始
    'I-ORG': 4,  # 组织名中间
    'B-LOC': 5,  # 地名开始
    'I-LOC': 6,  # 地名中间
    'B-MISC': 7,  # 其他实体开始
    'I-MISC': 8,  # 其他实体中间
}

NUM_TAGS = len(NER_TAGS)

# 数据预处理
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

# 加载数据集
train_iter = CoNLL2003(split='train')
test_iter = CoNLL2003(split='test')

# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_iter), max_tokens=VOCAB_SIZE)
vocab.set_default_index(vocab['<unk>'])

# 文本处理函数
def text_pipeline(text):
    return vocab(tokenizer(text))

# 标签处理函数
def tag_pipeline(tag):
    return NER_TAGS[tag]

# 数据集类
class NERDataset(Dataset):
    def __init__(self, data_iter, max_len=MAX_LEN):
        self.data = []
        for text, tags in data_iter:
            tokens = text_pipeline(text)
            tag_ids = [tag_pipeline(tag) for tag in tags]
            
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
                tag_ids = tag_ids[:max_len]
            else:
                tokens = tokens + [vocab['<pad>']] * (max_len - len(tokens))
                tag_ids = tag_ids + [NER_TAGS['O']] * (max_len - len(tag_ids))
            
            self.data.append((torch.tensor(tokens), torch.tensor(tag_ids)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 创建数据加载器
train_dataset = NERDataset(train_iter)
test_dataset = NERDataset(test_iter)

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
    num_classes=NUM_TAGS  # 添加 NER 分类头
)

# 损失函数和优化器
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

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
        loss = criterion(output.view(-1, NUM_TAGS), tgt.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=-1)
        correct += (pred == tgt).sum().item()
        total += tgt.numel()
    
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
            loss = criterion(output.view(-1, NUM_TAGS), tgt.view(-1))
            
            total_loss += loss.item()
            pred = output.argmax(dim=-1)
            correct += (pred == tgt).sum().item()
            total += tgt.numel()
    
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
torch.save(model.state_dict(), 'ner_model.pt')

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
    
    # 将预测结果转换为标签
    id2tag = {v: k for k, v in NER_TAGS.items()}
    tags = [id2tag[p.item()] for p in pred[0]]
    
    # 将标记和标签对齐
    words = tokenizer(text)
    if len(words) > MAX_LEN:
        words = words[:MAX_LEN]
    
    return list(zip(words, tags))

# 测试一些句子
test_sentences = [
    "Apple Inc. is headquartered in Cupertino, California.",
    "Barack Obama was the 44th president of the United States.",
    "The Eiffel Tower is located in Paris, France.",
    "Microsoft Corporation was founded by Bill Gates and Paul Allen."
]

print("\n预测示例:")
for sentence in test_sentences:
    print(f"\n句子: {sentence}")
    predictions = predict(sentence)
    print("实体识别结果:")
    for word, tag in predictions:
        if tag != 'O':
            print(f"{word}: {tag}") 