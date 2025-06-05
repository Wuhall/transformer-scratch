# Transformer 示例和教程

本文档提供了 Transformer 模型的使用示例和教程。

## 快速开始

### 1. 环境配置

1. 创建虚拟环境
   ```bash
   conda create -n transformer python=3.8
   conda activate transformer
   ```

2. 安装依赖
   ```bash
   pip install torch numpy tqdm tensorboard
   ```

### 2. 基本用法

1. 创建模型
   ```python
   from src.models.transformer import Transformer
   
   # 创建模型配置
   config = {
       'd_model': 512,
       'num_heads': 8,
       'num_encoder_layers': 6,
       'num_decoder_layers': 6,
       'd_ff': 2048,
       'dropout': 0.1,
       'max_seq_length': 512
   }
   
   # 创建模型实例
   model = Transformer(config)
   ```

2. 准备数据
   ```python
   from src.data.vocab import Vocabulary
   from src.data.dataset import TranslationDataset
   
   # 创建词汇表
   src_vocab = Vocabulary(min_freq=2)
   tgt_vocab = Vocabulary(min_freq=2)
   
   # 构建词汇表
   src_vocab.build_vocab(src_sentences)
   tgt_vocab.build_vocab(tgt_sentences)
   
   # 创建数据集
   dataset = TranslationDataset(
       src_sentences=src_sentences,
       tgt_sentences=tgt_sentences,
       src_vocab=src_vocab,
       tgt_vocab=tgt_vocab
   )
   ```

3. 训练模型
   ```python
   from src.utils.optimizer import TransformerOptimizer
   from src.utils.loss import LabelSmoothingLoss
   
   # 创建优化器
   optimizer = TransformerOptimizer(
       model.parameters(),
       learning_rate=1e-4,
       warmup_steps=4000
   )
   
   # 创建损失函数
   criterion = LabelSmoothingLoss(smoothing=0.1)
   
   # 训练循环
   for epoch in range(num_epochs):
       model.train()
       for batch in train_loader:
           # 前向传播
           output = model(batch.src, batch.tgt)
           loss = criterion(output, batch.tgt)
           
           # 反向传播
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
   ```

## 高级用法

### 1. 自定义训练

1. 自定义优化器
   ```python
   class CustomOptimizer(TransformerOptimizer):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.custom_param = kwargs.get('custom_param', 0.1)
           
       def step(self):
           # 自定义优化步骤
           for param in self.param_groups[0]['params']:
               if param.grad is not None:
                   param.data.add_(
                       -self.get_lr() * param.grad.data,
                       alpha=-self.custom_param
                   )
   ```

2. 自定义损失函数
   ```python
   class CustomLoss(LabelSmoothingLoss):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.custom_weight = kwargs.get('custom_weight', 1.0)
           
       def forward(self, pred, target):
           # 计算基础损失
           base_loss = super().forward(pred, target)
           
           # 添加自定义损失项
           custom_loss = self.custom_weight * torch.mean(pred ** 2)
           
           return base_loss + custom_loss
   ```

### 2. 模型微调

1. 加载预训练模型
   ```python
   def load_pretrained_model(model_path):
       # 加载模型配置
       config = torch.load(model_path)['config']
       
       # 创建模型实例
       model = Transformer(config)
       
       # 加载权重
       model.load_state_dict(torch.load(model_path)['model_state_dict'])
       
       return model
   ```

2. 微调模型
   ```python
   def finetune_model(model, train_data, num_epochs=5):
       # 冻结部分层
       for param in model.encoder.parameters():
           param.requires_grad = False
           
       # 创建优化器
       optimizer = TransformerOptimizer(
           model.parameters(),
           learning_rate=1e-5
       )
       
       # 微调循环
       for epoch in range(num_epochs):
           model.train()
           for batch in train_data:
               output = model(batch.src, batch.tgt)
               loss = criterion(output, batch.tgt)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
   ```

### 3. 模型导出

1. 导出为 ONNX
   ```python
   def export_to_onnx(model, save_path):
       # 创建示例输入
       dummy_input = torch.randn(1, 10, 512)
       
       # 导出模型
       torch.onnx.export(
           model,
           dummy_input,
           save_path,
           input_names=['input'],
           output_names=['output'],
           dynamic_axes={
               'input': {0: 'batch_size', 1: 'sequence_length'},
               'output': {0: 'batch_size', 1: 'sequence_length'}
           }
       )
   ```

2. 导出为 TorchScript
   ```python
   def export_to_torchscript(model, save_path):
       # 创建示例输入
       dummy_input = torch.randn(1, 10, 512)
       
       # 导出模型
       scripted_model = torch.jit.script(model)
       scripted_model.save(save_path)
   ```

## 实际应用

### 1. 机器翻译

1. 准备数据
   ```python
   def prepare_translation_data(src_file, tgt_file):
       # 读取数据
       with open(src_file, 'r', encoding='utf-8') as f:
           src_sentences = f.readlines()
       with open(tgt_file, 'r', encoding='utf-8') as f:
           tgt_sentences = f.readlines()
           
       # 预处理数据
       src_sentences = [s.strip() for s in src_sentences]
       tgt_sentences = [s.strip() for s in tgt_sentences]
       
       return src_sentences, tgt_sentences
   ```

2. 训练模型
   ```python
   def train_translation_model(model, train_data, val_data):
       # 创建优化器
       optimizer = TransformerOptimizer(
           model.parameters(),
           learning_rate=1e-4
       )
       
       # 创建损失函数
       criterion = LabelSmoothingLoss()
       
       # 训练循环
       for epoch in range(num_epochs):
           # 训练阶段
           model.train()
           train_loss = train_epoch(model, train_data, optimizer, criterion)
           
           # 验证阶段
           model.eval()
           val_loss = evaluate_epoch(model, val_data, criterion)
           
           # 保存检查点
           save_checkpoint(model, optimizer, epoch, train_loss, val_loss)
   ```

3. 模型推理
   ```python
   def translate_sentence(model, sentence, src_vocab, tgt_vocab):
       # 预处理输入
       tokens = tokenize(sentence)
       indices = [src_vocab[token] for token in tokens]
       input_ids = torch.tensor(indices).unsqueeze(0)
       
       # 生成翻译
       with torch.no_grad():
           output_ids = model.generate(input_ids)
           
       # 后处理输出
       output_tokens = [tgt_vocab.idx2word[idx] for idx in output_ids[0]]
       translation = detokenize(output_tokens)
       
       return translation
   ```

### 2. 文本摘要

1. 准备数据
   ```python
   def prepare_summarization_data(text_file, summary_file):
       # 读取数据
       with open(text_file, 'r', encoding='utf-8') as f:
           texts = f.readlines()
       with open(summary_file, 'r', encoding='utf-8') as f:
           summaries = f.readlines()
           
       # 预处理数据
       texts = [t.strip() for t in texts]
       summaries = [s.strip() for s in summaries]
       
       return texts, summaries
   ```

2. 训练模型
   ```python
   def train_summarization_model(model, train_data, val_data):
       # 创建优化器
       optimizer = TransformerOptimizer(
           model.parameters(),
           learning_rate=1e-4
       )
       
       # 创建损失函数
       criterion = LabelSmoothingLoss()
       
       # 训练循环
       for epoch in range(num_epochs):
           # 训练阶段
           model.train()
           train_loss = train_epoch(model, train_data, optimizer, criterion)
           
           # 验证阶段
           model.eval()
           val_loss = evaluate_epoch(model, val_data, criterion)
           
           # 保存检查点
           save_checkpoint(model, optimizer, epoch, train_loss, val_loss)
   ```

3. 模型推理
   ```python
   def generate_summary(model, text, vocab):
       # 预处理输入
       tokens = tokenize(text)
       indices = [vocab[token] for token in tokens]
       input_ids = torch.tensor(indices).unsqueeze(0)
       
       # 生成摘要
       with torch.no_grad():
           output_ids = model.generate(input_ids)
           
       # 后处理输出
       output_tokens = [vocab.idx2word[idx] for idx in output_ids[0]]
       summary = detokenize(output_tokens)
       
       return summary
   ```

## 最佳实践

### 1. 性能优化

1. 批处理优化
   ```python
   def optimize_batch_size(model, data_loader):
       # 测试不同批处理大小
       batch_sizes = [8, 16, 32, 64, 128]
       times = []
       
       for batch_size in batch_sizes:
           # 创建数据加载器
           loader = DataLoader(
               dataset,
               batch_size=batch_size,
               shuffle=True
           )
           
           # 测量时间
           start_time = time.time()
           for batch in loader:
               _ = model(batch.src, batch.tgt)
           end_time = time.time()
           
           times.append(end_time - start_time)
           
       # 选择最佳批处理大小
       best_batch_size = batch_sizes[np.argmin(times)]
       return best_batch_size
   ```

2. 内存优化
   ```python
   def optimize_memory_usage(model, data_loader):
       # 使用梯度检查点
       model.use_checkpointing = True
       
       # 使用混合精度
       scaler = torch.cuda.amp.GradScaler()
       
       # 训练循环
       for batch in data_loader:
           with torch.cuda.amp.autocast():
               output = model(batch.src, batch.tgt)
               loss = criterion(output, batch.tgt)
           
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()
   ```

### 2. 质量优化

1. 模型集成
   ```python
   def ensemble_models(models, input_ids):
       # 收集预测
       predictions = []
       for model in models:
           with torch.no_grad():
               output = model.generate(input_ids)
               predictions.append(output)
               
       # 集成预测
       ensemble_output = torch.stack(predictions).mean(dim=0)
       return ensemble_output
   ```

2. 后处理优化
   ```python
   def postprocess_output(output_ids, vocab):
       # 解码输出
       tokens = [vocab.idx2word[idx] for idx in output_ids]
       
       # 移除重复
       tokens = remove_duplicates(tokens)
       
       # 修复标点
       tokens = fix_punctuation(tokens)
       
       # 合并标记
       text = detokenize(tokens)
       
       return text
   ```

## 常见问题

### 1. 训练问题

1. 过拟合
   - 增加 Dropout
   - 使用权重衰减
   - 提前停止

2. 欠拟合
   - 增加模型容量
   - 调整学习率
   - 增加训练时间

### 2. 推理问题

1. 生成质量
   - 调整温度参数
   - 使用束搜索
   - 实现长度惩罚

2. 推理速度
   - 优化批处理
   - 使用量化
   - 实现缓存

## 参考

- [Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) 