# Transformer 推理指南

本文档详细介绍了 Transformer 模型的推理过程、优化策略和最佳实践。

## 推理流程

### 1. 模型加载

1. 加载检查点
   ```python
   def load_model(checkpoint_path, config):
       # 创建模型实例
       model = Transformer(config)
       
       # 加载权重
       checkpoint = torch.load(checkpoint_path)
       model.load_state_dict(checkpoint['model_state_dict'])
       
       # 设置为评估模式
       model.eval()
       
       return model
   ```

2. 模型配置
   ```python
   inference_config = {
       'max_length': 100,
       'beam_size': 5,
       'temperature': 1.0,
       'top_k': 50,
       'top_p': 0.9
   }
   ```

### 2. 输入处理

1. 文本预处理
   ```python
   def preprocess_text(text, vocab):
       # 分词
       tokens = tokenize(text)
       
       # 转换为索引
       indices = [vocab[token] for token in tokens]
       
       # 添加特殊标记
       indices = [BOS_IDX] + indices + [EOS_IDX]
       
       # 转换为张量
       return torch.tensor(indices).unsqueeze(0)
   ```

2. 批处理
   ```python
   def batch_inference(model, inputs, batch_size=32):
       results = []
       for i in range(0, len(inputs), batch_size):
           batch = inputs[i:i+batch_size]
           with torch.no_grad():
               outputs = model.generate(batch)
           results.extend(outputs)
       return results
   ```

### 3. 生成策略

1. 贪婪搜索
   ```python
   def greedy_search(model, input_ids):
       output_ids = []
       for _ in range(max_length):
           # 获取预测
           outputs = model(input_ids)
           next_token = outputs.argmax(dim=-1)[:, -1]
           
           # 添加到输出
           output_ids.append(next_token)
           
           # 检查是否结束
           if next_token == EOS_IDX:
               break
               
       return torch.cat(output_ids, dim=-1)
   ```

2. 束搜索
   ```python
   def beam_search(model, input_ids, beam_size=5):
       # 初始化束
       beams = [(input_ids, 0.0)]
       
       for _ in range(max_length):
           candidates = []
           for beam, score in beams:
               # 获取预测
               outputs = model(beam)
               logits = outputs[:, -1, :]
               
               # 获取 top-k 个候选
               values, indices = logits.topk(beam_size)
               
               # 添加到候选列表
               for value, index in zip(values[0], indices[0]):
                   candidates.append((
                       torch.cat([beam, index.unsqueeze(0)], dim=-1),
                       score + value.item()
                   ))
           
           # 选择 top-k 个候选
           beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
           
           # 检查是否所有束都结束
           if all(beam[0][-1] == EOS_IDX for beam in beams):
               break
               
       return beams[0][0]
   ```

3. 采样策略
   ```python
   def sample(model, input_ids, temperature=1.0, top_k=50, top_p=0.9):
       output_ids = []
       for _ in range(max_length):
           # 获取预测
           outputs = model(input_ids)
           logits = outputs[:, -1, :] / temperature
           
           # 应用 top-k 过滤
           if top_k > 0:
               indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
               logits[indices_to_remove] = float('-inf')
           
           # 应用 top-p 过滤
           if top_p < 1.0:
               sorted_logits, sorted_indices = torch.sort(logits, descending=True)
               cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
               sorted_indices_to_remove = cumulative_probs > top_p
               sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
               sorted_indices_to_remove[..., 0] = 0
               indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
               logits[indices_to_remove] = float('-inf')
           
           # 采样下一个标记
           probs = torch.softmax(logits, dim=-1)
           next_token = torch.multinomial(probs, num_samples=1)
           
           # 添加到输出
           output_ids.append(next_token)
           
           # 检查是否结束
           if next_token == EOS_IDX:
               break
               
       return torch.cat(output_ids, dim=-1)
   ```

## 优化策略

### 1. 性能优化

1. 批处理优化
   ```python
   def optimized_batch_inference(model, inputs, batch_size=32):
       # 按长度排序
       sorted_indices = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
       sorted_inputs = [inputs[i] for i in sorted_indices]
       
       # 批处理推理
       results = []
       for i in range(0, len(sorted_inputs), batch_size):
           batch = sorted_inputs[i:i+batch_size]
           with torch.no_grad():
               outputs = model.generate(batch)
           results.extend(outputs)
       
       # 恢复原始顺序
       return [results[sorted_indices.index(i)] for i in range(len(inputs))]
   ```

2. 内存优化
   ```python
   def memory_efficient_inference(model, input_ids):
       # 使用梯度检查点
       with torch.cuda.amp.autocast():
           with torch.no_grad():
               outputs = model.generate(input_ids)
       return outputs
   ```

3. 计算优化
   ```python
   def compute_efficient_inference(model, input_ids):
       # 使用混合精度
       with torch.cuda.amp.autocast():
           with torch.no_grad():
               outputs = model.generate(input_ids)
       return outputs
   ```

### 2. 质量优化

1. 长度惩罚
   ```python
   def length_penalty(length, alpha=0.6):
       return ((5 + length) ** alpha) / (6 ** alpha)
   ```

2. 重复惩罚
   ```python
   def repetition_penalty(logits, output_ids, penalty=1.2):
       for token_id in set(output_ids):
           logits[token_id] /= penalty
       return logits
   ```

3. 温度调整
   ```python
   def temperature_adjustment(logits, temperature=1.0):
       return logits / temperature
   ```

## 最佳实践

### 1. 推理设置

1. 模型配置
   - 使用评估模式
   - 禁用 Dropout
   - 固定随机种子

2. 生成配置
   - 设置最大长度
   - 配置束搜索
   - 调整采样参数

### 2. 错误处理

1. 输入验证
   ```python
   def validate_input(text):
       if not text:
           raise ValueError("输入文本不能为空")
       if len(text) > max_input_length:
           raise ValueError(f"输入文本长度超过限制: {max_input_length}")
   ```

2. 输出验证
   ```python
   def validate_output(output_ids):
       if len(output_ids) == 0:
           raise ValueError("生成结果为空")
       if len(output_ids) > max_output_length:
           raise ValueError(f"生成结果长度超过限制: {max_output_length}")
   ```

### 3. 监控和日志

1. 性能监控
   ```python
   def monitor_inference(model, inputs):
       # 记录开始时间
       start_time = time.time()
       
       # 执行推理
       outputs = model.generate(inputs)
       
       # 计算性能指标
       inference_time = time.time() - start_time
       memory_usage = torch.cuda.max_memory_allocated()
       
       return {
           'outputs': outputs,
           'inference_time': inference_time,
           'memory_usage': memory_usage
       }
   ```

2. 日志记录
   ```python
   def log_inference(inputs, outputs, metrics):
       logger.info({
           'input_length': len(inputs),
           'output_length': len(outputs),
           'inference_time': metrics['inference_time'],
           'memory_usage': metrics['memory_usage']
       })
   ```

## 常见问题

### 1. 性能问题

1. 现象
   - 推理速度慢
   - 内存使用高
   - 批处理效率低

2. 解决方案
   - 优化批处理大小
   - 使用混合精度
   - 实现缓存机制

### 2. 质量问题

1. 现象
   - 生成结果不连贯
   - 重复生成
   - 长度不合适

2. 解决方案
   - 调整生成参数
   - 使用长度惩罚
   - 实现重复惩罚

### 3. 稳定性问题

1. 现象
   - 结果不一致
   - 内存泄漏
   - 异常处理不当

2. 解决方案
   - 固定随机种子
   - 实现内存管理
   - 完善错误处理

## 参考

- [The Curious Case of Neural Text Generation](https://arxiv.org/abs/1801.00632)
- [A Theoretical Analysis of Beam Search](https://arxiv.org/abs/1906.03577)
- [Understanding Neural Text Generation](https://arxiv.org/abs/1903.06259) 