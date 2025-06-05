# Transformer 评估指南

本文档详细介绍了 Transformer 模型的评估方法、指标和最佳实践。

## 评估流程

### 1. 数据准备

1. 测试集划分
   - 从原始数据集中划分
   - 保持数据分布
   - 确保数据质量

2. 数据预处理
   - 与训练集相同的预处理
   - 保持一致性
   - 特殊标记处理

### 2. 评估指标

1. BLEU 分数
   ```python
   def calculate_bleu(predictions, references):
       # 计算 n-gram 匹配
       def get_ngrams(sequence, n):
           return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
       
       # 计算精确率
       def get_precision(pred_ngrams, ref_ngrams):
           matches = sum(1 for ngram in pred_ngrams if ngram in ref_ngrams)
           return matches / len(pred_ngrams) if pred_ngrams else 0
       
       # 计算 BLEU 分数
       bleu_scores = []
       for n in range(1, 5):
           pred_ngrams = get_ngrams(predictions, n)
           ref_ngrams = get_ngrams(references, n)
           precision = get_precision(pred_ngrams, ref_ngrams)
           bleu_scores.append(precision)
           
       # 计算几何平均
       return math.exp(sum(math.log(score) for score in bleu_scores) / len(bleu_scores))
   ```

2. 困惑度
   ```python
   def calculate_perplexity(model, data_loader):
       model.eval()
       total_loss = 0
       total_tokens = 0
       
       with torch.no_grad():
           for batch in data_loader:
               output = model(batch.src, batch.tgt)
               loss = criterion(output, batch.tgt)
               total_loss += loss.item() * batch.tgt.numel()
               total_tokens += batch.tgt.numel()
               
       return math.exp(total_loss / total_tokens)
   ```

3. 准确率
   ```python
   def calculate_accuracy(predictions, references):
       correct = sum(1 for p, r in zip(predictions, references) if p == r)
       return correct / len(predictions)
   ```

### 3. 评估过程

1. 模型推理
   ```python
   def evaluate(model, data_loader):
       model.eval()
       predictions = []
       references = []
       
       with torch.no_grad():
           for batch in data_loader:
               # 生成预测
               output = model.generate(
                   batch.src,
                   max_length=100,
                   beam_size=5
               )
               
               # 收集结果
               predictions.extend(output)
               references.extend(batch.tgt)
               
       return predictions, references
   ```

2. 指标计算
   ```python
   def compute_metrics(predictions, references):
       metrics = {
           'bleu': calculate_bleu(predictions, references),
           'perplexity': calculate_perplexity(predictions, references),
           'accuracy': calculate_accuracy(predictions, references)
       }
       return metrics
   ```

## 评估策略

### 1. 批量评估

1. 批处理大小
   - 根据 GPU 内存调整
   - 考虑序列长度
   - 平衡评估速度

2. 并行评估
   ```python
   def parallel_evaluate(model, data_loader, num_workers=4):
       with torch.multiprocessing.Pool(num_workers) as pool:
           results = pool.map(
               evaluate_batch,
               [(model, batch) for batch in data_loader]
           )
       return results
   ```

### 2. 错误分析

1. 错误类型
   - 词汇错误
   - 语法错误
   - 语义错误

2. 错误统计
   ```python
   def analyze_errors(predictions, references):
       error_types = {
           'vocabulary': 0,
           'grammar': 0,
           'semantics': 0
       }
       
       for pred, ref in zip(predictions, references):
           # 分析错误类型
           if pred not in vocabulary:
               error_types['vocabulary'] += 1
           elif not is_grammatically_correct(pred):
               error_types['grammar'] += 1
           elif not is_semantically_similar(pred, ref):
               error_types['semantics'] += 1
               
       return error_types
   ```

### 3. 性能分析

1. 推理时间
   ```python
   def measure_inference_time(model, data_loader):
       model.eval()
       total_time = 0
       
       with torch.no_grad():
           for batch in data_loader:
               start_time = time.time()
               _ = model(batch.src, batch.tgt)
               end_time = time.time()
               total_time += end_time - start_time
               
       return total_time / len(data_loader)
   ```

2. 内存使用
   ```python
   def measure_memory_usage(model, data_loader):
       model.eval()
       max_memory = 0
       
       with torch.no_grad():
           for batch in data_loader:
               torch.cuda.reset_peak_memory_stats()
               _ = model(batch.src, batch.tgt)
               current_memory = torch.cuda.max_memory_allocated()
               max_memory = max(max_memory, current_memory)
               
       return max_memory
   ```

## 最佳实践

### 1. 评估设置

1. 模型配置
   - 使用评估模式
   - 禁用 Dropout
   - 固定随机种子

2. 数据配置
   - 使用完整测试集
   - 保持数据顺序
   - 记录数据统计

### 2. 结果分析

1. 定量分析
   - 指标统计
   - 性能比较
   - 趋势分析

2. 定性分析
   - 错误分析
   - 案例研究
   - 人工评估

### 3. 报告生成

1. 评估报告
   ```python
   def generate_evaluation_report(metrics, error_analysis, performance):
       report = {
           'metrics': metrics,
           'error_analysis': error_analysis,
           'performance': performance,
           'timestamp': datetime.now().isoformat(),
           'model_version': model.version
       }
       return report
   ```

2. 可视化
   - 指标图表
   - 错误分布
   - 性能曲线

## 常见问题

### 1. 评估偏差

1. 原因
   - 数据分布不均衡
   - 评估指标不全面
   - 测试集不具代表性

2. 解决方案
   - 使用多个指标
   - 进行交叉验证
   - 增加测试集多样性

### 2. 性能问题

1. 现象
   - 评估速度慢
   - 内存使用高
   - 结果不稳定

2. 解决方案
   - 优化批处理
   - 使用混合精度
   - 增加缓存机制

### 3. 结果解释

1. 挑战
   - 指标含义不明确
   - 结果难以比较
   - 结论不具说服力

2. 解决方案
   - 提供详细说明
   - 使用基准比较
   - 进行统计分析

## 参考

- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf)
- [Perplexity: A Measure of Language Model Performance](https://arxiv.org/abs/1906.08237)
- [Evaluation Metrics for Machine Translation](https://www.mt-archive.info/AMTA-2006-Papineni.pdf) 