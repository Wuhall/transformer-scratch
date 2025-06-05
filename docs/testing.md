# Transformer 测试指南

本文档详细介绍了 Transformer 模型的测试策略、方法和最佳实践。

## 测试策略

### 1. 单元测试

1. 模型组件测试
   ```python
   class TestTransformerComponents(unittest.TestCase):
       def setUp(self):
           self.config = {
               'd_model': 512,
               'num_heads': 8,
               'd_ff': 2048,
               'dropout': 0.1
           }
           
       def test_multi_head_attention(self):
           # 创建多头注意力层
           mha = MultiHeadAttention(**self.config)
           
           # 创建输入
           batch_size = 2
           seq_length = 10
           x = torch.randn(batch_size, seq_length, self.config['d_model'])
           
           # 测试前向传播
           output = mha(x, x, x)
           
           # 验证输出形状
           self.assertEqual(
               output.shape,
               (batch_size, seq_length, self.config['d_model'])
           )
           
       def test_feed_forward(self):
           # 创建前馈网络
           ff = FeedForward(**self.config)
           
           # 创建输入
           batch_size = 2
           seq_length = 10
           x = torch.randn(batch_size, seq_length, self.config['d_model'])
           
           # 测试前向传播
           output = ff(x)
           
           # 验证输出形状
           self.assertEqual(
               output.shape,
               (batch_size, seq_length, self.config['d_model'])
           )
   ```

2. 数据处理测试
   ```python
   class TestDataProcessing(unittest.TestCase):
       def setUp(self):
           self.vocab = Vocabulary(min_freq=2)
           self.dataset = TranslationDataset(
               src_sentences=['hello world', 'good morning'],
               tgt_sentences=['你好世界', '早上好'],
               src_vocab=self.vocab,
               tgt_vocab=self.vocab
           )
           
       def test_vocabulary(self):
           # 测试词汇表构建
           self.vocab.build_vocab(['hello world', 'good morning'])
           
           # 验证特殊标记
           self.assertEqual(self.vocab['<pad>'], 0)
           self.assertEqual(self.vocab['<bos>'], 1)
           self.assertEqual(self.vocab['<eos>'], 2)
           self.assertEqual(self.vocab['<unk>'], 3)
           
       def test_dataset(self):
           # 测试数据集
           sample = self.dataset[0]
           
           # 验证样本格式
           self.assertIsInstance(sample['src'], torch.Tensor)
           self.assertIsInstance(sample['tgt'], torch.Tensor)
           self.assertEqual(sample['src'].shape[0], sample['tgt'].shape[0])
   ```

3. 优化器测试
   ```python
   class TestOptimizer(unittest.TestCase):
       def setUp(self):
           self.model = Transformer(config)
           self.optimizer = TransformerOptimizer(
               self.model.parameters(),
               learning_rate=1e-4,
               warmup_steps=4000
           )
           
       def test_learning_rate_schedule(self):
           # 测试学习率调度
           initial_lr = self.optimizer.get_lr()
           
           # 模拟训练步骤
           for step in range(1000):
               self.optimizer.step()
               
           # 验证学习率变化
           current_lr = self.optimizer.get_lr()
           self.assertNotEqual(initial_lr, current_lr)
           
       def test_gradient_clipping(self):
           # 测试梯度裁剪
           loss = self.model(torch.randn(2, 10, 512))
           loss.backward()
           
           # 验证梯度范数
           grad_norm = torch.nn.utils.clip_grad_norm_(
               self.model.parameters(),
               max_norm=1.0
           )
           self.assertLessEqual(grad_norm, 1.0)
   ```

### 2. 集成测试

1. 模型训练测试
   ```python
   class TestModelTraining(unittest.TestCase):
       def setUp(self):
           self.config = {
               'd_model': 512,
               'num_heads': 8,
               'num_encoder_layers': 6,
               'num_decoder_layers': 6,
               'd_ff': 2048,
               'dropout': 0.1
           }
           self.model = Transformer(self.config)
           self.optimizer = TransformerOptimizer(
               self.model.parameters(),
               learning_rate=1e-4
           )
           self.criterion = LabelSmoothingLoss()
           
       def test_training_step(self):
           # 创建输入
           src = torch.randn(2, 10, 512)
           tgt = torch.randn(2, 10, 512)
           
           # 执行训练步骤
           self.model.train()
           output = self.model(src, tgt)
           loss = self.criterion(output, tgt)
           
           # 反向传播
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()
           
           # 验证损失
           self.assertIsInstance(loss.item(), float)
           self.assertGreater(loss.item(), 0)
   ```

2. 模型推理测试
   ```python
   class TestModelInference(unittest.TestCase):
       def setUp(self):
           self.model = Transformer(config)
           self.model.eval()
           
       def test_inference(self):
           # 创建输入
           src = torch.randn(2, 10, 512)
           
           # 执行推理
           with torch.no_grad():
               output = self.model.generate(
                   src,
                   max_length=100,
                   beam_size=5
               )
           
           # 验证输出
           self.assertIsInstance(output, torch.Tensor)
           self.assertEqual(output.shape[0], src.shape[0])
   ```

### 3. 性能测试

1. 训练性能测试
   ```python
   class TestTrainingPerformance(unittest.TestCase):
       def setUp(self):
           self.model = Transformer(config)
           self.optimizer = TransformerOptimizer(
               self.model.parameters(),
               learning_rate=1e-4
           )
           
       def test_training_speed(self):
           # 记录开始时间
           start_time = time.time()
           
           # 执行训练
           for _ in range(100):
               src = torch.randn(32, 10, 512)
               tgt = torch.randn(32, 10, 512)
               output = self.model(src, tgt)
               loss = self.criterion(output, tgt)
               self.optimizer.zero_grad()
               loss.backward()
               self.optimizer.step()
           
           # 计算训练速度
           training_time = time.time() - start_time
           steps_per_second = 100 / training_time
           
           # 验证性能
           self.assertGreater(steps_per_second, 10)
   ```

2. 推理性能测试
   ```python
   class TestInferencePerformance(unittest.TestCase):
       def setUp(self):
           self.model = Transformer(config)
           self.model.eval()
           
       def test_inference_speed(self):
           # 记录开始时间
           start_time = time.time()
           
           # 执行推理
           with torch.no_grad():
               for _ in range(100):
                   src = torch.randn(32, 10, 512)
                   _ = self.model.generate(src)
           
           # 计算推理速度
           inference_time = time.time() - start_time
           samples_per_second = 100 / inference_time
           
           # 验证性能
           self.assertGreater(samples_per_second, 50)
   ```

## 测试最佳实践

### 1. 测试组织

1. 测试结构
   ```
   tests/
   ├── models/
   │   ├── test_transformer.py
   │   ├── test_encoder.py
   │   └── test_decoder.py
   ├── utils/
   │   ├── test_optimizer.py
   │   └── test_loss.py
   └── data/
       ├── test_dataset.py
       └── test_vocab.py
   ```

2. 测试命名
   - 使用描述性名称
   - 遵循命名约定
   - 包含测试目的

### 2. 测试覆盖

1. 代码覆盖
   ```python
   def test_coverage():
       # 运行测试
       coverage.run('tests')
       
       # 生成报告
       coverage.report()
       
       # 验证覆盖率
       assert coverage.report()['coverage'] > 0.8
   ```

2. 边界测试
   ```python
   def test_edge_cases():
       # 测试空输入
       with self.assertRaises(ValueError):
           model.generate([])
           
       # 测试最大长度
       with self.assertRaises(ValueError):
           model.generate(['a' * 1000])
           
       # 测试无效输入
       with self.assertRaises(ValueError):
           model.generate([None])
   ```

### 3. 测试维护

1. 测试文档
   ```python
   def test_documentation():
       # 检查文档字符串
       assert model.__doc__ is not None
       assert model.forward.__doc__ is not None
       
       # 检查参数文档
       assert 'Parameters' in model.__doc__
       assert 'Returns' in model.__doc__
   ```

2. 测试更新
   - 及时更新测试
   - 保持测试同步
   - 维护测试质量

## 常见问题

### 1. 测试失败

1. 原因
   - 代码变更
   - 环境变化
   - 测试错误

2. 解决方案
   - 检查代码变更
   - 验证环境配置
   - 修复测试用例

### 2. 测试性能

1. 问题
   - 测试速度慢
   - 内存使用高
   - 资源消耗大

2. 解决方案
   - 优化测试用例
   - 使用测试夹具
   - 并行执行测试

### 3. 测试维护

1. 挑战
   - 测试代码复杂
   - 维护成本高
   - 更新不及时

2. 解决方案
   - 简化测试代码
   - 自动化测试
   - 定期审查

## 参考

- [Python Testing with pytest](https://pytest.org/)
- [Unit Testing in Python](https://docs.python.org/3/library/unittest.html)
- [Test-Driven Development with Python](https://www.obeythetestinggoat.com/) 