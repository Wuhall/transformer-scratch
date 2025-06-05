# Transformer Scratch Implementation

这个项目是对论文 "Attention Is All You Need" 中提出的 Transformer 架构的从头实现。本项目使用 PyTorch 框架，旨在帮助理解 Transformer 的核心概念和实现细节。

## 项目结构

```
.
├── src/
│   ├── models/        # Transformer模型相关代码
│   ├── utils/         # 工具函数
│   ├── data/          # 数据处理相关代码
│   └── train.py       # 训练脚本
├── requirements.txt   # 项目依赖
└── README.md         # 项目说明
```

## 环境配置

1. 创建并激活conda环境：
```bash
conda create -n transformer python=3.8
conda activate transformer
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 主要创新点

1. Self-Attention机制：允许模型直接建模序列中任意位置之间的关系
2. Multi-Head Attention：通过多个注意力头捕获不同类型的关系
3. Positional Encoding：为模型提供序列位置信息
4. 残差连接和层归一化：帮助训练更深的网络
5. 并行计算能力：相比RNN，可以并行处理整个序列

## 使用说明

待补充... 