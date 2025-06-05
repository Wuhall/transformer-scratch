# 贡献指南

感谢您对 Transformer 项目的关注！我们欢迎各种形式的贡献，包括但不限于：

- 报告问题
- 提交改进建议
- 提交代码修改
- 改进文档
- 分享使用经验

## 开发环境

### 1. 环境配置

1. 克隆仓库
   ```bash
   git clone https://github.com/yourusername/transformer-scratch.git
   cd transformer-scratch
   ```

2. 创建虚拟环境
   ```bash
   conda create -n transformer python=3.8
   conda activate transformer
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

### 2. 开发工具

1. 代码格式化
   - 使用 black 进行代码格式化
   - 使用 isort 进行导入排序
   - 使用 flake8 进行代码检查

2. 类型检查
   - 使用 mypy 进行类型检查
   - 添加类型注解
   - 确保类型安全

3. 测试工具
   - 使用 pytest 进行单元测试
   - 使用 coverage 进行代码覆盖
   - 使用 tox 进行环境测试

## 贡献流程

### 1. 问题报告

1. 检查现有问题
   - 搜索相关问题
   - 避免重复报告
   - 提供详细信息

2. 创建问题
   - 使用问题模板
   - 提供复现步骤
   - 附加相关日志

### 2. 功能请求

1. 检查现有功能
   - 搜索相关功能
   - 避免重复请求
   - 提供使用场景

2. 提交请求
   - 使用功能请求模板
   - 描述功能需求
   - 提供实现建议

### 3. 代码贡献

1. 创建分支
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. 开发功能
   - 遵循代码规范
   - 添加单元测试
   - 更新文档

3. 提交代码
   ```bash
   git add .
   git commit -m "feat: add your feature"
   git push origin feature/your-feature-name
   ```

4. 创建拉取请求
   - 使用 PR 模板
   - 描述修改内容
   - 关联相关问题

## 代码规范

### 1. 代码风格

1. Python 代码
   ```python
   # 使用 4 个空格缩进
   def function_name(param1, param2):
       """函数文档字符串"""
       # 使用空行分隔逻辑块
       result = param1 + param2
       return result
   ```

2. 命名规范
   - 类名使用驼峰命名
   - 函数名使用下划线命名
   - 常量使用大写字母

3. 注释规范
   - 使用文档字符串
   - 添加类型注解
   - 解释复杂逻辑

### 2. 测试规范

1. 单元测试
   ```python
   def test_function():
       """测试函数文档字符串"""
       # 准备测试数据
       input_data = prepare_test_data()
       
       # 执行测试
       result = function_to_test(input_data)
       
       # 验证结果
       assert result == expected_output
   ```

2. 测试覆盖
   - 覆盖核心功能
   - 测试边界条件
   - 测试异常情况

### 3. 文档规范

1. 代码文档
   ```python
   class MyClass:
       """类文档字符串"""
       
       def my_method(self, param):
           """方法文档字符串
           
           Args:
               param: 参数描述
               
           Returns:
               返回值描述
               
           Raises:
               异常描述
           """
           pass
   ```

2. 项目文档
   - 更新 README
   - 更新 API 文档
   - 添加使用示例

## 审查流程

### 1. 代码审查

1. 审查重点
   - 代码质量
   - 测试覆盖
   - 文档完整

2. 审查反馈
   - 及时响应
   - 友好交流
   - 持续改进

### 2. 合并流程

1. 合并条件
   - 通过所有测试
   - 通过代码审查
   - 解决所有问题

2. 合并步骤
   - 更新主分支
   - 解决冲突
   - 合并代码

## 发布流程

### 1. 版本管理

1. 版本号
   - 遵循语义化版本
   - 更新版本号
   - 更新更新日志

2. 发布步骤
   - 创建发布分支
   - 更新文档
   - 创建发布标签

### 2. 更新日志

1. 日志格式
   ```markdown
   # 版本号 (日期)
   
   ## 新功能
   - 功能 1
   - 功能 2
   
   ## 修复
   - 修复 1
   - 修复 2
   
   ## 改进
   - 改进 1
   - 改进 2
   ```

2. 日志内容
   - 记录重要变更
   - 提供升级指南
   - 说明兼容性

## 行为准则

### 1. 基本原则

1. 尊重他人
   - 友好交流
   - 接受批评
   - 帮助他人

2. 专业行为
   - 遵守规范
   - 保持专注
   - 持续学习

### 2. 沟通准则

1. 交流方式
   - 使用清晰语言
   - 提供具体例子
   - 保持耐心

2. 反馈方式
   - 建设性反馈
   - 具体建议
   - 积极态度

## 参考

- [Python 代码规范](https://www.python.org/dev/peps/pep-0008/)
- [Git 工作流](https://git-scm.com/book/zh/v2)
- [语义化版本](https://semver.org/lang/zh-CN/) 