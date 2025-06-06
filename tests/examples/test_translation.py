import examples.translation as translation

def test_translation():
    # 获取数据
    src_vocab, tgt_vocab, train_data, val_data = translation.prepare_data()
    
    print("=== 词汇表信息 ===")
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 测试一个具体的翻译对
    test_pair = train_data[0]  # 使用第一个训练样本
    src_text, tgt_text = test_pair
    
    print("\n=== 转换过程演示 ===")
    print(f"源语言文本: {src_text}")
    print(f"目标语言文本: {tgt_text}")
    
    # 源语言转换过程
    print("\n源语言转换过程:")
    src_tokens = src_text.split()
    print(f"1. 分词结果: {src_tokens}")
    
    src_indices = [src_vocab.get_index(token) for token in src_tokens]
    print(f"2. 转换为索引: {src_indices}")
    
    src_sequence = src_vocab.tokenize(src_text)
    print(f"3. 添加特殊标记后的完整序列: {src_sequence}")
    print(f"4. 序列含义: {[src_vocab.idx2word[idx] for idx in src_sequence]}")
    
    # 目标语言转换过程
    print("\n目标语言转换过程:")
    tgt_tokens = tgt_text.split()
    print(f"1. 分词结果: {tgt_tokens}")
    
    tgt_indices = [tgt_vocab.get_index(token) for token in tgt_tokens]
    print(f"2. 转换为索引: {tgt_indices}")
    
    tgt_sequence = tgt_vocab.tokenize(tgt_text)
    print(f"3. 添加特殊标记后的完整序列: {tgt_sequence}")
    print(f"4. 序列含义: {[tgt_vocab.idx2word[idx] for idx in tgt_sequence]}")
    
    # 展示词汇表中的一些映射关系
    print("\n=== 词汇表示例 ===")
    print("源语言词汇表示例:")
    for i in range(5):  # 显示前5个词
        if i in src_vocab.idx2word:
            print(f"索引 {i}: {src_vocab.idx2word[i]}")
    
    print("\n目标语言词汇表示例:")
    for i in range(5):  # 显示前5个词
        if i in tgt_vocab.idx2word:
            print(f"索引 {i}: {tgt_vocab.idx2word[i]}")

if __name__ == "__main__":
    test_translation()