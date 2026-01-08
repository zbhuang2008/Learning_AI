import numpy as np
from simple_transformer import Transformer

def greedy_decode(logits):
    """贪心解码：选择每个位置概率最高的词"""
    return np.argmax(logits, axis=-1)


# ========== 翻译示例 ==========
if __name__ == "__main__":
    print("=== 中英文翻译演示 ===\n")

    # 1. 创建词汇表（简化示例）
    # 中文词汇表
    chinese_vocab = {
        '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
        '我': 4, '爱': 5, '学习': 6, '人工智能': 7, '技术': 8,
        '今天': 9, '天气': 10, '很好': 11
    }

    # 英文词汇表
    english_vocab = {
        '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
        'I': 4, 'love': 5, 'learning': 6, 'AI': 7, 'technology': 8,
        'Today': 9, 'weather': 10, 'is': 11, 'good': 12
    }

    # 反向词汇表用于解码
    idx_to_chinese = {v: k for k, v in chinese_vocab.items()}
    idx_to_english = {v: k for k, v in english_vocab.items()}

    print("中文词汇表（部分）:", list(chinese_vocab.keys()))
    print("英文词汇表（部分）:", list(english_vocab.keys()))
    print()

    # 2. 初始化Transformer模型
    transformer = Transformer(
        src_vocab_size=len(chinese_vocab),  # 源语言（中文）词汇大小
        tgt_vocab_size=len(english_vocab),  # 目标语言（英文）词汇大小
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6
    )

    print("模型初始化完成！")
    print(f"Encoder层数: {len(transformer.encoders)}")
    print(f"Decoder层数: {len(transformer.decoders)}")
    print()

    # 3. 准备翻译示例
    # 示例1：简单句子
    print("示例1：简单句子翻译")
    chinese_sentence = "我 爱 学习"
    english_sentence = "I love learning"

    print(f"中文原文: {chinese_sentence}")
    print(f"英文参考: {english_sentence}")

    # 转换为词汇ID序列（添加特殊标记）
    src_ids = [chinese_vocab['<SOS>']] + [chinese_vocab.get(word, chinese_vocab['<UNK>'])
                                          for word in chinese_sentence.split()] + [chinese_vocab['<EOS>']]

    # 解码器输入（训练时使用真实目标序列，推理时使用逐步生成）
    tgt_ids = [english_vocab['<SOS>']] + [english_vocab.get(word, english_vocab['<UNK>'])
                                          for word in english_sentence.split()] + [english_vocab['<EOS>']]

    # 转换为batch格式（batch_size=1）
    src_ids = np.array([src_ids])  # shape: (1, seq_len)
    tgt_ids = np.array([tgt_ids])  # shape: (1, seq_len)

    print(f"中文ID序列: {src_ids[0]}")
    print(f"英文ID序列: {tgt_ids[0]}")
    print()

    # 4. 前向传播（模拟推理过程）
    print("执行Transformer前向传播...")
    logits = transformer.forward(src_ids, tgt_ids)  # shape: (1, tgt_seq_len, tgt_vocab_size)

    print(f"输出logits形状: {logits.shape}")
    print()

    # 5. 解码输出
    print("解码翻译结果...")
    output_ids = greedy_decode(logits[0])  # 取第一个batch

    # 将ID转换回单词
    decoded_words = []
    for idx in output_ids:
        word = idx_to_english.get(idx, '<UNK>')
        if word == '<EOS>':
            break
        if word not in ['<SOS>', '<PAD>']:
            decoded_words.append(word)

    translation = ' '.join(decoded_words)
    print(f"模型翻译结果: {translation}")
    print()

    # 6. 另一个示例
    print("示例2：稍复杂句子")
    chinese_sentence2 = "今天 天气 很好"
    english_sentence2 = "Today weather is good"

    print(f"中文原文: {chinese_sentence2}")
    print(f"英文参考: {english_sentence2}")

    # 准备输入
    src_ids2 = [chinese_vocab['<SOS>']] + [chinese_vocab.get(word, chinese_vocab['<UNK>'])
                                           for word in chinese_sentence2.split()] + [chinese_vocab['<EOS>']]
    tgt_ids2 = [english_vocab['<SOS>']] + [english_vocab.get(word, english_vocab['<UNK>'])
                                           for word in english_sentence2.split()] + [english_vocab['<EOS>']]

    src_ids2 = np.array([src_ids2])
    tgt_ids2 = np.array([tgt_ids2])

    # 前向传播
    logits2 = transformer.forward(src_ids2, tgt_ids2)
    output_ids2 = greedy_decode(logits2[0])

    # 解码
    decoded_words2 = []
    for idx in output_ids2:
        word = idx_to_english.get(idx, '<UNK>')
        if word == '<EOS>':
            break
        if word not in ['<SOS>', '<PAD>']:
            decoded_words2.append(word)

    translation2 = ' '.join(decoded_words2)
    print(f"模型翻译结果: {translation2}")
    print()

    # 7. 模型结构验证
    print("=== 模型结构详细信息 ===")
    print(f"源语言词汇表大小: {len(chinese_vocab)}")
    print(f"目标语言词汇表大小: {len(english_vocab)}")
    print(f"模型维度 (d_model): {transformer.d_model}")
    print(f"注意力头数量: {len(transformer.encoders[0].wq) // transformer.d_model}")
    print(f"前馈网络隐藏层维度: {transformer.encoders[0].w1.shape[1]}")

    # 8. 输出示例词向量
    print("\n=== 词嵌入示例 ===")
    print("中文词 '我' 的嵌入向量（前5个维度）:")
    print(transformer.src_embedding[chinese_vocab['我']][:5])
    print("英文词 'I' 的嵌入向量（前5个维度）:")
    print(transformer.tgt_embedding[english_vocab['I']][:5])

'''
/Users/zbhuang/miniconda3/envs/build-llm-from-scratch-ch05/bin/python /Users/zbhuang/MyDev/AIProjects/LLMs-from-scratch/ch04/03_kv-cache/simple_transformer_app.py 
=== 中英文翻译演示 ===

中文词汇表（部分）: ['<PAD>', '<SOS>', '<EOS>', '<UNK>', '我', '爱', '学习', '人工智能', '技术', '今天', '天气', '很好']
英文词汇表（部分）: ['<PAD>', '<SOS>', '<EOS>', '<UNK>', 'I', 'love', 'learning', 'AI', 'technology', 'Today', 'weather', 'is', 'good']

模型初始化完成！
Encoder层数: 6
Decoder层数: 6

示例1：简单句子翻译
中文原文: 我 爱 学习
英文参考: I love learning
中文ID序列: [1 4 5 6 2]
英文ID序列: [1 4 5 6 2]

执行Transformer前向传播...
输出logits形状: (1, 5, 13)

解码翻译结果...
模型翻译结果: good

示例2：稍复杂句子
中文原文: 今天 天气 很好
英文参考: Today weather is good
模型翻译结果: good

=== 模型结构详细信息 ===
源语言词汇表大小: 12
目标语言词汇表大小: 13
模型维度 (d_model): 512
注意力头数量: 1
前馈网络隐藏层维度: 2048

=== 词嵌入示例 ===
中文词 '我' 的嵌入向量（前5个维度）:
[-0.00764814  0.00653177 -0.00536528  0.02156546  0.01641358]
英文词 'I' 的嵌入向量（前5个维度）:
[-0.00269433  0.01921359  0.00081778  0.00116542 -0.00056021]

Process finished with exit code 0
'''