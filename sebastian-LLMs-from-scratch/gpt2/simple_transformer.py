import numpy as np
import math


def layer_norm(x, eps=1e-6):
    """简化的 LayerNorm (无学习参数)"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def softmax(x):
    """稳定的 softmax"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def positional_encoding(seq_len, d_model):
    """位置编码"""
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    # 交替使用 sin 和 cos
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    return pos_encoding


def attention(q, k, v, mask=None):
    """缩放点积注意力"""
    d_k = q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn_weights = softmax(scores)
    output = np.matmul(attn_weights, v)
    return output, attn_weights


def multi_head_attention(x, wq, wk, wv, wo, num_heads, mask=None):
    """简化的多头注意力 (无 bias)"""
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads

    # 线性变换并分割头
    q = np.matmul(x, wq).reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)
    k = np.matmul(x, wk).reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)
    v = np.matmul(x, wv).reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)

    # 注意力计算
    attn_output, attn_weights = attention(q, k, v, mask)
    attn_output = attn_output.swapaxes(1, 2).reshape(batch_size, seq_len, d_model)

    # 输出线性层
    output = np.matmul(attn_output, wo)
    return output, attn_weights


def feed_forward(x, w1, w2):
    """前馈网络 (ReLU 激活)"""
    return np.matmul(np.maximum(0, np.matmul(x, w1)), w2)


class EncoderLayer:
    """单层 Encoder"""

    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # 初始化权重 (小随机值)
        self.wq = np.random.randn(d_model, d_model) * 0.01
        self.wk = np.random.randn(d_model, d_model) * 0.01
        self.wv = np.random.randn(d_model, d_model) * 0.01
        self.wo = np.random.randn(d_model, d_model) * 0.01
        self.w1 = np.random.randn(d_model, d_ff) * 0.01
        self.w2 = np.random.randn(d_ff, d_model) * 0.01

    def forward(self, x, mask=None):
        # 多头注意力 + 残差
        attn_output, _ = multi_head_attention(x, self.wq, self.wk, self.wv, self.wo,
                                              self.num_heads, mask)
        x = layer_norm(x + attn_output)

        # 前馈网络 + 残差
        ff_output = feed_forward(x, self.w1, self.w2)
        x = layer_norm(x + ff_output)
        return x


class DecoderLayer:
    """单层 Decoder (带 Encoder-Decoder Attention)"""

    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # 自注意力权重
        self.wq_self = np.random.randn(d_model, d_model) * 0.01
        self.wk_self = np.random.randn(d_model, d_model) * 0.01
        self.wv_self = np.random.randn(d_model, d_model) * 0.01
        self.wo_self = np.random.randn(d_model, d_model) * 0.01

        # 交叉注意力权重
        self.wq_cross = np.random.randn(d_model, d_model) * 0.01
        self.wk_cross = np.random.randn(d_model, d_model) * 0.01
        self.wv_cross = np.random.randn(d_model, d_model) * 0.01
        self.wo_cross = np.random.randn(d_model, d_model) * 0.01

        # 前馈网络权重
        self.w1 = np.random.randn(d_model, d_ff) * 0.01
        self.w2 = np.random.randn(d_ff, d_model) * 0.01

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # 自注意力 + 残差
        self_attn_output, _ = multi_head_attention(x, self.wq_self, self.wk_self,
                                                   self.wv_self, self.wo_self,
                                                   self.num_heads, self_mask)
        x = layer_norm(x + self_attn_output)

        # 交叉注意力 + 残差
        cross_attn_output, _ = multi_head_attention(x, self.wq_cross, self.wk_cross,
                                                    self.wv_cross, self.wo_cross,
                                                    self.num_heads, cross_mask)
        x = layer_norm(x + cross_attn_output)

        # 前馈网络 + 残差
        ff_output = feed_forward(x, self.w1, self.w2)
        x = layer_norm(x + ff_output)
        return x


class Transformer:
    """完整的 Transformer (6层 Encoder + 6层 Decoder)"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, max_seq_len=100):
        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding 层 (简化: 随机初始化)
        self.src_embedding = np.random.randn(src_vocab_size, d_model) * 0.01
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * 0.01
        self.final_linear = np.random.randn(d_model, tgt_vocab_size) * 0.01

        # 6层 Encoder
        self.encoders = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

        # 6层 Decoder
        self.decoders = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

        # 位置编码缓存
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

    def encode(self, src_ids):
        """编码器前向传播"""
        batch_size, src_len = src_ids.shape
        x = self.src_embedding[src_ids] * np.sqrt(self.d_model)
        x += self.pos_encoding[:src_len, :]

        # 编码器掩码 (padding mask, 这里简化)
        src_mask = None  # 实际应用需实现 padding mask

        for encoder in self.encoders:
            x = encoder.forward(x, src_mask)
        return x

    def decode(self, tgt_ids, encoder_output):
        """解码器前向传播"""
        batch_size, tgt_len = tgt_ids.shape
        x = self.tgt_embedding[tgt_ids] * np.sqrt(self.d_model)
        x += self.pos_encoding[:tgt_len, :]

        # 解码器掩码 (causal mask)
        causal_mask = np.triu(np.ones((1, tgt_len, tgt_len)) * -1e9, k=1)

        for decoder in self.decoders:
            x = decoder.forward(x, encoder_output, causal_mask, None)
        return x

    def forward(self, src_ids, tgt_ids):
        """完整前向传播"""
        encoder_output = self.encode(src_ids)
        decoder_output = self.decode(tgt_ids, encoder_output)
        logits = np.matmul(decoder_output, self.final_linear)
        return logits


# ========== 测试 ==========
if __name__ == "__main__":
    # 超参数 (与原论文 Base 模型一致)
    SRC_VOCAB_SIZE = 10000
    TGT_VOCAB_SIZE = 10000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6  # 6层 Encoder 和 6层 Decoder

    # 创建模型
    transformer = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS
    )

    # 创建随机输入
    batch_size = 2
    src_len = 10
    tgt_len = 12

    src_ids = np.random.randint(0, SRC_VOCAB_SIZE, size=(batch_size, src_len))
    tgt_ids = np.random.randint(0, TGT_VOCAB_SIZE, size=(batch_size, tgt_len))

    # 前向传播
    logits = transformer.forward(src_ids, tgt_ids)

    print("Transformer 架构验证:")
    print(f"- Encoder 层数: {len(transformer.encoders)}")
    print(f"- Decoder 层数: {len(transformer.decoders)}")
    print(f"- 输入形状: src_ids {src_ids.shape}, tgt_ids {tgt_ids.shape}")
    print(f"- 输出 logits 形状: {logits.shape}")  # (batch_size, tgt_len, tgt_vocab_size)
    print(f"- 输出示例值范围: [{logits.min():.4f}, {logits.max():.4f}]")

'''
/Users/zbhuang/miniconda3/envs/build-llm-from-scratch-ch05/bin/python /Users/zbhuang/MyDev/AIProjects/LLMs-from-scratch/ch04/03_kv-cache/simple_transformer.py 
Transformer 架构验证:
- Encoder 层数: 6
- Decoder 层数: 6
- 输入形状: src_ids (2, 10), tgt_ids (2, 12)
- 输出 logits 形状: (2, 12, 10000)
- 输出示例值范围: [-1.0158, 0.9944]

Process finished with exit code 0
'''