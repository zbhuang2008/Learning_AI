import numpy as np
import time
from typing import Dict, Any


class SimpleLayerNorm:
    """简化的 Layer Normalization"""

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        self.gamma = np.ones(emb_dim)
        self.beta = np.zeros(emb_dim)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class SimpleGELU:
    """简化的 GELU 激活函数"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
        ))


class SimpleMultiHeadAttention:
    """简化的多头注意力，没有 KV Cache"""

    def __init__(self, d_in: int, d_out: int, context_length: int,
                 num_heads: int, dropout: float = 0.1):
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length
        self.dropout_rate = dropout

        # 初始化权重
        scale = np.sqrt(2.0 / (d_in + d_out))
        self.W_q = np.random.randn(d_in, d_out) * scale
        self.W_k = np.random.randn(d_in, d_out) * scale
        self.W_v = np.random.randn(d_in, d_out) * scale
        self.W_o = np.random.randn(d_out, d_out) * scale

        # 创建因果掩码（上三角矩阵）
        self.mask = np.triu(np.ones((context_length, context_length)), k=1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_in = x.shape

        # 线性投影：Q, K, V
        Q = x @ self.W_q  # (batch, seq_len, d_out)
        K = x @ self.W_k  # (batch, seq_len, d_out)
        V = x @ self.W_v  # (batch, seq_len, d_out)

        # 为多头注意力重塑形状
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置以进行注意力计算
        Q = np.transpose(Q, (0, 2, 1, 3))  # (batch, heads, seq_len, head_dim)
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))

        # 计算注意力分数
        # Q @ K^T: (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
        # = (batch, heads, seq_len, seq_len)
        attn_scores = Q @ np.transpose(K, (0, 1, 3, 2))

        # 应用因果掩码
        mask_slice = self.mask[:seq_len, :seq_len].astype(bool)
        mask_expanded = np.expand_dims(np.expand_dims(mask_slice, 0), 0)
        attn_scores = np.where(mask_expanded, -np.inf, attn_scores)

        # Softmax（数值稳定版本）
        def stable_softmax(x, axis=-1):
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        attn_weights = stable_softmax(attn_scores / np.sqrt(self.head_dim))

        # Dropout（训练时使用）
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, attn_weights.shape)
            attn_weights = attn_weights * dropout_mask / (1 - self.dropout_rate)

        # 计算上下文向量
        context = attn_weights @ V  # (batch, heads, seq_len, head_dim)

        # 重塑回原始形状
        context = np.transpose(context, (0, 2, 1, 3))  # (batch, seq_len, heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.d_out)

        # 输出投影
        output = context @ self.W_o
        return output


class SimpleFeedForward:
    """简化的前馈网络"""

    def __init__(self, emb_dim: int):
        self.emb_dim = emb_dim
        scale = np.sqrt(2.0 / (emb_dim * 5))

        self.W1 = np.random.randn(emb_dim, 4 * emb_dim) * scale
        self.b1 = np.zeros(4 * emb_dim)
        self.W2 = np.random.randn(4 * emb_dim, emb_dim) * scale
        self.b2 = np.zeros(emb_dim)
        self.gelu = SimpleGELU()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.W1 + self.b1
        h = self.gelu(h)
        return h @ self.W2 + self.b2


class SimpleTransformerBlock:
    """简化的 Transformer 块"""

    def __init__(self, config: Dict[str, Any]):
        self.norm1 = SimpleLayerNorm(config["emb_dim"])
        self.norm2 = SimpleLayerNorm(config["emb_dim"])
        self.attn = SimpleMultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"],
            dropout=config.get("drop_rate", 0.1)
        )
        self.ff = SimpleFeedForward(config["emb_dim"])
        self.dropout_rate = config.get("drop_rate", 0.1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # 注意力块 + 残差连接
        residual = x
        x = self.norm1(x)
        x = self.attn(x)

        # Dropout
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
            x = x * dropout_mask / (1 - self.dropout_rate)

        x = x + residual

        # 前馈网络 + 残差连接
        residual = x
        x = self.norm2(x)
        x = self.ff(x)

        # Dropout
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
            x = x * dropout_mask / (1 - self.dropout_rate)

        x = x + residual
        return x


class SimpleGPTModel:
    """简化的 GPT 模型，没有 KV Cache"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vocab_size = config["vocab_size"]
        emb_dim = config["emb_dim"]
        context_length = config["context_length"]

        # 词嵌入和位置嵌入
        scale = 0.02
        self.token_emb = np.random.randn(vocab_size, emb_dim) * scale
        self.pos_emb = np.random.randn(context_length, emb_dim) * scale

        # Transformer 块
        self.blocks = [SimpleTransformerBlock(config) for _ in range(config["n_layers"])]
        self.final_norm = SimpleLayerNorm(emb_dim)

        # 输出头
        self.output_head = np.random.randn(emb_dim, vocab_size) * scale

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = token_ids.shape

        # 词嵌入
        token_embeds = self.token_emb[token_ids]  # (batch, seq_len, emb_dim)

        # 位置嵌入
        pos_ids = np.arange(seq_len)
        pos_embeds = self.pos_emb[pos_ids]  # (seq_len, emb_dim)
        pos_embeds = np.expand_dims(pos_embeds, 0)  # 添加批次维度

        x = token_embeds + pos_embeds

        # 通过所有 Transformer 块
        for block in self.blocks:
            x = block(x)

        # 最终层归一化和输出
        x = self.final_norm(x)
        logits = x @ self.output_head  # (batch, seq_len, vocab_size)
        return logits


def simple_generate_text(model, prompt_ids, max_new_tokens, context_size):
    """简单的文本生成函数，没有 KV Cache"""
    batch_size = prompt_ids.shape[0]
    current_ids = prompt_ids.copy()

    print(f"开始生成 {max_new_tokens} 个新token...")

    for i in range(max_new_tokens):
        # 如果序列太长，只使用最后 context_size 个token
        if current_ids.shape[1] > context_size:
            input_ids = current_ids[:, -context_size:]
        else:
            input_ids = current_ids

        # 获取logits
        logits = model(input_ids)

        # 贪婪采样（取最大值）
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)

        # 添加到序列
        current_ids = np.concatenate([current_ids, next_token], axis=1)

        # 每生成10个token打印一次进度
        if (i + 1) % 10 == 0:
            print(f"  已生成 {i + 1}/{max_new_tokens} 个token")

    return current_ids


def benchmark_simple_model():
    """基准测试简单模型"""
    print("=" * 60)
    print("纯 NumPy 实现的简化 Transformer 架构")
    print("=" * 60)

    # 小模型配置用于测试
    config = {
        "vocab_size": 10000,
        "context_length": 128,
        "emb_dim": 64,
        "n_heads": 4,
        "n_layers": 3,
        "drop_rate": 0.0  # 基准测试时禁用dropout
    }

    print(f"模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 初始化模型
    np.random.seed(42)
    model = SimpleGPTModel(config)

    # 创建随机提示
    prompt_length = 8
    max_new_tokens = 30
    prompt_ids = np.random.randint(0, config["vocab_size"], size=(1, prompt_length))

    print(f"\n提示长度: {prompt_length}")
    print(f"要生成的token数: {max_new_tokens}")
    print(f"总序列长度: {prompt_length + max_new_tokens}")

    # 运行生成
    print("\n开始生成（无 KV Cache）...")
    start_time = time.time()

    output = simple_generate_text(
        model,
        prompt_ids,
        max_new_tokens,
        config["context_length"]
    )

    total_time = time.time() - start_time

    # 打印结果
    print(f"\n生成完成!")
    print(f"总时间: {total_time:.4f} 秒")
    print(f"Tokens/秒: {max_new_tokens / total_time:.2f}")
    print(f"总序列长度: {output.shape[1]}")

    # 计算复杂度分析
    print("\n" + "=" * 60)
    print("计算复杂度分析:")
    print("=" * 60)

    n = prompt_length + max_new_tokens
    print(f"总序列长度: n = {n}")
    print(f"注意力计算复杂度: O(n²)")
    print(f"每个生成步骤的计算量:")

    for step in range(max_new_tokens):
        current_len = prompt_length + step + 1
        print(f"  步骤 {step + 1}: 序列长度 = {current_len}, 计算量 ≈ {current_len ** 2}")

    total_ops = sum([(prompt_length + i + 1) ** 2 for i in range(max_new_tokens)])
    print(f"\n总计算量（近似）: Σ(i=1 to {n}) i² ≈ {total_ops:,} 次操作")

    return {
        "model": model,
        "output": output,
        "time": total_time,
        "config": config
    }


def analyze_computation():
    """分析计算过程"""
    print("\n" + "=" * 60)
    print("详细计算过程分析:")
    print("=" * 60)

    # 更小的模型用于演示
    mini_config = {
        "vocab_size": 100,
        "context_length": 10,
        "emb_dim": 8,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.0
    }

    print("创建微型模型用于演示...")
    mini_model = SimpleGPTModel(mini_config)

    # 演示输入
    demo_input = np.array([[1, 2, 3]])  # 批次大小=1, 序列长度=3

    print(f"\n输入形状: {demo_input.shape}")
    print("输入 token IDs: [1, 2, 3]")

    # 单次前向传播
    print("\n1. 第一次前向传播（处理初始提示）:")
    print("   - 计算 token 1 的 Q, K, V")
    print("   - 计算 token 2 的 Q, K, V")
    print("   - 计算 token 3 的 Q, K, V")
    print("   - 注意力计算: 3×3 的矩阵")

    # 生成新token
    print("\n2. 生成第一个新token:")
    print("   - 计算 token 1-3 的 Q, K, V（重新计算！）")
    print("   - 计算新 token 4 的 Q, K, V")
    print("   - 注意力计算: 4×4 的矩阵")

    print("\n3. 生成第二个新token:")
    print("   - 计算 token 1-4 的 Q, K, V（再次重新计算！）")
    print("   - 计算新 token 5 的 Q, K, V")
    print("   - 注意力计算: 5×5 的矩阵")

    print("\n关键问题: 重复计算!")
    print("- token 1-3 的 K 和 V 在每一步都被重新计算")
    print("- 这导致了 O(n²) 的计算复杂度")


def main():
    """主函数"""
    print("简化版 Transformer（无 KV Cache）")
    print("=" * 60)

    # 运行基准测试
    results = benchmark_simple_model()

    # 分析计算过程
    analyze_computation()

    # 总结
    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print("这个简化版 Transformer 实现了基本功能但效率较低:")
    print("1. 每次生成新token时都要重新计算所有历史token的K和V")
    print("2. 计算复杂度为 O(n²)，n是序列长度")
    print("3. 对于长序列生成，这会非常慢")
    print("\n下一步: 添加 KV Cache 来优化性能!")

    # 显示模型结构
    print("\n模型结构总结:")
    config = results["config"]
    print(f"- 词表大小: {config['vocab_size']}")
    print(f"- 嵌入维度: {config['emb_dim']}")
    print(f"- 注意力头数: {config['n_heads']}")
    print(f"- Transformer 层数: {config['n_layers']}")
    print(f"- 上下文长度: {config['context_length']}")


if __name__ == "__main__":
    main()