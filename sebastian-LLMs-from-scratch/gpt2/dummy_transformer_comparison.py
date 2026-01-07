import numpy as np


# ============================================================================
# 1. MULTI-HEAD ATTENTION 对比
# ============================================================================

class MultiHeadAttention_NoCache:
    """没有 KV Cache 的版本"""

    def __init__(self, d_in, d_out, context_length, num_heads):
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 权重初始化
        scale = np.sqrt(2.0 / (d_in + d_out))
        self.W_q = np.random.randn(d_in, d_out) * scale
        self.W_k = np.random.randn(d_in, d_out) * scale
        self.W_v = np.random.randn(d_in, d_out) * scale
        self.W_o = np.random.randn(d_out, d_out) * scale

        # 因果掩码
        self.mask = np.triu(np.ones((context_length, context_length)), k=1)

    def __call__(self, x):
        batch_size, seq_len, d_in = x.shape

        # 每次都要重新计算所有token的Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k  # ❌ 每次重新计算
        V = x @ self.W_v  # ❌ 每次重新计算

        # 重塑和转置...
        return output


class MultiHeadAttention_WithCache:
    """有 KV Cache 的版本"""

    def __init__(self, d_in, d_out, context_length, num_heads):
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 权重初始化（相同）
        scale = np.sqrt(2.0 / (d_in + d_out))
        self.W_q = np.random.randn(d_in, d_out) * scale
        self.W_k = np.random.randn(d_in, d_out) * scale
        self.W_v = np.random.randn(d_in, d_out) * scale
        self.W_o = np.random.randn(d_out, d_out) * scale

        # 因果掩码（相同）
        self.mask = np.triu(np.ones((context_length, context_length)), k=1)

        # ============== KV CACHE 新增 ==============
        self.cache_k = None  # ✅ 新增：K缓存
        self.cache_v = None  # ✅ 新增：V缓存
        self.cache_length = 0
        # ===========================================

    def reset_cache(self):
        """✅ 新增：重置缓存方法"""
        self.cache_k = None
        self.cache_v = None
        self.cache_length = 0

    def __call__(self, x, use_cache=False):  # ✅ 新增：use_cache参数
        batch_size, seq_len, d_in = x.shape

        # 计算当前输入的K和V
        Q = x @ self.W_q
        K_new = x @ self.W_k
        V_new = x @ self.W_v

        # ============== KV CACHE 处理 ==============
        if use_cache:
            if self.cache_k is None:
                # 第一次调用：初始化缓存
                self.cache_k = K_new
                self.cache_v = V_new
            else:
                # 后续调用：追加到缓存
                self.cache_k = np.concatenate([self.cache_k, K_new], axis=1)
                self.cache_v = np.concatenate([self.cache_v, V_new], axis=1)

            # 使用缓存的K和V
            K = self.cache_k  # ✅ 复用缓存的K
            V = self.cache_v  # ✅ 复用缓存的V
            self.cache_length += seq_len
            total_seq_len = self.cache_length
        else:
            # 不使用缓存：和原来一样
            K = K_new
            V = V_new
            total_seq_len = seq_len
        # ===========================================

        # 动态掩码处理（需要调整以适应缓存）
        if use_cache:
            mask_slice = self.mask[self.cache_length - seq_len:self.cache_length, :self.cache_length]
        else:
            mask_slice = self.mask[:seq_len, :seq_len]

        # 重塑和转置...
        return output


# ============================================================================
# 2. TRANSFORMER BLOCK 对比
# ============================================================================

class TransformerBlock_NoCache:
    """没有 KV Cache 的版本"""

    def __init__(self, config):
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.attn = MultiHeadAttention_NoCache(  # ❌ 使用无缓存版本
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"]
        )
        self.ff = FeedForward(config["emb_dim"])

    def __call__(self, x):
        # 注意力块
        residual = x
        x = self.norm1(x)
        x = self.attn(x)  # ❌ 没有缓存参数
        x = x + residual

        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual

        return x


class TransformerBlock_WithCache:
    """有 KV Cache 的版本"""

    def __init__(self, config):
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.attn = MultiHeadAttention_WithCache(  # ✅ 使用有缓存版本
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"]
        )
        self.ff = FeedForward(config["emb_dim"])

    def __call__(self, x, use_cache=False):  # ✅ 新增：use_cache参数
        # 注意力块
        residual = x
        x = self.norm1(x)
        x = self.attn(x, use_cache=use_cache)  # ✅ 传递use_cache参数
        x = x + residual

        # 前馈网络（不变）
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual

        return x


# ============================================================================
# 3. GPT MODEL 对比
# ============================================================================

class GPTModel_NoCache:
    """没有 KV Cache 的版本"""

    def __init__(self, config):
        self.config = config
        vocab_size = config["vocab_size"]
        emb_dim = config["emb_dim"]
        context_length = config["context_length"]

        # 嵌入
        scale = 0.02
        self.token_emb = np.random.randn(vocab_size, emb_dim) * scale
        self.pos_emb = np.random.randn(context_length, emb_dim) * scale

        # Transformer块
        self.blocks = [TransformerBlock_NoCache(config) for _ in range(config["n_layers"])]
        self.final_norm = LayerNorm(emb_dim)
        self.output_head = np.random.randn(emb_dim, vocab_size) * scale

    def __call__(self, token_ids):
        batch_size, seq_len = token_ids.shape

        # 词嵌入
        token_embeds = self.token_emb[token_ids]

        # 位置嵌入（总是从0开始）
        pos_ids = np.arange(seq_len)  # ❌ 总是从0开始
        pos_embeds = self.pos_emb[pos_ids]
        pos_embeds = np.expand_dims(pos_embeds, 0)

        x = token_embeds + pos_embeds

        # 通过所有Transformer块
        for block in self.blocks:
            x = block(x)  # ❌ 没有缓存参数

        x = self.final_norm(x)
        logits = x @ self.output_head
        return logits


class GPTModel_WithCache:
    """有 KV Cache 的版本"""

    def __init__(self, config):
        self.config = config
        vocab_size = config["vocab_size"]
        emb_dim = config["emb_dim"]
        context_length = config["context_length"]

        # 嵌入（相同）
        scale = 0.02
        self.token_emb = np.random.randn(vocab_size, emb_dim) * scale
        self.pos_emb = np.random.randn(context_length, emb_dim) * scale

        # Transformer块（使用有缓存版本）
        self.blocks = [TransformerBlock_WithCache(config) for _ in range(config["n_layers"])]
        self.final_norm = LayerNorm(emb_dim)
        self.output_head = np.random.randn(emb_dim, vocab_size) * scale

        # ============== 新增：位置跟踪 ==============
        self.current_pos = 0  # ✅ 跟踪当前位置
        # ===========================================

    def reset_cache(self):
        """✅ 新增：重置所有缓存"""
        self.current_pos = 0
        for block in self.blocks:
            block.attn.reset_cache()

    def __call__(self, token_ids, use_cache=False):  # ✅ 新增：use_cache参数
        batch_size, seq_len = token_ids.shape

        # 词嵌入（相同）
        token_embeds = self.token_emb[token_ids]

        # ============== 动态位置嵌入 ==============
        if use_cache:
            # 使用缓存：从当前位置继续
            pos_ids = np.arange(self.current_pos, self.current_pos + seq_len)  # ✅ 动态位置
            self.current_pos += seq_len  # ✅ 更新位置
        else:
            # 不使用缓存：从0开始（和无缓存版本一样）
            pos_ids = np.arange(seq_len)

        pos_embeds = self.pos_emb[pos_ids]
        pos_embeds = np.expand_dims(pos_embeds, 0)
        # ===========================================

        x = token_embeds + pos_embeds

        # 通过所有Transformer块，传递use_cache参数
        for block in self.blocks:
            x = block(x, use_cache=use_cache)  # ✅ 传递use_cache参数

        x = self.final_norm(x)
        logits = x @ self.output_head
        return logits


# ============================================================================
# 4. 生成函数对比
# ============================================================================

def generate_text_no_cache(model, prompt_ids, max_new_tokens, context_size):
    """没有 KV Cache 的生成"""
    current_ids = prompt_ids.copy()

    for i in range(max_new_tokens):
        # 如果序列太长，裁剪到上下文长度
        if current_ids.shape[1] > context_size:
            input_ids = current_ids[:, -context_size:]  # ❌ 每次都传入完整序列
        else:
            input_ids = current_ids

        # 前向传播（重新计算所有）
        logits = model(input_ids, use_cache=False)  # ❌ 不使用缓存

        # 获取下一个token
        next_token = np.argmax(logits[:, -1, :], axis=-1)
        next_token = next_token.reshape(1, 1)

        # 追加到序列
        current_ids = np.concatenate([current_ids, next_token], axis=1)

    return current_ids


def generate_text_with_cache(model, prompt_ids, max_new_tokens):
    """有 KV Cache 的生成"""
    # 重置缓存
    model.reset_cache()  # ✅ 新增：开始前重置缓存

    # 第一次前向传播：处理完整提示
    logits = model(prompt_ids, use_cache=True)  # ✅ 使用缓存，初始化缓存

    current_ids = prompt_ids.copy()

    for i in range(max_new_tokens):
        # 获取下一个token（从上次的logits）
        next_token = np.argmax(logits[:, -1, :], axis=-1)
        next_token = next_token.reshape(1, 1)

        # 追加到序列
        current_ids = np.concatenate([current_ids, next_token], axis=1)

        # 关键优化：只传入新token！
        logits = model(next_token, use_cache=True)  # ✅ 只传入1个token

    return current_ids


# ============================================================================
# 5. 代码差异总结
# ============================================================================

def print_comparison_summary():
    print("=" * 80)
    print("KV CACHE 代码差异总结")
    print("=" * 80)

    print("\n一、核心数据结构差异:")
    print("┌────────────────────────────────┬────────────────────────────────┐")
    print("│       无 KV Cache              │       有 KV Cache              │")
    print("├────────────────────────────────┼────────────────────────────────┤")
    print("│ class MultiHeadAttention:      │ class MultiHeadAttention:      │")
    print("│   - 无缓存存储                 │   + cache_k = None             │")
    print("│   - 每次计算所有K/V            │   + cache_v = None             │")
    print("│                                │   + reset_cache()方法          │")
    print("└────────────────────────────────┴────────────────────────────────┘")

    print("\n二、方法签名差异:")
    print("无 KV Cache:                        有 KV Cache:")
    print("def __call__(self, x):              def __call__(self, x, use_cache=False):")
    print("    ...                                 ...")
    print("def forward(self, x):               def forward(self, x, use_cache=False):")
    print("    ...                                 ...")

    print("\n三、生成过程差异:")
    print("无 KV Cache 生成过程:               有 KV Cache 生成过程:")
    print("1. 输入: 完整序列                  1. 输入: 完整提示 (初始化缓存)")
    print("2. 计算: 所有token的Q,K,V          2. 计算: 所有token的Q,K,V, 存入缓存")
    print("3. 输出: 下一个token               3. 输出: 下一个token")
    print("4. 重复1-3 (重新计算所有!)         4. 输入: 只新token")
    print("                                  5. 计算: 新token的Q, 复用缓存的K,V")
    print("                                  6. 输出: 下一个token")
    print("                                  7. 重复4-6")

    print("\n四、性能影响:")
    print("┌─────────────┬──────────────┬──────────────┬─────────────┐")
    print("│ 序列长度 n  │ 无缓存复杂度 │ 有缓存复杂度 │ 加速比      │")
    print("├─────────────┼──────────────┼──────────────┼─────────────┤")
    print("│    10       │    O(100)    │    O(55)     │    1.8x     │")
    print("│    50       │    O(2500)   │    O(1275)   │    2.0x     │")
    print("│   100       │    O(10000)  │    O(5050)   │    2.0x     │")
    print("│   500       │   O(250000)  │   O(125250)  │    2.0x     │")
    print("└─────────────┴──────────────┴──────────────┴─────────────┘")

    print("\n五、内存使用差异:")
    print("无 KV Cache:                        有 KV Cache:")
    print("- 无额外内存开销                    - 需要存储K和V的缓存")
    print("- 但计算开销大                      - 内存换计算时间")
    print("- 适合短序列                        - 适合长序列生成")

    print("\n六、适用场景:")
    print("• 无 KV Cache 适合:")
    print("  - 短文本生成")
    print("  - 训练阶段")
    print("  - 内存受限环境")
    print("  - 单次前向传播")

    print("• 有 KV Cache 适合:")
    print("  - 长文本生成")
    print("  - 推理/部署阶段")
    print("  - 对话系统")
    print("  - 流式生成")

    print("\n七、实现注意事项:")
    print("1. 缓存管理: 需要reset_cache()在生成开始前调用")
    print("2. 位置编码: 需要动态调整以适应缓存")
    print("3. 注意力掩码: 需要根据缓存长度动态生成")
    print("4. 批处理: 缓存需要支持批量推理")
    print("5. 内存管理: 长序列可能需要缓存截断或分块")

    print("\n八、关键代码片段对比:")

    print("\n[无缓存] 注意力计算:")
    print("""
    Q = x @ W_q
    K = x @ W_k  # 每次重新计算
    V = x @ W_v  # 每次重新计算
    attn_scores = Q @ K^T  # 每次都计算完整矩阵
    """)

    print("\n[有缓存] 注意力计算:")
    print("""
    if use_cache:
        if cache_k is None:
            cache_k = K_new
            cache_v = V_new
        else:
            cache_k = concat(cache_k, K_new)
            cache_v = concat(cache_v, V_new)
        K = cache_k  # 复用缓存的K
        V = cache_v  # 复用缓存的V
    else:
        K = K_new
        V = V_new
    attn_scores = Q @ K^T  # Q小，K大（包含历史）
    """)

    print("\n[无缓存] 生成循环:")
    print("""
    for i in range(max_new_tokens):
        input_ids = current_ids[-context_size:]  # 传入完整序列
        logits = model(input_ids)  # 重新计算所有
        next_token = argmax(logits[-1])
        current_ids.append(next_token)
    """)

    print("\n[有缓存] 生成循环:")
    print("""
    model.reset_cache()
    logits = model(prompt_ids, use_cache=True)  # 初始化缓存

    for i in range(max_new_tokens):
        next_token = argmax(logits[-1])
        current_ids.append(next_token)
        logits = model(next_token, use_cache=True)  # 只传入新token
    """)


# ============================================================================
# 辅助类（简化版）
# ============================================================================

class LayerNorm:
    def __init__(self, emb_dim):
        self.gamma = np.ones(emb_dim)
        self.beta = np.zeros(emb_dim)

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + 1e-5) + self.beta


class FeedForward:
    def __init__(self, emb_dim):
        self.W1 = np.random.randn(emb_dim, 4 * emb_dim)
        self.W2 = np.random.randn(4 * emb_dim, emb_dim)

    def __call__(self, x):
        return (x @ self.W1) @ self.W2


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print_comparison_summary()

    # 演示两种生成方式的差异
    print("\n" + "=" * 80)
    print("实际生成过程演示")
    print("=" * 80)

    config = {
        "vocab_size": 100,
        "context_length": 20,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 1
    }

    np.random.seed(42)

    print("\n示例: 生成 3 个新token，初始提示为 [1, 2, 3]")
    print("-" * 60)

    print("\n无 KV Cache 的生成步骤:")
    print("1. 输入: [1, 2, 3]")
    print("   计算: Q,K,V for tokens 1,2,3")
    print("   注意力矩阵: 3×3")
    print("   输出: token 4")

    print("\n2. 输入: [2, 3, 4]")
    print("   重新计算: Q,K,V for tokens 2,3,4")
    print("   注意力矩阵: 3×3")
    print("   输出: token 5")

    print("\n3. 输入: [3, 4, 5]")
    print("   重新计算: Q,K,V for tokens 3,4,5")
    print("   注意力矩阵: 3×3")
    print("   输出: token 6")

    print("\n总计算: 3次完整的注意力计算")

    print("\n" + "-" * 60)
    print("\n有 KV Cache 的生成步骤:")
    print("1. 输入: [1, 2, 3] (初始化缓存)")
    print("   计算: Q,K,V for tokens 1,2,3")
    print("   存入缓存: K(1,2,3), V(1,2,3)")
    print("   注意力矩阵: 3×3")
    print("   输出: token 4")

    print("\n2. 输入: [4] (只新token)")
    print("   计算: Q for token 4")
    print("   复用缓存: K(1,2,3), V(1,2,3)")
    print("   更新缓存: K(1,2,3,4), V(1,2,3,4)")
    print("   注意力矩阵: 1×4")
    print("   输出: token 5")

    print("\n3. 输入: [5] (只新token)")
    print("   计算: Q for token 5")
    print("   复用缓存: K(1,2,3,4), V(1,2,3,4)")
    print("   更新缓存: K(1,2,3,4,5), V(1,2,3,4,5)")
    print("   注意力矩阵: 1×5")
    print("   输出: token 6")

    print("\n总计算: 1次完整 + 2次部分计算")

    print("\n" + "=" * 80)
    print("核心差异总结:")
    print("=" * 80)
    print("""
1. 数据结构:
   - 无缓存: 每次从头计算K和V
   - 有缓存: 存储历史K和V，避免重复计算

2. 计算模式:  
   - 无缓存: 每个token都重新计算整个序列
   - 有缓存: 新token只计算自己的Q，复用缓存的K和V

3. 生成效率:
   - 无缓存: O(n²) 复杂度，序列越长越慢
   - 有缓存: O(n) 复杂度，适合长序列生成

4. 内存使用:
   - 无缓存: 计算时内存需求小，但计算量大
   - 有缓存: 需要存储缓存，但计算量大大减少

5. 适用场景:
   - 无缓存: 训练、短序列、单次推理
   - 有缓存: 部署、长序列、对话系统、流式生成
""")