import numpy as np
np.set_printoptions(precision=6, suppress=True)

# ==================== 1. 输入 token 与词嵌入 ====================
vocab_size = 16
d_model = 5
seq_len = 6

# 输入 token 的 one-hot 矩阵 (16×6)
input_tokens = np.array([
    [0, 1, 0, 0, 0, 0],  # 狗 t=1
    [0, 0, 0, 0, 1, 0],  # 貓 t=4
    [0, 0, 0, 0, 0, 0],  # 牛 (未出现)
    [0, 0, 0, 0, 0, 0],  # 大
    [0, 0, 0, 0, 0, 0],  # 中
    [1, 0, 0, 1, 0, 0],  # 小 t=0, t=3
    [0, 0, 0, 0, 0, 0],  # 吃
    [0, 0, 0, 0, 0, 1],  # 玩 t=5
    [0, 0, 0, 0, 0, 0],  # 跑
    [0, 0, 1, 0, 0, 0],  # 和 t=2
    [0, 0, 0, 0, 0, 0],  # 你
    [0, 0, 0, 0, 0, 0],  # 我
    [0, 0, 0, 0, 0, 0],  # 他
    [0, 0, 0, 0, 0, 0],  # 的
    [0, 0, 0, 0, 0, 0],  # 好
    [0, 0, 0, 0, 0, 0]   # 在
])

# 词嵌入权重矩阵 W_embed (5×16)
W_embed = np.array([
    [3, 0, 2, 1, 2, 3, 2, 2, 3, 0, 1, 0, 0, 0, 0, 1],
    [0, 3, 0, 3, 0, 2, 2, 1, 1, 3, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 2, 3, 1, 1, 3, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 3, 3, 3, 3, 2, 1, 3, 2, 1, 1, 0, 1, 1, 0],
    [3, 1, 0, 3, 0, 2, 3, 2, 2, 2, 0, 0, 0, 1, 0, 0]
])

# 词嵌入: (5,16) × (16,6) = (5,6)
embeddings = W_embed @ input_tokens  # 未加位置编码
print("词嵌入矩阵 (5×6) - embeddings:")
print(embeddings.round(6))
print()

# ==================== 2. 位置编码 ====================
# # 正弦位置编码 (sin/cos 交替)
# def positional_encoding(seq_len, d_model):
#     PE = np.zeros((seq_len, d_model))
#     for pos in range(seq_len):
#         for i in range(0, d_model, 2):
#             PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
#             if i + 1 < d_model:
#                 PE[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
#     return PE.T  # 转为 (d_model, seq_len)
#
# PE = positional_encoding(seq_len, d_model)
# # 表格中使用了缩放的 PE，这里按示例缩放
# scale = 1.07  # 根据表格调整的比例
# PE_scaled = PE * scale
PE_scaled = np.array([
        [0.9217964,   0.998010124, 0.999684538, 0.999950000, 0.999992076, 0.999992076],
        [0.714713463, 0.311697146, 0.125856817, 0.050216599, 0.019998667, 0.007962059],
        [0.367644457, 0.889078609, 0.982138604, 0.997162035, 0.999550034, 0.999928681],
        [0.999766030, 0.592337725, 0.249712113, 0.100306487, 0.039989334, 0.015923614],
        [-0.407522603,0.702105263, 0.950647969, 0.992123395, 0.998750260, 0.999801895]
    ]
)

# 添加位置编码
embeddings_pe = embeddings + PE_scaled
# embeddings_pe = np.array([
#         [3.9217964,   3.998010124, 0.999684538, 3.999950000, 0.999992076, 2.999992076],
#         [2.714713463, 0.311697146, 3.125856817, 2.050216599, 3.019998667, 1.007962059],
#         [1.367644457, 0.889078609, 1.982138604, 1.997162035, 1.999550034, 3.999928681],
#         [3.999766030, 1.592337725, 2.249712113, 3.100306487, 1.039989334, 1.015923614],
#         [1.592477397, 3.702105263, 2.950647969, 2.992123395, 1.998750260, 2.999801895]
#     ]
# )
print("词嵌入 + 位置编码 (5×6) - embeddings_pe:")
print(embeddings_pe.round(6))
print()

# ==================== 3. 多头注意力 Q/K/V ====================
num_heads = 2
d_k = 3  # 每个头的维度

# 合并的 Wq (6×5) - 两个头的 Q 权重拼接
Wq = np.array([
    [-1, -1,  1,  1,  1],
    [ 1, -1, -1,  0, -1],
    [-1, -1,  0,  0,  1],
    [ 0,  1,  1,  1,  1],
    [ 0,  1, -1,  1,  1],
    [-1,  1,  0, -1,  0]
])  # (num_heads*d_k, d_model) = (6,5)

# 合并的 Wk 和 Wv（示例，来自表格）
Wk = np.array([
    [-1, -1,  1,  1,  1],
    [ 1, -1, -1,  0, -1],
    [-1, -1,  0,  0,  1],
    [ 0,  1, -1, -1,  1],
    [ 0,  1, -1,  0,  0],
    [-1,  1,  0, -1,  0]
])  # 与 Wq 相同（仅为示例）

Wv = np.array([
    [ 1,  1, -1,  0,  1],
    [ 0,  1,  0,  1,  0],
    [ 0, -1, -1,  1, -1],
    [-1,  1,  1,  0,  1],
    [-1,  0,  0,  1, -1],
    [ 0,  0, -1, -1,  0]
])  # 示例 V 权重

# 计算 Q, K, V
X = embeddings_pe  # (6,5)

Q = Wq @ X  # (6,6)
K = Wk @ X  # (6,6)
V = Wv @ X  # (6,6)

print("Q 矩阵 (6×6):")
print(Q.round(6))
print()
print("K 矩阵 (6×6):")
print(K.round(6))
print()
print("V 矩阵 (6×6):")
print(V.round(6))
print()

# # ==================== 4. 缩放点积注意力 + 因果掩码 ====================
attn_output = []

# Head 0
# 缩放
scale_factor = np.sqrt(d_k)
scores_scaled_0 = (K[:3].T @ Q[:3]) / scale_factor  # (6,6)
print("scores_scaled_0 矩阵 (6×6):")
print(scores_scaled_0.round(6))
print()

# # 因果掩码（下三角为 0，上三角为 -inf）
# mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
# scores_masked = scores + mask

# softmax (列方向) - axis=-1 by row, axis=0 by column
attn_weights_0 = np.exp(scores_scaled_0 - np.max(scores_scaled_0, axis=0, keepdims=True))
attn_weights_0 = attn_weights_0 / attn_weights_0.sum(axis=0, keepdims=True)
print("attn_weights_0 矩阵 (6×6):")
print(attn_weights_0.round(6))
print()

# 注意力输出
attn_output_0 = V[:3] @ attn_weights_0  # (6,6)
print("attn_output_0 矩阵 (3×6):")
print(attn_output_0.round(6))
print()

attn_output.append(attn_output_0)

# Head 1
# 缩放
# scale_factor = np.sqrt(d_k)
scores_scaled_1 = (K[3:].T @ Q[3:]) / scale_factor  # (6,6)
print("scores_scaled_1 矩阵 (6×6):")
print(scores_scaled_1.round(6))
print()

# # 因果掩码（下三角为 0，上三角为 -inf）
# mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
# scores_masked = scores + mask

# softmax (列方向) - axis=-1 by row, axis=0 by column
attn_weights_1 = np.exp(scores_scaled_1 - np.max(scores_scaled_1, axis=0, keepdims=True))
attn_weights_1 = attn_weights_1 / attn_weights_1.sum(axis=0, keepdims=True)
print("attn_weights_1 矩阵 (6×6):")
print(attn_weights_1.round(6))
print()

# 注意力输出
attn_output_1 = V[3:] @ attn_weights_1  # (6,6)
print("attn_output_1 矩阵 (3×6):")
print(attn_output_1.round(6))
print()
attn_output.append(attn_output_1)

# 合并两个头的输出
attn_output = np.vstack(attn_output)  # (6,6)
print("attn_output 矩阵 (6×6):")
print(attn_output.round(6))
print()

# ==================== 5. 输出线性变换 ====================
# 输出线性变换权重 Wo (5×6)
Wo = np.array([
    [ 0,  0,  1,  1,  0, -1],
    [ 1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1,  1],
    [-1,  0, -1,  1,  0,  1],
    [-1, -1, -1, -1,  0,  1]
])

# ==================== 6. 输出结果 ====================
# 输出结果即为 Transformer 编码器层的输出
output = Wo @ attn_output  # (5,6)
print("output 矩阵 (5×6):")
print(output.round(6))
print()

# ==================== 7. 残差连接与层归一化 ====================
# 残差连接
residual = output + embeddings_pe  # (5,6)
print("残差连接结果 (5×6) - residual:")
print(residual.round(6))
print()

# 层归一化
def layer_norm(X, eps=1e-6):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + eps)
normalized_output = layer_norm(residual)  # (5,6)
print("层归一化结果 (5×6) - normalized_output:")
print(normalized_output.round(6))
print()


# scale & shift
gamma = np.array([
    [1, 0, 0, 0, 0],
    [0, 2, 0, 0, 0],
    [0, 0, 5, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 5]
])  # (5, 5)
beta = np.array([
    [2], [3], [0], [1], [2]
])   # (5,1)
gamma_beta = np.hstack((gamma, beta))  # (5,6)
print("gamma_beta (5×6) - gamma_beta:")
print(gamma_beta.shape)
print(gamma_beta.round(6))
print()

normalized_output_appended = np.vstack((normalized_output, np.ones((1, seq_len))))  # (6,6)
print("normalized_output_appended (6×6) - normalized_output_appended:")
print(normalized_output_appended.shape)
print(normalized_output_appended.round(6))
print()

# final_output = gamma_beta @ normalized_output_appended  # (5,6)
final_output = np.array([
        [2.678, 2.442, 2.362,  2.27, 2.378, 2.131],
        [ 5.58, 1.763, 5.528,  4.81,  5.57, 4.749],
        [-3.15, 3.382, -4.93, -0.32, -4.67, 3.227],
        [0.612, 3.822, 2.256, 2.676, 2.098, 1.065],
        [-4.05,  -5.2,  -3.3, -6.35, -3.48, -6.36]
    ]
)
print("final_output (5×6) - final_output:")
print(final_output.round(6))
print()

# ==================== 8. 最终输出 ====================
final_output_appended = np.vstack((final_output, np.ones((1, seq_len)))) # (6,6)
# final_output_appended = np.array([
#         [2.678, 2.442, 2.362,  2.27, 2.378, 2.131],
#         [ 5.58, 1.763, 5.528,  4.81,  5.57, 4.749],
#         [-3.15, 3.382, -4.93, -0.32, -4.67, 3.227],
#         [0.612, 3.822, 2.256, 2.676, 2.098, 1.065],
#         [-4.05,  -5.2,  -3.3, -6.35, -3.48, -6.36],
#         [    1,     1,     1,     1,     1,     1]
#     ]
# )
print("final_output_appended (6×6) - final_output_appended:")
print(final_output_appended.shape)
print(final_output_appended.round(6))
print()


# 将输入升维到 4 * d_model 维度，这里放大2倍
W_linear_expand = np.array([
    [-1, -1,  1,  1, -1,  0],
    [ 1,  0,  1,  0,  1,  0],
    [-1,  1, -1,  1,  1, -1],
    [ 0,  0,  0, -1,  0, -1],
    [-1,  0,  0,  1, -1,  0],
    [ 0,  1,  0,  1,  1,  1],
    [ 1,  0,  1,  1,  1,  0],
    [-1,  0,  0,  1,  0, -1],
    [-1,  0,  1, -1,  0,  1],
    [ 0,  0,  0, -1,  1, -1],
    [ 1,  1,  0,  0, -1,  0],
    [ 1,  0,  0,  1, -1,  0],
])  # (12, 6)

final_output_expanded = W_linear_expand @ final_output_appended  # (12,6)
# print(final_output_expanded.shape)
print("final_output_expanded (12×6) - final_output_expanded:")
print(final_output_expanded.round(6))
print()

# 激活函数 ReLU
def relu(X):
    return np.maximum(0, X)
final_output_activated = relu(final_output_expanded)  # (12,6)
# print(final_output_activated.shape)
print("final_output_activated (12×6) - final_output_activated:")
print(final_output_activated.round(6))
print()

# 将维度降回 d_model 维度，这里缩小2倍
W_linear_shrink = np.array([ #5*
        [ 0, 1, 0, 0, 0, 0, 1,-1, 1,-1, 0,-1, 0],
        [ 0,-1, 0, 0, 0, 0, 1,-1, 1, 0, 0,-1,-1],
        [-1, 1, 0, 1, 1,-1, 0, 1, 1,-1, 0, 1, 0],
        [ 1, 1, 1, 1,-1, 1,-1,-1, 1, 0, 0,-1,-1],
        [ 0,-1, 0, 1, 0,-1, 1, 1, 0, 1, 0, 1,-1]
    ]
)

final_output_activated_appended = np.vstack((final_output_activated, np.ones((1, seq_len))))  # (6,6)
final_output_shrunk = W_linear_shrink @ final_output_activated_appended  # (5,6)
# print(final_output_shrunk.shape)
print("final_output_shrunk (5×6) - final_output_shrunk:")
print(final_output_shrunk.round(6))
print()

final_output_shrunk_activated = relu(final_output_shrunk)  # (12,6)
# print(final_output_shrunk_activated.shape)
print("final_output_shrunk_activated (12×6) - final_output_shrunk_activated:")
print(final_output_shrunk_activated.round(6))
print()

# Add & Norm
final_residual = final_output_shrunk_activated + final_output  # (5,6)
print(final_residual.shape)
print(final_residual.round(6))
print()

# final_normalized = layer_norm(final_residual)  # (5,6)
final_normalized = np.array([
    [0.191, -0.72, 0.21, -0.56, 0.173, -0.49],
    [1.37, -0.87, 1.384, -0.06, 1.398, -0.03],
    [0.334, 1.518, -0.41, 1.742, -0.24, 1.735],
    [-0.65, -0.42, 0.171, -0.48, 0.066, -0.67],
    [-1.25, 0.493, -1.36, -0.65, -1.4, -0.54]
])
# print(final_normalized.shape)
print("final_normalized (5×6) - final_normalized:")
print(final_normalized.round(6))
print()

# Scale & Shift again
gamma_02 = np.array([
    [2, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 2]
])  # (5, 5)
gamma_beta_02 = np.hstack((gamma_02, beta))  # (5,6)

final_normalized_appended = np.vstack((final_normalized, np.ones((1, seq_len))))  # (6,6)

output_encoder = gamma_beta_02 @ final_normalized_appended  # (5,6)
print("output_encoder (5×6) - output_encoder:")
# print(output_encoder.shape)
print(output_encoder.round(6))
print()


