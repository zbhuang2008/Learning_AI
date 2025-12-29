import numpy as np
import math

# ==================== 从 medium 表提取数据 ====================

# 根据表格结构，medium表有：输入长度8，输入维度6，Key维度4，Value维度6，输出维度6

# 1. 提取输入矩阵 X (6×8)
# 从表格 J14:Q19 行 (x1..x8)
X_medium = np.array([
    [2, 0, 0, 0, 0, 0, 0, 2],  # x1..x8 第1行 (J14:Q14)
    [0, 1, 1, 0, 1, 0, 0, 0],  # x1..x8 第2行 (J15:Q15)
    [0, 2, 2, 0, 0, 2, 1, 0],  # x1..x8 第3行 (J16:Q16)
    [0, 0, 0, 2, 0, 0, 1, 1],  # x1..x8 第4行 (J17:Q17)
    [2, 0, 0, 1, 1, 0, 0, 0],  # x1..x8 第5行 (J18:Q18)
    [1, 0, 0, 0, 0, 1, 1, 1]  # x1..x8 第6行 (J19:Q19)
]).T  # 转置为 8×6

print("1. medium表输入矩阵 X (8×6):")
print("形状:", X_medium.shape)
print(X_medium)
print()

# 2. 提取权重矩阵 Wq, Wk, Wv
# 从表格结构看，medium表的Wq应该是 (4×6)，Wk应该是 (4×6)，Wv应该是 (6×6)

# Wq (4×6) - 从表格中提取（根据small表类推）
Wq_medium = np.array([
    [1, 1, 0, 0, 0, 0],  # 第1行
    [0, 1, 0, 1, 0, 0],  # 第2行
    [0, 1, 0, 1, 0, 0],  # 第3行（注意：medium表中第3行与第2行相同？）
    [0, 0, 1, 0, 1, 1]  # 第4行
])

# Wk (4×6) - 从表格中提取
Wk_medium = np.array([
    [0, 0, 1, 0, 0, 0],  # 第1行
    [0, 1, 0, 0, 0, 0],  # 第2行
    [0, 1, 0, 0, 0, 0],  # 第3行
    [1, 0, 0, 0, 0, -1]  # 第4行
])

# Wv (6×6) - 从表格最后部分提取（V = Wv * X）
Wv_medium = np.array([
    [10, 0, 0, 0, 0, 0],  # 第1行
    [0, 10, 0, 10, 0, 0],  # 第2行
    [0, 10, 5, 10, 0, 0],  # 第3行
    [0, 0, 10, 10, 0, 0],  # 第4行
    [0, 0, 0, 10, 0, 0],  # 第5行
    [0, 10, 0, 0, 0, 0]  # 第6行
])

print("2. medium表权重矩阵:")
print("Wq 形状:", Wq_medium.shape)
print("Wq (4×6):")
print(Wq_medium)
print()

print("Wk 形状:", Wk_medium.shape)
print("Wk (4×6):")
print(Wk_medium)
print()

print("Wv 形状:", Wv_medium.shape)
print("Wv (6×6):")
print(Wv_medium)
print()

# ==================== 计算 Q, K, V ====================

# 计算 Q = Wq * X^T (4×8)
X_T_medium = X_medium.T  # 6×8
Q_medium = Wq_medium @ X_T_medium  # 4×8

# 计算 K = Wk * X^T (4×8)
K_medium = Wk_medium @ X_T_medium  # 4×8

# 计算 V = Wv * X^T (6×8)
V_medium = Wv_medium @ X_T_medium  # 6×8

print("3. 计算得到的 Q, K, V:")
print("Q (4×8) - Query矩阵:")
print(Q_medium)
print()

print("K (4×8) - Key矩阵:")
print(K_medium)
print()

print("V (6×8) - Value矩阵:")
print(V_medium)
print()

# ==================== 计算 MatMul1: K^T * Q ====================

# K^T (8×4) * Q (4×8) = (8×8)
K_T_medium = K_medium.T  # 8×4
R_medium = K_T_medium @ Q_medium  # 8×8

print("4. 计算 MatMul1: R = K^T * Q (8×8):")
print("R 矩阵形状:", R_medium.shape)
print("前4行前4列:")
print(R_medium[:4, :4])
print()

# ==================== 缩放：除以 sqrt(dk) ====================

dk_medium = 4  # Key Dims
scale_factor_medium = math.sqrt(dk_medium)
R_scaled_medium = R_medium / scale_factor_medium

print(f"5. 缩放: R / sqrt({dk_medium}) = R / {scale_factor_medium:.4f}")
print("缩放后的 R 矩阵 (8×8):")
print("前4行前4列:")
print(R_scaled_medium[:4, :4])
print()


# ==================== Softmax ====================

def softmax_by_column(matrix):
    """对矩阵的每一列应用softmax"""
    # 数值稳定性：减去每列的最大值
    max_vals = np.max(matrix, axis=0, keepdims=True)
    exp_matrix = np.exp(matrix - max_vals)
    sum_vals = np.sum(exp_matrix, axis=0, keepdims=True)
    return exp_matrix / sum_vals


A_medium = softmax_by_column(R_scaled_medium)

print("6. Softmax 得到注意力权重矩阵 A (8×8):")
print("A 矩阵形状:", A_medium.shape)
print("每列和为:", np.sum(A_medium, axis=0)[:4], "...")
print("A 矩阵前4行前4列:")
print(A_medium[:4, :4])
print()

# ==================== 计算 MatMul2: Z = V * A ====================

# V (6×8) * A (8×8) = (6×8)
Z_medium = V_medium @ A_medium  # 6×8

print("7. 计算 MatMul2: Z = V * A (6×8) - Attention Weighted Features:")
print("Z 矩阵形状:", Z_medium.shape)
print("Z 矩阵 (6×8):")
for i in range(6):
    print(f"维度 {i + 1}: {Z_medium[i, 0]:.6f}, {Z_medium[i, 1]:.6f}, {Z_medium[i, 2]:.6f}, "
          f"{Z_medium[i, 3]:.6f}, {Z_medium[i, 4]:.6f}, {Z_medium[i, 5]:.6f}, "
          f"{Z_medium[i, 6]:.6f}, {Z_medium[i, 7]:.6f}")
print()

# ==================== 详细验证和中间值计算 ====================

print("=" * 80)
print("详细验证和中间值计算")
print("=" * 80)

# 计算具体的中间值用于验证
print("\n1. 验证维度匹配:")
print(f"   X: {X_medium.shape} (8个token, 每个6维)")
print(f"   Wq: {Wq_medium.shape}, Wk: {Wk_medium.shape}, Wv: {Wv_medium.shape}")
print(f"   Q: {Q_medium.shape}, K: {K_medium.shape}, V: {V_medium.shape}")
print(f"   K^T: {K_T_medium.shape}")
print(f"   R = K^T @ Q: {R_medium.shape}")
print(f"   A = softmax(R/√dk): {A_medium.shape}")
print(f"   Z = V @ A: {Z_medium.shape}")

print("\n2. 计算第一个Query (q1) 的注意力权重:")
print("   q1 =", Q_medium[:, 0])
print("   K^T 的每一行是一个Key:")
for i in range(8):
    print(f"   k{i + 1}^T = {K_T_medium[i, :]}")

print("\n3. 计算 R[0,0] (k1^T * q1):")
k1_T = K_T_medium[0, :]  # 第一行
q1 = Q_medium[:, 0]
R_00 = np.dot(k1_T, q1)
print(f"   k1^T = {k1_T}")
print(f"   q1 = {q1}")
print(f"   R[0,0] = {R_00:.6f}")

print(f"\n4. 缩放: R[0,0] / √{dk_medium} = {R_00} / {scale_factor_medium:.6f} = {R_00 / scale_factor_medium:.6f}")

print(f"\n5. 指数化: exp({R_00 / scale_factor_medium:.6f}) = {math.exp(R_00 / scale_factor_medium):.6f}")

print("\n6. Softmax第一列的计算:")
col_0_exp = np.exp(R_scaled_medium[:, 0] - np.max(R_scaled_medium[:, 0]))
col_0_sum = np.sum(col_0_exp)
print(f"   第一列的指数值 (减去最大值后):")
for i in range(8):
    print(
        f"   A[{i},0] 的指数 = exp({R_scaled_medium[i, 0]:.6f} - {np.max(R_scaled_medium[:, 0]):.6f}) = {col_0_exp[i]:.6f}")
print(f"   第一列指数和 = {col_0_sum:.6f}")
print(f"   第一列Softmax结果:")
for i in range(8):
    A_i0 = col_0_exp[i] / col_0_sum
    print(f"   A[{i},0] = {col_0_exp[i]:.6f} / {col_0_sum:.6f} = {A_i0:.6f}")

print("\n7. 验证Softmax性质:")
for j in range(8):
    col_sum = np.sum(A_medium[:, j])
    if abs(col_sum - 1.0) > 1e-10:
        print(f"   第{j + 1}列和 = {col_sum:.10f} ✗")
    else:
        print(f"   第{j + 1}列和 = {col_sum:.10f} ✓")

print("\n8. 计算最终的输出 z1:")
print("   z1 = v1*A[0,0] + v2*A[1,0] + v3*A[2,0] + v4*A[3,0] + v5*A[4,0] + v6*A[5,0] + v7*A[6,0] + v8*A[7,0]")
z1_calc = np.zeros(6)
for j in range(8):
    z1_calc += V_medium[:, j] * A_medium[j, 0]
print(f"   计算得到的 z1 = {z1_calc}")
print(f"   Z矩阵第一列 = {Z_medium[:, 0]}")
print(f"   是否一致: {np.allclose(z1_calc, Z_medium[:, 0], rtol=1e-10)}")

# ==================== 与Excel中可能的值比对 ====================

print("\n" + "=" * 80)
print("与Excel中可能的值比对参考")
print("=" * 80)

print("\n注意：由于我们无法看到Excel中具体的数值，以下是根据公式计算的参考值")
print("你可以在Excel中验证这些中间值:")

print("\nA) Q矩阵的值 (4×8):")
print("   q1 =", [f"{val:.6f}" for val in Q_medium[:, 0]])
print("   q2 =", [f"{val:.6f}" for val in Q_medium[:, 1]])
print("   q3 =", [f"{val:.6f}" for val in Q_medium[:, 2]])
print("   q4 =", [f"{val:.6f}" for val in Q_medium[:, 3]])

print("\nB) K矩阵的值 (4×8):")
print("   k1 =", [f"{val:.6f}" for val in K_medium[:, 0]])
print("   k2 =", [f"{val:.6f}" for val in K_medium[:, 1]])
print("   k3 =", [f"{val:.6f}" for val in K_medium[:, 2]])
print("   k4 =", [f"{val:.6f}" for val in K_medium[:, 3]])

print("\nC) R矩阵的第一个元素:")
print(f"   R[0,0] = k1^T * q1 = {R_medium[0, 0]:.6f}")
print(f"   R[0,0] / √{dk_medium} = {R_scaled_medium[0, 0]:.6f}")
print(f"   exp(R[0,0]/√{dk_medium}) = {math.exp(R_scaled_medium[0, 0]):.6f}")

print("\nD) 第一列Softmax分母:")
first_col_exp = np.exp(R_scaled_medium[:, 0])
first_col_sum = np.sum(first_col_exp)
print(f"   Σ exp(R[:,0]/√{dk_medium}) = {first_col_sum:.6f}")

print(f"\nE) A[0,0] = {A_medium[0, 0]:.6f}")
print(f"   A[1,0] = {A_medium[1, 0]:.6f}")
print(f"   A[2,0] = {A_medium[2, 0]:.6f}")
print(f"   A[3,0] = {A_medium[3, 0]:.6f}")
print(f"   ...")
print(f"   验证: A[:,0] 和 = {np.sum(A_medium[:, 0]):.10f}")

print("\nF) V矩阵的值 (6×8):")
print("   v1 =", [f"{val:.6f}" for val in V_medium[:, 0]])
print("   v2 =", [f"{val:.6f}" for val in V_medium[:, 1]])
print("   v3 =", [f"{val:.6f}" for val in V_medium[:, 2]])
print("   v4 =", [f"{val:.6f}" for val in V_medium[:, 3]])

print("\nG) 最终的Z矩阵 (6×8):")
print("   z1 =", [f"{val:.6f}" for val in Z_medium[:, 0]])
print("   z2 =", [f"{val:.6f}" for val in Z_medium[:, 1]])
print("   z3 =", [f"{val:.6f}" for val in Z_medium[:, 2]])
print("   z4 =", [f"{val:.6f}" for val in Z_medium[:, 3]])
print("   z5 =", [f"{val:.6f}" for val in Z_medium[:, 4]])
print("   z6 =", [f"{val:.6f}" for val in Z_medium[:, 5]])
print("   z7 =", [f"{val:.6f}" for val in Z_medium[:, 6]])
print("   z8 =", [f"{val:.6f}" for val in Z_medium[:, 7]])

# ==================== 计算完整性和正确性检查 ====================

print("\n" + "=" * 80)
print("计算完整性和正确性检查")
print("=" * 80)

# 1. 检查Softmax列和为1
col_sums = np.sum(A_medium, axis=0)
max_col_error = np.max(np.abs(col_sums - 1.0))
print(f"1. Softmax列和检查: 最大误差 = {max_col_error:.2e} {'✓' if max_col_error < 1e-10 else '✗'}")

# 2. 检查所有值在[0,1]范围内
A_min = np.min(A_medium)
A_max = np.max(A_medium)
print(f"2. 注意力权重范围: [{A_min:.6f}, {A_max:.6f}] {'✓' if A_min >= 0 and A_max <= 1 else '✗'}")

# 3. 检查Z矩阵维度
print(f"3. Z矩阵维度: {Z_medium.shape} = (输出维度6, 输入长度8) {'✓' if Z_medium.shape == (6, 8) else '✗'}")

# 4. 检查注意力模式
print("\n4. 注意力模式分析:")
print("   每个Query最关注的Key:")
for j in range(8):
    max_key_idx = np.argmax(A_medium[:, j])
    max_weight = np.max(A_medium[:, j])
    print(f"   Query {j + 1} → Key {max_key_idx + 1} (权重: {max_weight:.4f})")

# 5. 计算注意力矩阵的稀疏性
threshold = 0.1
sparse_count = np.sum(A_medium < threshold)
sparse_ratio = sparse_count / A_medium.size
print(f"\n5. 注意力矩阵稀疏性:")
print(f"   权重 < {threshold}: {sparse_count}/{A_medium.size} = {sparse_ratio:.2%}")

# 6. 计算注意力分布的熵
print("\n6. 注意力分布熵 (每列):")
for j in range(4):  # 只显示前4个
    col = A_medium[:, j]
    entropy = -np.sum([p * np.log(p + 1e-10) for p in col])  # 加小量避免log(0)
    max_entropy = np.log(8)  # 8个Key的均匀分布
    normalized_entropy = entropy / max_entropy
    print(f"   Query {j + 1}: 熵 = {entropy:.4f}, 归一化 = {normalized_entropy:.4f}")


# ==================== 可视化函数（可选） ====================

def visualize_attention_weights(A_matrix, title="Medium Table 注意力权重矩阵"):
    """可视化注意力权重矩阵"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(A_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=[f'q{i + 1}' for i in range(8)],
                yticklabels=[f'k{i + 1}' for i in range(8)],
                cbar_kws={'label': '注意力权重'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Query (Q)', fontsize=12)
    plt.ylabel('Key (K)', fontsize=12)
    plt.tight_layout()
    plt.show()


print("\n" + "=" * 80)
print("计算完成!")
print("=" * 80)
print(f"\n总结:")
print(f"1. 输入: {X_medium.shape[0]}个token, 每个{X_medium.shape[1]}维")
print(f"2. 输出: {Z_medium.shape[0]}维特征, {Z_medium.shape[1]}个输出位置")
print(f"3. 注意力权重矩阵: {A_medium.shape} (Keys × Queries)")
print(f"4. 最显著的注意力: Query {np.argmax(A_medium) // 8 + 1} → Key {np.argmax(A_medium) % 8 + 1}")
print(f"   权重值: {np.max(A_medium):.4f}")

'''
/Users/zbhuang/miniconda3/envs/ai_by_hand_excel_00_py310/bin/python 
/Users/zbhuang/MyDev/AIProjects/ai-by-hand-excel/src/self_attention_medium.py 

1. medium表输入矩阵 X (8×6):
形状: (8, 6)
[[2 0 0 0 2 1]
 [0 1 2 0 0 0]
 [0 1 2 0 0 0]
 [0 0 0 2 1 0]
 [0 1 0 0 1 0]
 [0 0 2 0 0 1]
 [0 0 1 1 0 1]
 [2 0 0 1 0 1]]

2. medium表权重矩阵:
Wq 形状: (4, 6)
Wq (4×6):
[[1 1 0 0 0 0]
 [0 1 0 1 0 0]
 [0 1 0 1 0 0]
 [0 0 1 0 1 1]]

Wk 形状: (4, 6)
Wk (4×6):
[[ 0  0  1  0  0  0]
 [ 0  1  0  0  0  0]
 [ 0  1  0  0  0  0]
 [ 1  0  0  0  0 -1]]

Wv 形状: (6, 6)
Wv (6×6):
[[10  0  0  0  0  0]
 [ 0 10  0 10  0  0]
 [ 0 10  5 10  0  0]
 [ 0  0 10 10  0  0]
 [ 0  0  0 10  0  0]
 [ 0 10  0  0  0  0]]

3. 计算得到的 Q, K, V:
Q (4×8) - Query矩阵:
[[2 1 1 0 1 0 0 2]
 [0 1 1 2 1 0 1 1]
 [0 1 1 2 1 0 1 1]
 [3 2 2 1 1 3 2 1]]

K (4×8) - Key矩阵:
[[ 0  2  2  0  0  2  1  0]
 [ 0  1  1  0  1  0  0  0]
 [ 0  1  1  0  1  0  0  0]
 [ 1  0  0  0  0 -1 -1  1]]

V (6×8) - Value矩阵:
[[20  0  0  0  0  0  0 20]
 [ 0 10 10 20 10  0 10 10]
 [ 0 20 20 20 10 10 15 10]
 [ 0 20 20 20  0 20 20 10]
 [ 0  0  0 20  0  0 10 10]
 [ 0 10 10  0 10  0  0  0]]

4. 计算 MatMul1: R = K^T * Q (8×8):
R 矩阵形状: (8, 8)
前4行前4列:
[[3 2 2 1]
 [4 4 4 4]
 [4 4 4 4]
 [0 0 0 0]]

5. 缩放: R / sqrt(4) = R / 2.0000
缩放后的 R 矩阵 (8×8):
前4行前4列:
[[1.5 1.  1.  0.5]
 [2.  2.  2.  2. ]
 [2.  2.  2.  2. ]
 [0.  0.  0.  0. ]]

6. Softmax 得到注意力权重矩阵 A (8×8):
A 矩阵形状: (8, 8)
每列和为: [1. 1. 1. 1.] ...
A 矩阵前4行前4列:
[[0.16007895 0.10643447 0.10643447 0.05956864]
 [0.26392557 0.28931888 0.28931888 0.26696812]
 [0.26392557 0.28931888 0.28931888 0.26696812]
 [0.03571844 0.03915505 0.03915505 0.03613021]]

7. 计算 MatMul2: Z = V * A (6×8) - Attention Weighted Features:
Z 矩阵形状: (6, 8)
Z 矩阵 (6×8):
维度 1: 6.403158, 4.257379, 4.257379, 2.382746, 2.698116, 13.368560, 7.094022, 1.236915
维度 2: 8.167497, 8.935655, 8.935655, 9.546475, 9.060065, 7.237197, 8.638913, 9.037757
维度 3: 14.143228, 15.232327, 15.232327, 15.214548, 15.985213, 8.978254, 12.545951, 17.567297
维度 4: 14.483263, 14.678277, 14.678277, 12.873578, 15.752193, 8.482115, 11.132472, 18.052650
维度 5: 2.531802, 2.084933, 2.084933, 1.537431, 1.901896, 5.000000, 3.318396, 0.993571
维度 6: 5.635696, 6.850722, 6.850722, 8.009044, 7.158168, 2.237197, 5.320517, 8.044186

================================================================================
详细验证和中间值计算
================================================================================

1. 验证维度匹配:
   X: (8, 6) (8个token, 每个6维)
   Wq: (4, 6), Wk: (4, 6), Wv: (6, 6)
   Q: (4, 8), K: (4, 8), V: (6, 8)
   K^T: (8, 4)
   R = K^T @ Q: (8, 8)
   A = softmax(R/√dk): (8, 8)
   Z = V @ A: (6, 8)

2. 计算第一个Query (q1) 的注意力权重:
   q1 = [2 0 0 3]
   K^T 的每一行是一个Key:
   k1^T = [0 0 0 1]
   k2^T = [2 1 1 0]
   k3^T = [2 1 1 0]
   k4^T = [0 0 0 0]
   k5^T = [0 1 1 0]
   k6^T = [ 2  0  0 -1]
   k7^T = [ 1  0  0 -1]
   k8^T = [0 0 0 1]

3. 计算 R[0,0] (k1^T * q1):
   k1^T = [0 0 0 1]
   q1 = [2 0 0 3]
   R[0,0] = 3.000000

4. 缩放: R[0,0] / √4 = 3 / 2.000000 = 1.500000

5. 指数化: exp(1.500000) = 4.481689

6. Softmax第一列的计算:
   第一列的指数值 (减去最大值后):
   A[0,0] 的指数 = exp(1.500000 - 2.000000) = 0.606531
   A[1,0] 的指数 = exp(2.000000 - 2.000000) = 1.000000
   A[2,0] 的指数 = exp(2.000000 - 2.000000) = 1.000000
   A[3,0] 的指数 = exp(0.000000 - 2.000000) = 0.135335
   A[4,0] 的指数 = exp(0.000000 - 2.000000) = 0.135335
   A[5,0] 的指数 = exp(0.500000 - 2.000000) = 0.223130
   A[6,0] 的指数 = exp(-0.500000 - 2.000000) = 0.082085
   A[7,0] 的指数 = exp(1.500000 - 2.000000) = 0.606531
   第一列指数和 = 3.788947
   第一列Softmax结果:
   A[0,0] = 0.606531 / 3.788947 = 0.160079
   A[1,0] = 1.000000 / 3.788947 = 0.263926
   A[2,0] = 1.000000 / 3.788947 = 0.263926
   A[3,0] = 0.135335 / 3.788947 = 0.035718
   A[4,0] = 0.135335 / 3.788947 = 0.035718
   A[5,0] = 0.223130 / 3.788947 = 0.058890
   A[6,0] = 0.082085 / 3.788947 = 0.021664
   A[7,0] = 0.606531 / 3.788947 = 0.160079

7. 验证Softmax性质:
   第1列和 = 1.0000000000 ✓
   第2列和 = 1.0000000000 ✓
   第3列和 = 1.0000000000 ✓
   第4列和 = 1.0000000000 ✓
   第5列和 = 1.0000000000 ✓
   第6列和 = 1.0000000000 ✓
   第7列和 = 1.0000000000 ✓
   第8列和 = 1.0000000000 ✓

8. 计算最终的输出 z1:
   z1 = v1*A[0,0] + v2*A[1,0] + v3*A[2,0] + v4*A[3,0] + v5*A[4,0] + v6*A[5,0] + v7*A[6,0] + v8*A[7,0]
   计算得到的 z1 = [ 6.40315795  8.16749739 14.14322793 14.48326271  2.53180162  5.63569577]
   Z矩阵第一列 = [ 6.40315795  8.16749739 14.14322793 14.48326271  2.53180162  5.63569577]
   是否一致: True

================================================================================
与Excel中可能的值比对参考
================================================================================

注意：由于我们无法看到Excel中具体的数值，以下是根据公式计算的参考值
你可以在Excel中验证这些中间值:

A) Q矩阵的值 (4×8):
   q1 = ['2.000000', '0.000000', '0.000000', '3.000000']
   q2 = ['1.000000', '1.000000', '1.000000', '2.000000']
   q3 = ['1.000000', '1.000000', '1.000000', '2.000000']
   q4 = ['0.000000', '2.000000', '2.000000', '1.000000']

B) K矩阵的值 (4×8):
   k1 = ['0.000000', '0.000000', '0.000000', '1.000000']
   k2 = ['2.000000', '1.000000', '1.000000', '0.000000']
   k3 = ['2.000000', '1.000000', '1.000000', '0.000000']
   k4 = ['0.000000', '0.000000', '0.000000', '0.000000']

C) R矩阵的第一个元素:
   R[0,0] = k1^T * q1 = 3.000000
   R[0,0] / √4 = 1.500000
   exp(R[0,0]/√4) = 4.481689

D) 第一列Softmax分母:
   Σ exp(R[:,0]/√4) = 27.996742

E) A[0,0] = 0.160079
   A[1,0] = 0.263926
   A[2,0] = 0.263926
   A[3,0] = 0.035718
   ...
   验证: A[:,0] 和 = 1.0000000000

F) V矩阵的值 (6×8):
   v1 = ['20.000000', '0.000000', '0.000000', '0.000000', '0.000000', '0.000000']
   v2 = ['0.000000', '10.000000', '20.000000', '20.000000', '0.000000', '10.000000']
   v3 = ['0.000000', '10.000000', '20.000000', '20.000000', '0.000000', '10.000000']
   v4 = ['0.000000', '20.000000', '20.000000', '20.000000', '20.000000', '0.000000']

G) 最终的Z矩阵 (6×8):
   z1 = ['6.403158', '8.167497', '14.143228', '14.483263', '2.531802', '5.635696']
   z2 = ['4.257379', '8.935655', '15.232327', '14.678277', '2.084933', '6.850722']
   z3 = ['4.257379', '8.935655', '15.232327', '14.678277', '2.084933', '6.850722']
   z4 = ['2.382746', '9.546475', '15.214548', '12.873578', '1.537431', '8.009044']
   z5 = ['2.698116', '9.060065', '15.985213', '15.752193', '1.901896', '7.158168']
   z6 = ['13.368560', '7.237197', '8.978254', '8.482115', '5.000000', '2.237197']
   z7 = ['7.094022', '8.638913', '12.545951', '11.132472', '3.318396', '5.320517']
   z8 = ['1.236915', '9.037757', '17.567297', '18.052650', '0.993571', '8.044186']

================================================================================
计算完整性和正确性检查
================================================================================
1. Softmax列和检查: 最大误差 = 1.11e-16 ✓
2. 注意力权重范围: [0.016640, 0.376718] ✓
3. Z矩阵维度: (6, 8) = (输出维度6, 输入长度8) ✓

4. 注意力模式分析:
   每个Query最关注的Key:
   Query 1 → Key 2 (权重: 0.2639)
   Query 2 → Key 2 (权重: 0.2893)
   Query 3 → Key 2 (权重: 0.2893)
   Query 4 → Key 2 (权重: 0.2670)
   Query 5 → Key 2 (权重: 0.3023)
   Query 6 → Key 1 (权重: 0.3342)
   Query 7 → Key 1 (权重: 0.1774)
   Query 8 → Key 2 (权重: 0.3767)

5. 注意力矩阵稀疏性:
   权重 < 0.1: 35/64 = 54.69%

6. 注意力分布熵 (每列):
   Query 1: 熵 = 1.7775, 归一化 = 0.8548
   Query 2: 熵 = 1.7755, 归一化 = 0.8538
   Query 3: 熵 = 1.7755, 归一化 = 0.8538
   Query 4: 熵 = 1.6812, 归一化 = 0.8085

================================================================================
计算完成!
================================================================================

总结:
1. 输入: 8个token, 每个6维
2. 输出: 6维特征, 8个输出位置
3. 注意力权重矩阵: (8, 8) (Keys × Queries)
4. 最显著的注意力: Query 2 → Key 8
   权重值: 0.3767

Process finished with exit code 0
'''