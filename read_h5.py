import h5py
import pandas as pd
import numpy as np
import pdb
from numpy.linalg import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 打开 h5 文件（r = read）
with h5py.File('semantic_vectors.h5', 'r') as f:
    raw_filenames = f['filenames'][:]
    filenames = [name.decode('utf-8') for name in raw_filenames]  # 手动 UTF-8 解码
    vectors = f['vectors'][:]
pdb.set_trace()
# L2归一化处理
normalized_vectors = vectors / np.expand_dims(norm(vectors, axis=1), axis=1)

# 创建DataFrame
df_vectors = pd.DataFrame({
    'filename': filenames,
    'vector': list(normalized_vectors)  # 将归一化后的向量存储
})

# df_variables = pd.read_excel('image_level_quantitative_features_cleaned.xlsx')

# 将向量转换为矩阵形式进行PCA分析
vector_matrix = np.vstack(df_vectors['vector'].values)

# 执行PCA分析
pca = PCA()
pca_result = pca.fit_transform(vector_matrix)

# 创建可视化图表
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Explained Variance Ratio
axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 'bo-', markersize=4)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Explained Variance Ratio')
axes[0, 0].set_title('Explained Variance Ratio by Component')
axes[0, 0].grid(True, alpha=0.3)

# 2. Cumulative Explained Variance (Weights Accumulation)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
axes[0, 1].plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 'ro-', markersize=4)
axes[0, 1].axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
axes[0, 1].axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Cumulative Explained Variance')
axes[0, 1].set_title('Cumulative Explained Variance (Weights Accumulation)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. PCA 2D Scatter Plot (First two components)
scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=range(len(pca_result)), cmap='viridis', alpha=0.6, s=20)
axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
axes[1, 0].set_title('PCA 2D Projection')
plt.colorbar(scatter, ax=axes[1, 0], label='Sample Index')

# 4. First 10 Components Variance
axes[1, 1].bar(range(1, 11), pca.explained_variance_ratio_[:10])
axes[1, 1].set_xlabel('Principal Component')
axes[1, 1].set_ylabel('Explained Variance Ratio')
axes[1, 1].set_title('Top 10 Components Variance')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印分析结果
print("=== PCA Analysis Results ===")
print(f"Original vector dimension: {vector_matrix.shape[1]}")
print(f"Number of samples: {vector_matrix.shape[0]}")
print(f"Top 5 components explain {cumulative_variance[4]:.3f} of total variance")
print(f"Top 10 components explain {cumulative_variance[9]:.3f} of total variance")
print(f"Components needed for 90% variance: {np.argmax(cumulative_variance >= 0.90) + 1}")
print(f"Components needed for 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")

# 将PCA结果添加到DataFrame
df_vectors['pc1'] = pca_result[:, 0]
df_vectors['pc2'] = pca_result[:, 1]
df_vectors['pc3'] = pca_result[:, 2]

pdb.set_trace() 