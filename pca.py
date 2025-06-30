import torch
import pandas as pd
import os
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图表样式
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 获取 finetune_vectors_sft 文件夹下所有的 .pt 文件
pt_files = glob.glob("finetune_vectors_sft/*.pt")

print(f"Found {len(pt_files)} .pt files")

# 存储数据的列表
data_list = []

def l2_normalize(vector):
    """Perform L2 normalization on the vector"""
    if isinstance(vector, np.ndarray):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    elif isinstance(vector, torch.Tensor):
        norm = torch.norm(vector)
        return (vector / norm).numpy() if norm > 0 else vector.numpy()
    return vector

# 遍历所有 .pt 文件
for pt_file in pt_files:
    try:
        # 加载 tensor 数据
        tensor_data = torch.load(pt_file, map_location="cpu")
        
        # 获取文件名（不包含路径）
        filename = os.path.basename(pt_file)
        
        # 将 tensor 转换为 numpy 数组并进行L2归一化
        if isinstance(tensor_data, torch.Tensor):
            vector = l2_normalize(tensor_data)
        else:
            vector = l2_normalize(tensor_data)
        
        # 确保向量是一维的（从(1, 3584)转换为(3584,)）
        if vector.ndim > 1:
            vector = vector.flatten()
        
        # 添加到数据列表
        data_list.append({
            'filename': filename,
            'vector': vector
        })
        
        print(f"Successfully read and normalized: {filename}, vector shape: {vector.shape if hasattr(vector, 'shape') else 'N/A'}")
        
    except Exception as e:
        print(f"Error reading file {pt_file}: {e}")

# 创建 DataFrame
df = pd.DataFrame(data_list)

print(f"\nCreated DataFrame with shape: {df.shape}")
print("\nFirst few rows of DataFrame:")
print(df.head())

# 保存完整的DataFrame（包含向量）到pickle文件
df.to_pickle("vectors_dataframe.pkl")
print("\nComplete DataFrame with vectors saved to vectors_dataframe.pkl")

# ================== PCA Analysis Section ==================
print("\nStarting PCA analysis...")

# Combine all vectors into a matrix
vectors_matrix = np.vstack(df['vector'].values)
print(f"Vector matrix shape: {vectors_matrix.shape}")

# Standardization (optional, since we already did L2 normalization)
scaler = StandardScaler()
vectors_scaled = scaler.fit_transform(vectors_matrix)

# Perform PCA analysis
pca_2d = PCA(n_components=2)
pca_3d = PCA(n_components=3)
pca_explained = PCA()  # For analyzing explained variance

# 2D PCA
vectors_pca_2d = pca_2d.fit_transform(vectors_scaled)
print(f"2D PCA explained variance ratio: {pca_2d.explained_variance_ratio_}")
print(f"2D PCA cumulative explained variance: {pca_2d.explained_variance_ratio_.sum():.4f}")

# 3D PCA
vectors_pca_3d = pca_3d.fit_transform(vectors_scaled)
print(f"3D PCA explained variance ratio: {pca_3d.explained_variance_ratio_}")
print(f"3D PCA cumulative explained variance: {pca_3d.explained_variance_ratio_.sum():.4f}")

# Analyze explained variance
pca_explained.fit(vectors_scaled)
explained_variance_ratio = pca_explained.explained_variance_ratio_

# 找到累积方差达到80%的主成分个数
cumulative_variance = np.cumsum(explained_variance_ratio)
n_components_80 = np.where(cumulative_variance >= 0.8)[0][0] + 1  # +1因为索引从0开始

print(f"Number of components for 80% variance: {n_components_80}")
print(f"Actual cumulative variance with {n_components_80} components: {cumulative_variance[n_components_80-1]:.4f}")

# 使用找到的主成分个数进行PCA降维
pca_80 = PCA(n_components=n_components_80)
vectors_pca_80 = pca_80.fit_transform(vectors_scaled)

print(f"Shape of 80% variance vectors: {vectors_pca_80.shape}")

# 将80%方差的向量添加到DataFrame中
df['vector_80_variance'] = [row for row in vectors_pca_80]

print(f"Added 80% variance vectors to DataFrame as new column 'vector_80_variance'")

# 保存更新后的DataFrame
df.to_pickle("vectors_dataframe.pkl")
print("Updated DataFrame with 80% variance vectors saved to vectors_dataframe.pkl")

