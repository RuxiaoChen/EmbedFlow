import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("Loading both DataFrames...")

# 读取两个DataFrame
df_features = pd.read_pickle("variables_combined_vectors.pkl")
df_vectors = pd.read_pickle("vectors_dataframe.pkl")

print(f"Features DataFrame shape: {df_features.shape}")
print(f"Vectors DataFrame shape: {df_vectors.shape}")

# 检查数据结构
print(f"\nFeatures DataFrame columns: {list(df_features.columns)}")
print(f"Vectors DataFrame columns: {list(df_vectors.columns)}")

# 检查向量维度
print(f"\nFeature vector dimension: {df_features['feature_vector'].iloc[0].shape}")
print(f"80% variance vector dimension: {df_vectors['vector_80_variance'].iloc[0].shape}")

# 检查数据量是否匹配
print(f"\nData size comparison:")
print(f"Features: {len(df_features)} samples")
print(f"Vectors: {len(df_vectors)} samples")

# 由于数据量可能不同，我们需要处理匹配问题
# 假设按照index或者某种顺序匹配，取较小的数量
min_samples = min(len(df_features), len(df_vectors))
print(f"Using {min_samples} samples for analysis")

# 提取向量数据
feature_vectors = np.vstack(df_features['feature_vector'].iloc[:min_samples].values)
variance_vectors = np.vstack(df_vectors['vector_80_variance'].iloc[:min_samples].values)

print(f"\nFeature vectors matrix shape: {feature_vectors.shape}")
print(f"Variance vectors matrix shape: {variance_vectors.shape}")

# 拼接向量
# combined_vectors = np.hstack([feature_vectors, variance_vectors])

combined_vectors = np.hstack([feature_vectors])

print(f"Combined vectors shape: {combined_vectors.shape}")

# 标准化拼接后的向量
scaler = StandardScaler()
combined_vectors_scaled = scaler.fit_transform(combined_vectors)
print(f"Scaled combined vectors shape: {combined_vectors_scaled.shape}")

# ================== 确定最佳聚类数量 ==================
print("\nDetermining optimal number of clusters...")

# 使用肘部法则和轮廓系数
max_clusters = min(10, min_samples // 10)  # 合理的最大聚类数
sse = []
silhouette_scores = []
k_range = range(2, max_clusters + 1)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(combined_vectors_scaled)
    sse.append(kmeans.inertia_)
    
    silhouette_avg = silhouette_score(combined_vectors_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    print(f"k={k}: SSE={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.3f}")

# 找到最佳k值（轮廓系数最高）
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")

# ================== 聚类分析 ==================
print(f"\nPerforming K-means clustering with k={optimal_k}...")

# 使用最佳k值进行聚类
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(combined_vectors_scaled)

print(f"Clustering completed. Cluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"Cluster {cluster}: {count} samples")

# ================== PCA降维用于可视化 ==================
print("\nPerforming PCA for visualization...")

# 2D PCA
pca_2d = PCA(n_components=2, random_state=42)
vectors_2d = pca_2d.fit_transform(combined_vectors_scaled)
print(f"2D PCA explained variance: {pca_2d.explained_variance_ratio_.sum():.3f}")

# 3D PCA
pca_3d = PCA(n_components=3, random_state=42)
vectors_3d = pca_3d.fit_transform(combined_vectors_scaled)
print(f"3D PCA explained variance: {pca_3d.explained_variance_ratio_.sum():.3f}")

# ================== 可视化 ==================
print("\nCreating visualizations...")

# 创建颜色映射
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

# 图1: 肘部法则和轮廓系数
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 肘部法则
ax1.plot(k_range, sse, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Sum of Squared Errors (SSE)')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True)

# 轮廓系数
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k={optimal_k}')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score vs Number of Clusters')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 图2: 2D聚类可视化
plt.figure(figsize=(12, 8))
for i in range(optimal_k):
    mask = cluster_labels == i
    plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title('2D Clustering Visualization (Combined Feature + Semantic Vectors)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('clustering_2d.png', dpi=300, bbox_inches='tight')
plt.show()

# 图3: 3D聚类可视化
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(optimal_k):
    mask = cluster_labels == i
    ax.scatter(vectors_3d[mask, 0], vectors_3d[mask, 1], vectors_3d[mask, 2],
               c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
ax.set_title('3D Clustering Visualization (Combined Feature + Semantic Vectors)')
ax.legend()

plt.savefig('clustering_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# ================== 创建结果DataFrame ==================
print("\nCreating results DataFrame...")

# 创建包含聚类结果的DataFrame
results_df = pd.DataFrame()
results_df['image_source'] = df_features['image_source'].iloc[:min_samples]
results_df['image_label'] = df_features['image_label'].iloc[:min_samples]
results_df['cluster'] = cluster_labels
results_df['pc1'] = vectors_2d[:, 0]
results_df['pc2'] = vectors_2d[:, 1]
results_df['pc3'] = vectors_3d[:, 0]
results_df['combined_vector'] = [row for row in combined_vectors_scaled]

print(f"Results DataFrame shape: {results_df.shape}")

# 保存结果
results_df.to_pickle("clustering_results.pkl")
results_df.to_csv("clustering_results.csv", index=False)
print("\nResults saved to clustering_results.pkl and clustering_results.csv")

# 显示聚类统计
print("\n" + "="*60)
print("CLUSTERING ANALYSIS SUMMARY")
print("="*60)
print(f"Total samples analyzed: {min_samples}")
print(f"Combined vector dimension: {combined_vectors.shape[1]}")
print(f"Optimal number of clusters: {optimal_k}")
print(f"Silhouette score: {silhouette_scores[optimal_k-2]:.3f}")

print("\nCluster distribution by image label:")
cluster_label_crosstab = pd.crosstab(results_df['image_label'], results_df['cluster'])
print(cluster_label_crosstab)

print("\n" + "="*60)
print("Analysis completed successfully!")
print("Generated files:")
print("- clustering_analysis.png: Elbow method and silhouette analysis")
print("- clustering_2d.png: 2D PCA visualization")
print("- clustering_3d.png: 3D PCA visualization")
print("- clustering_results.pkl: Complete results DataFrame")
print("- clustering_results.csv: Results in CSV format")
print("="*60) 