import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pdb

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
combined_vectors = np.hstack([feature_vectors, variance_vectors])

print(f"Combined vectors shape: {combined_vectors.shape}")

# 标准化拼接后的向量
scaler = StandardScaler()
combined_vectors_scaled = scaler.fit_transform(combined_vectors)
print(f"Scaled combined vectors shape: {combined_vectors_scaled.shape}")

# 获取标签
labels = df_features['image_label'].iloc[:min_samples].values
unique_labels = np.unique(labels)
print(f"Unique labels: {unique_labels}")
print(f"Label distribution: {np.unique(labels, return_counts=True)}")

# ================== 对比学习分析 ==================
print("\n" + "="*60)
print("CONTRASTIVE LEARNING ANALYSIS")
print("="*60)

# 1. 计算余弦相似度矩阵
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

print("\nComputing similarity matrices...")
cosine_sim_matrix = cosine_similarity(combined_vectors_scaled)
euclidean_dist_matrix = euclidean_distances(combined_vectors_scaled)

print(f"Cosine similarity matrix shape: {cosine_sim_matrix.shape}")
print(f"Euclidean distance matrix shape: {euclidean_dist_matrix.shape}")

# 2. 创建标签掩码
def create_label_masks(labels):
    """创建正样本对和负样本对的掩码"""
    n = len(labels)
    positive_mask = np.zeros((n, n), dtype=bool)
    negative_mask = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i != j:  # 排除自己与自己的对比
                if labels[i] == labels[j]:
                    positive_mask[i, j] = True
                else:
                    negative_mask[i, j] = True
    
    return positive_mask, negative_mask

positive_mask, negative_mask = create_label_masks(labels)
print(f"Positive pairs: {positive_mask.sum()}")
print(f"Negative pairs: {negative_mask.sum()}")

# 3. 计算类内和类间相似度/距离
positive_cosine_similarities = cosine_sim_matrix[positive_mask]
negative_cosine_similarities = cosine_sim_matrix[negative_mask]
positive_euclidean_distances = euclidean_dist_matrix[positive_mask]
negative_euclidean_distances = euclidean_dist_matrix[negative_mask]

print(f"\nSimilarity Analysis:")
print(f"Positive pairs (same class) - Cosine similarity: {positive_cosine_similarities.mean():.4f} ± {positive_cosine_similarities.std():.4f}")
print(f"Negative pairs (diff class) - Cosine similarity: {negative_cosine_similarities.mean():.4f} ± {negative_cosine_similarities.std():.4f}")
print(f"Positive pairs (same class) - Euclidean distance: {positive_euclidean_distances.mean():.4f} ± {positive_euclidean_distances.std():.4f}")
print(f"Negative pairs (diff class) - Euclidean distance: {negative_euclidean_distances.mean():.4f} ± {negative_euclidean_distances.std():.4f}")

# 4. 计算对比学习相关指标
def compute_contrastive_metrics(similarities, distances, pos_mask, neg_mask):
    """计算对比学习相关指标"""
    # 类内相似度 vs 类间相似度
    intra_class_sim = similarities[pos_mask].mean()
    inter_class_sim = similarities[neg_mask].mean()
    
    # 类内距离 vs 类间距离
    intra_class_dist = distances[pos_mask].mean()
    inter_class_dist = distances[neg_mask].mean()
    
    # 分离度指标
    similarity_separation = intra_class_sim - inter_class_sim
    distance_separation = inter_class_dist - intra_class_dist
    
    return {
        'intra_class_similarity': intra_class_sim,
        'inter_class_similarity': inter_class_sim,
        'intra_class_distance': intra_class_dist,
        'inter_class_distance': inter_class_dist,
        'similarity_separation': similarity_separation,
        'distance_separation': distance_separation
    }

metrics = compute_contrastive_metrics(cosine_sim_matrix, euclidean_dist_matrix, positive_mask, negative_mask)

print(f"\nContrastive Learning Metrics:")
print(f"Intra-class similarity: {metrics['intra_class_similarity']:.4f}")
print(f"Inter-class similarity: {metrics['inter_class_similarity']:.4f}")
print(f"Similarity separation: {metrics['similarity_separation']:.4f}")
print(f"Intra-class distance: {metrics['intra_class_distance']:.4f}")
print(f"Inter-class distance: {metrics['inter_class_distance']:.4f}")
print(f"Distance separation: {metrics['distance_separation']:.4f}")

# 5. t-SNE降维可视化
from sklearn.manifold import TSNE

print("\nPerforming t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, min_samples//4))
tsne_embedding = tsne.fit_transform(combined_vectors_scaled)

# ================== 可视化 ==================
print("\nCreating visualizations...")

# 图1: 相似度分布对比
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Contrastive Learning Analysis: Morphological + Semantic Features\n'
             f'Separation Score: {metrics["similarity_separation"]:.4f}', 
             fontsize=16, fontweight='bold', y=0.98)

# 1.1 余弦相似度分布
axes[0,0].hist(positive_cosine_similarities, bins=50, alpha=0.7, label='Same Class (Positive)', color='green', density=True)
axes[0,0].hist(negative_cosine_similarities, bins=50, alpha=0.7, label='Different Class (Negative)', color='red', density=True)
axes[0,0].axvline(positive_cosine_similarities.mean(), color='green', linestyle='--', linewidth=2)
axes[0,0].axvline(negative_cosine_similarities.mean(), color='red', linestyle='--', linewidth=2)
axes[0,0].set_xlabel('Cosine Similarity')
axes[0,0].set_ylabel('Density')
axes[0,0].set_title('Cosine Similarity Distribution')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 1.2 欧氏距离分布
axes[0,1].hist(positive_euclidean_distances, bins=50, alpha=0.7, label='Same Class (Positive)', color='green', density=True)
axes[0,1].hist(negative_euclidean_distances, bins=50, alpha=0.7, label='Different Class (Negative)', color='red', density=True)
axes[0,1].axvline(positive_euclidean_distances.mean(), color='green', linestyle='--', linewidth=2)
axes[0,1].axvline(negative_euclidean_distances.mean(), color='red', linestyle='--', linewidth=2)
axes[0,1].set_xlabel('Euclidean Distance')
axes[0,1].set_ylabel('Density')
axes[0,1].set_title('Euclidean Distance Distribution')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 1.3 t-SNE可视化
label_to_color = {'Lymphocytes': 'blue', 'Tumor Cells': 'orange'}
for label in unique_labels:
    mask = labels == label
    axes[1,0].scatter(tsne_embedding[mask, 0], tsne_embedding[mask, 1], 
                     c=label_to_color[label], label=label, alpha=0.6, s=20)
axes[1,0].set_xlabel('t-SNE 1')
axes[1,0].set_ylabel('t-SNE 2')
axes[1,0].set_title('t-SNE Visualization')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 1.4 对比指标条形图
metrics_names = ['Intra-class\nSimilarity', 'Inter-class\nSimilarity', 'Similarity\nSeparation',
                'Intra-class\nDistance', 'Inter-class\nDistance', 'Distance\nSeparation']
metrics_values = [metrics['intra_class_similarity'], metrics['inter_class_similarity'], metrics['similarity_separation'],
                 metrics['intra_class_distance'], metrics['inter_class_distance'], metrics['distance_separation']]
colors = ['green', 'red', 'purple', 'green', 'red', 'purple']

bars = axes[1,1].bar(range(len(metrics_names)), metrics_values, color=colors, alpha=0.7)
axes[1,1].set_xticks(range(len(metrics_names)))
axes[1,1].set_xticklabels(metrics_names, rotation=45, ha='right')
axes[1,1].set_ylabel('Score')
axes[1,1].set_title('Contrastive Learning Metrics')
axes[1,1].grid(True, alpha=0.3)

# 添加数值标签
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                  f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig('contrastive_learning_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 图2: 相似度矩阵热力图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Similarity Matrices Visualization', fontsize=16, fontweight='bold')

# 对标签进行排序以便更好地可视化块结构
sorted_indices = np.argsort(labels)
sorted_cosine_sim = cosine_sim_matrix[sorted_indices][:, sorted_indices]
sorted_euclidean_dist = euclidean_dist_matrix[sorted_indices][:, sorted_indices]

# 余弦相似度矩阵
im1 = axes[0].imshow(sorted_cosine_sim, cmap='RdYlBu_r', aspect='auto')
axes[0].set_title('Cosine Similarity Matrix\n(Sorted by Labels)')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Sample Index')
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# 欧氏距离矩阵
im2 = axes[1].imshow(sorted_euclidean_dist, cmap='viridis', aspect='auto')
axes[1].set_title('Euclidean Distance Matrix\n(Sorted by Labels)')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Sample Index')
plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.savefig('similarity_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# 图3: 每个类别的详细分析
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Per-Class Contrastive Analysis', fontsize=16, fontweight='bold')

# 计算每个类别内部的相似度
for idx, label in enumerate(unique_labels):
    label_mask = labels == label
    label_indices = np.where(label_mask)[0]
    
    # 类内相似度
    intra_similarities = []
    for i in label_indices:
        for j in label_indices:
            if i != j:
                intra_similarities.append(cosine_sim_matrix[i, j])
    
    axes[0, idx].hist(intra_similarities, bins=30, alpha=0.7, color=label_to_color[label])
    axes[0, idx].axvline(np.mean(intra_similarities), color='black', linestyle='--', linewidth=2)
    axes[0, idx].set_title(f'{label}\nIntra-class Similarity')
    axes[0, idx].set_xlabel('Cosine Similarity')
    axes[0, idx].set_ylabel('Frequency')
    axes[0, idx].grid(True, alpha=0.3)
    axes[0, idx].text(0.05, 0.95, f'Mean: {np.mean(intra_similarities):.3f}\nStd: {np.std(intra_similarities):.3f}',
                     transform=axes[0, idx].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 类间相似度分析
inter_similarities = cosine_sim_matrix[labels == unique_labels[0]][:, labels == unique_labels[1]]
axes[1, 0].hist(inter_similarities.flatten(), bins=30, alpha=0.7, color='purple')
axes[1, 0].axvline(np.mean(inter_similarities), color='black', linestyle='--', linewidth=2)
axes[1, 0].set_title('Inter-class Similarity\n(Lymphocytes vs Tumor Cells)')
axes[1, 0].set_xlabel('Cosine Similarity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].text(0.05, 0.95, f'Mean: {np.mean(inter_similarities):.3f}\nStd: {np.std(inter_similarities):.3f}',
               transform=axes[1, 0].transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 分离度分析（violin plot）
all_similarities = [positive_cosine_similarities, negative_cosine_similarities]
labels_for_violin = ['Same Class\n(Positive)', 'Different Class\n(Negative)']
violin_parts = axes[1, 1].violinplot(all_similarities, positions=[1, 2], showmeans=True, showmedians=True)
axes[1, 1].set_xticks([1, 2])
axes[1, 1].set_xticklabels(labels_for_violin)
axes[1, 1].set_ylabel('Cosine Similarity')
axes[1, 1].set_title('Similarity Distribution Comparison')
axes[1, 1].grid(True, alpha=0.3)

# 设置violin plot颜色
colors = ['green', 'red']
for pc, color in zip(violin_parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

plt.tight_layout()
plt.savefig('per_class_contrastive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ================== 保存结果 ==================
print("\nSaving results...")

# 保存分析结果
contrastive_results = {
    'metrics': metrics,
    'positive_similarities_stats': {
        'mean': float(positive_cosine_similarities.mean()),
        'std': float(positive_cosine_similarities.std()),
        'median': float(np.median(positive_cosine_similarities)),
        'min': float(positive_cosine_similarities.min()),
        'max': float(positive_cosine_similarities.max())
    },
    'negative_similarities_stats': {
        'mean': float(negative_cosine_similarities.mean()),
        'std': float(negative_cosine_similarities.std()),
        'median': float(np.median(negative_cosine_similarities)),
        'min': float(negative_cosine_similarities.min()),
        'max': float(negative_cosine_similarities.max())
    },
    'sample_info': {
        'total_samples': int(min_samples),
        'positive_pairs': int(positive_mask.sum()),
        'negative_pairs': int(negative_mask.sum()),
        'feature_dimension': int(combined_vectors.shape[1])
    }
}

import json
with open('contrastive_analysis_results.json', 'w') as f:
    json.dump(contrastive_results, f, indent=2)

# 保存嵌入向量和标签
results_df = pd.DataFrame()
results_df['image_source'] = df_features['image_source'].iloc[:min_samples]
results_df['image_label'] = labels
results_df['tsne_x'] = tsne_embedding[:, 0]
results_df['tsne_y'] = tsne_embedding[:, 1]
results_df['combined_vector'] = [row for row in combined_vectors_scaled]

results_df.to_pickle('contrastive_learning_results.pkl')
results_df.to_csv('contrastive_learning_results.csv', index=False)

print(f"\n" + "="*60)
print("CONTRASTIVE LEARNING ANALYSIS SUMMARY")
print("="*60)
print(f"Feature separation quality: {'EXCELLENT' if metrics['similarity_separation'] > 0.3 else 'GOOD' if metrics['similarity_separation'] > 0.1 else 'MODERATE'}")
print(f"Similarity separation score: {metrics['similarity_separation']:.4f}")
print(f"Distance separation score: {metrics['distance_separation']:.4f}")
print(f"Intra-class cohesion: {metrics['intra_class_similarity']:.4f}")
print(f"Inter-class distinction: {metrics['inter_class_similarity']:.4f}")

print(f"\nGenerated files:")
print("- contrastive_learning_analysis.png: Main analysis dashboard")
print("- similarity_matrices.png: Similarity matrix visualizations")
print("- per_class_contrastive_analysis.png: Detailed per-class analysis")
print("- contrastive_analysis_results.json: Numerical results")
print("- contrastive_learning_results.pkl: Complete results with embeddings")
print("- contrastive_learning_results.csv: Results in CSV format")
print("="*60)
