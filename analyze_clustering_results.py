import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 读取聚类结果
print("Loading clustering results...")
df_results = pd.read_pickle("clustering_results.pkl")

print(f"Results DataFrame shape: {df_results.shape}")
print(f"Columns: {list(df_results.columns)}")

# 基本统计
print("\n" + "="*50)
print("CLUSTERING RESULTS ANALYSIS")
print("="*50)

# 数据分布
print("\nData distribution:")
print(f"Total samples: {len(df_results)}")
print(f"Labels distribution:")
print(df_results['image_label'].value_counts())

print(f"\nCluster distribution:")
print(df_results['cluster'].value_counts())

# 交叉表分析
print("\nCross-tabulation (Label vs Cluster):")
crosstab = pd.crosstab(df_results['image_label'], df_results['cluster'], margins=True)
print(crosstab)

# 计算聚类的"准确性"（假设我们将聚类结果作为预测结果）
# 我们需要确定哪个聚类对应哪个标签
cluster_0_lymph = len(df_results[(df_results['cluster'] == 0) & (df_results['image_label'] == 'Lymphocytes')])
cluster_0_tumor = len(df_results[(df_results['cluster'] == 0) & (df_results['image_label'] == 'Tumor Cells')])
cluster_1_lymph = len(df_results[(df_results['cluster'] == 1) & (df_results['image_label'] == 'Lymphocytes')])
cluster_1_tumor = len(df_results[(df_results['cluster'] == 1) & (df_results['image_label'] == 'Tumor Cells')])

print(f"\nDetailed breakdown:")
print(f"Cluster 0: {cluster_0_lymph} Lymphocytes, {cluster_0_tumor} Tumor Cells")
print(f"Cluster 1: {cluster_1_lymph} Lymphocytes, {cluster_1_tumor} Tumor Cells")

# 基于主要成分确定聚类映射
if cluster_0_tumor > cluster_0_lymph:
    # Cluster 0 主要是 Tumor Cells
    cluster_to_label = {0: 'Tumor Cells', 1: 'Lymphocytes'}
    print("\nMapping: Cluster 0 -> Tumor Cells, Cluster 1 -> Lymphocytes")
else:
    # Cluster 0 主要是 Lymphocytes
    cluster_to_label = {0: 'Lymphocytes', 1: 'Tumor Cells'}
    print("\nMapping: Cluster 0 -> Lymphocytes, Cluster 1 -> Tumor Cells")

# 创建预测标签
df_results['predicted_label'] = df_results['cluster'].map(cluster_to_label)

# 计算准确性指标
accuracy = accuracy_score(df_results['image_label'], df_results['predicted_label'])
print(f"\nClustering Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 分类报告
print("\nClassification Report:")
print(classification_report(df_results['image_label'], df_results['predicted_label']))

# 混淆矩阵
cm = confusion_matrix(df_results['image_label'], df_results['predicted_label'])
print("\nConfusion Matrix:")
print(cm)

# 计算每个类别的精确度
lymph_precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
tumor_precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0

lymph_recall = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
tumor_recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0

print(f"\nDetailed Metrics:")
print(f"Lymphocytes - Precision: {lymph_precision:.4f}, Recall: {lymph_recall:.4f}")
print(f"Tumor Cells - Precision: {tumor_precision:.4f}, Recall: {tumor_recall:.4f}")

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Lymphocytes', 'Tumor Cells'],
            yticklabels=['Lymphocytes', 'Tumor Cells'])
plt.title('Confusion Matrix for Clustering Results')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建聚类质量分析图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 添加总标题
# 'Clustering Quality Analysis: Combined Morphological & Semantic Features\n'
fig.suptitle('Using only Combined Morphological features\n'
             f'Accuracy: {accuracy*100:.1f}% | Total Samples: {len(df_results)}', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. 标签分布
df_results['image_label'].value_counts().plot(kind='bar', ax=axes[0,0], color=['lightblue', 'lightcoral'])
axes[0,0].set_title('Distribution of True Labels')
axes[0,0].set_xlabel('Cell Type')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=0)

# 2. 聚类分布
df_results['cluster'].value_counts().plot(kind='bar', ax=axes[0,1], color=['orange', 'green'])
axes[0,1].set_title('Distribution of Clusters')
axes[0,1].set_xlabel('Cluster')
axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=0)

# 3. 交叉表热力图
crosstab_norm = pd.crosstab(df_results['image_label'], df_results['cluster'], normalize='index')
sns.heatmap(crosstab_norm, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,0])
axes[1,0].set_title('Normalized Cross-tabulation\n(Row percentages)')
axes[1,0].set_xlabel('Cluster')
axes[1,0].set_ylabel('True Label')

# 4. PCA散点图（按真实标签着色）
scatter = axes[1,1].scatter(df_results['pc1'], df_results['pc2'], 
                           c=df_results['image_label'].map({'Lymphocytes': 0, 'Tumor Cells': 1}),
                           cmap='viridis', alpha=0.6, s=20)
axes[1,1].set_title('PCA Visualization (Colored by True Labels)')
axes[1,1].set_xlabel('PC1')
axes[1,1].set_ylabel('PC2')
# 添加图例
import matplotlib.patches as patches
legend_elements = [patches.Patch(color='purple', label='Lymphocytes'),
                  patches.Patch(color='yellow', label='Tumor Cells')]
axes[1,1].legend(handles=legend_elements)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])  # 调整布局为总标题留出空间
plt.savefig('clustering_quality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存分析结果
analysis_summary = {
    'total_samples': len(df_results),
    'accuracy': accuracy,
    'lymphocytes_count': len(df_results[df_results['image_label'] == 'Lymphocytes']),
    'tumor_cells_count': len(df_results[df_results['image_label'] == 'Tumor Cells']),
    'cluster_0_count': len(df_results[df_results['cluster'] == 0]),
    'cluster_1_count': len(df_results[df_results['cluster'] == 1]),
    'lymphocytes_precision': lymph_precision,
    'lymphocytes_recall': lymph_recall,
    'tumor_cells_precision': tumor_precision,
    'tumor_cells_recall': tumor_recall
}

# 保存到文件
import json
with open('clustering_analysis_summary.json', 'w') as f:
    json.dump(analysis_summary, f, indent=2)

print(f"\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"The clustering successfully separated the two cell types with {accuracy*100:.2f}% accuracy!")
print(f"This demonstrates that the combined feature vector (morphological + semantic)")
print(f"effectively captures the differences between Lymphocytes and Tumor Cells.")
print("\nGenerated files:")
print("- confusion_matrix.png: Confusion matrix visualization")
print("- clustering_quality_analysis.png: Comprehensive analysis plots")
print("- clustering_analysis_summary.json: Numerical summary")
print("="*50) 