import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
import pdb

# 读取数据
print("Loading features DataFrame...")
df = pd.read_pickle('vectors_dataframe.pkl')
df_label = pd.read_pickle('variables_combined_vectors.pkl')

# 去掉df['filename']的.pt后缀，创建新列用于合并
df['merge_key'] = df['filename'].str.replace('.pt', '', regex=False)
df_label['merge_key'] = df_label['image_source']

# 以merge_key为键合并
df_merged = pd.merge(df, df_label, on='merge_key', suffixes=('_vec', '_label'))

print(f"合并后DataFrame形状: {df_merged.shape}")
print(df_merged[['merge_key', 'image_label']].head())

X = np.vstack(df_merged['vector'].values)
X = X[:,:2]
y = df_merged['image_label'].values

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_names = le.classes_
print(f"Label mapping: {dict(zip(label_names, range(len(label_names))) )}")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 训练Random Forest
print("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = clf.score(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=label_names))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=300)
plt.show()

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('rf_roc_curve.png', dpi=300)
plt.show()

# 特征重要性
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title('Feature Importances (Top 20)')
plt.bar(range(20), importances[indices[:20]], align='center')
plt.xticks(range(20), [f'F{i}' for i in indices[:20]], rotation=45)
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300)
plt.show()

# t-SNE降维可视化
print("Performing t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
for i, label in enumerate(label_names):
    plt.scatter(X_embedded[y_encoded==i, 0], X_embedded[y_encoded==i, 1], label=label, alpha=0.6, s=20)
plt.title('t-SNE Visualization of Feature Vectors')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.tight_layout()
plt.savefig('rf_tsne.png', dpi=300)
plt.show()

print("\nAll results saved: rf_confusion_matrix.png, rf_roc_curve.png, rf_feature_importance.png, rf_tsne.png") 