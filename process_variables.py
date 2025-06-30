import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("Loading image level quantitative features...")

# 读取Excel文件
df_variables = pd.read_excel('image_level_quantitative_features_cleaned.xlsx')

print(f"Original DataFrame shape: {df_variables.shape}")
print(f"Columns: {list(df_variables.columns)}")

# 检查数据
print("\nDataFrame info:")
print(df_variables.info())

# 分离需要保留的列和需要合并的列
id_columns = ['image_source', 'image_label']
feature_columns = [col for col in df_variables.columns if col not in id_columns]

print(f"\nID columns to keep: {id_columns}")
print(f"Feature columns to combine: {len(feature_columns)} columns")
print(f"Feature columns: {feature_columns}")

# 检查是否存在缺失值
print(f"\nMissing values in feature columns:")
missing_values = df_variables[feature_columns].isnull().sum()
print(missing_values[missing_values > 0])

# 处理缺失值（如果有的话）
if df_variables[feature_columns].isnull().sum().sum() > 0:
    print("Filling missing values with column means...")
    df_variables[feature_columns] = df_variables[feature_columns].fillna(df_variables[feature_columns].mean())

# 提取特征数据
feature_data = df_variables[feature_columns].values
print(f"Feature matrix shape: {feature_data.shape}")

# 标准化特征数据
scaler = StandardScaler()
feature_data_scaled = scaler.fit_transform(feature_data)
print(f"Scaled feature matrix shape: {feature_data_scaled.shape}")

# 创建新的DataFrame
df_combined = pd.DataFrame()
df_combined['image_source'] = df_variables['image_source']
df_combined['image_label'] = df_variables['image_label']

# 将每行的特征向量作为一个整体存储
df_combined['feature_vector'] = [row for row in feature_data_scaled]

print(f"\nNew DataFrame shape: {df_combined.shape}")
print(f"Feature vector dimension: {df_combined['feature_vector'].iloc[0].shape}")

# 显示前几行
print("\nFirst few rows:")
print(df_combined[['image_source', 'image_label']].head())

# 保存结果
df_combined.to_pickle("variables_combined_vectors.pkl")
print("\nCombined DataFrame saved to variables_combined_vectors.pkl")

# 也保存标准化器，以便后续使用
import pickle
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Feature scaler saved to feature_scaler.pkl")

# 验证加载
print("\n" + "="*50)
print("Verifying saved data...")

loaded_df = pd.read_pickle("variables_combined_vectors.pkl")
print(f"Loaded DataFrame shape: {loaded_df.shape}")
print(f"Loaded feature vector shape (first row): {loaded_df['feature_vector'].iloc[0].shape}")

with open('feature_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)
print("Feature scaler loaded successfully")

print("="*50)
print("Process completed successfully!") 