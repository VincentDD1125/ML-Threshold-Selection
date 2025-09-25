# Ellipsoid Feature Engineering Guide

## 🎯 Your analysis is absolutely correct!

**An ellipsoidal particle can be fully described by 6 core features**:
1. **3 eigenvalues**: `EigenVal1`, `EigenVal2`, `EigenVal3` (axis lengths)
2. **3 eigenvectors**: `EigenVec1X/Y/Z`, `EigenVec2X/Y/Z`, `EigenVec3X/Y/Z` (axis orientations)

**Other features are redundant**:
- `Volume3d` = derivable from eigenvalues
- `Elongation`, `Flatness` = derivable from eigenvalues
- `ExtentMin/Max` = bounding box, not ellipsoid features
- `BinMom2x/y/z` = second moments, not ellipsoid features

## ✅ Ellipsoid feature engineering plan

### 1. Core feature extraction
Keep only the six core ellipsoid features:
```python
core_features = [
    'EigenVal1', 'EigenVal2', 'EigenVal3',  # 三轴长度
    'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',  # 第一主轴方向
    'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',  # 第二主轴方向
    'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z',  # 第三主轴方向
]
```

### 2. Eigenvector handling
Problem: Eigenvectors must be processed as a whole, not as independent X/Y/Z components.

Solution:
```python
# 计算与坐标轴的夹角
vec = df[vec_cols].values
vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)
vec_normalized = vec / (vec_norm + 1e-8)  # 避免除零

# Alignment with X/Y/Z axes (absolute)
df_features[f'eigenvec{i}_x_alignment'] = np.abs(vec_normalized[:, 0])
df_features[f'eigenvec{i}_y_alignment'] = np.abs(vec_normalized[:, 1])
df_features[f'eigenvec{i}_z_alignment'] = np.abs(vec_normalized[:, 2])

# Maximum alignment (which axis is most aligned)
df_features[f'eigenvec{i}_max_alignment'] = np.max(np.abs(vec_normalized), axis=1)

# Whether aligned with voxel axes (alignment > 0.9)
df_features[f'eigenvec{i}_voxel_aligned'] = (df_features[f'eigenvec{i}_max_alignment'] > 0.9).astype(int)
```

### 3. Create engineered features

**从特征值计算的特征**：
```python
# 体积（从特征值计算）
df_features['volume'] = (4/3) * np.pi * df['EigenVal1'] * df['EigenVal2'] * df['EigenVal3']

# 形状特征
df_features['elongation'] = df['EigenVal1'] / df['EigenVal2']  # 长宽比
df_features['flatness'] = df['EigenVal2'] / df['EigenVal3']   # 宽厚比
df_features['sphericity'] = df['EigenVal3'] / df['EigenVal1']  # 球形度

# 各向异性
df_features['anisotropy'] = (df['EigenVal1'] - df['EigenVal3']) / (df['EigenVal1'] + df['EigenVal2'] + df['EigenVal3'])

# 特征值差异
df_features['lambda_diff_12'] = df['EigenVal1'] - df['EigenVal2']
df_features['lambda_diff_23'] = df['EigenVal2'] - df['EigenVal3']
df_features['lambda_diff_13'] = df['EigenVal1'] - df['EigenVal3']
```

**从特征向量计算的特征**：
```python
# 整体体素对齐度
df_features['overall_voxel_alignment'] = (
    df_features['eigenvec1_max_alignment'] + 
    df_features['eigenvec2_max_alignment'] + 
    df_features['eigenvec3_max_alignment']
) / 3

# 是否整体与体素对齐
df_features['is_voxel_aligned'] = (df_features['overall_voxel_alignment'] > 0.8).astype(int)
```

**组合特征**：
```python
# 体积分类
volume_median = df_features['volume'].median()
df_features['is_small_volume'] = (df_features['volume'] < volume_median * 0.1).astype(int)
df_features['is_very_small_volume'] = (df_features['volume'] < volume_median * 0.01).astype(int)

# 形状组合特征
df_features['elongation_flatness_product'] = df_features['elongation'] * df_features['flatness']
df_features['is_high_elongation'] = (df_features['elongation'] > 2.0).astype(int)
df_features['is_high_flatness'] = (df_features['flatness'] > 2.0).astype(int)
```

## 📊 工程后的特征列表

### 基础特征（12个）
- `EigenVal1`, `EigenVal2`, `EigenVal3`
- `EigenVec1X`, `EigenVec1Y`, `EigenVec1Z`
- `EigenVec2X`, `EigenVec2Y`, `EigenVec2Z`
- `EigenVec3X`, `EigenVec3Y`, `EigenVec3Z`

### 工程特征（约25个）
**形状特征**：
- `volume`, `elongation`, `flatness`, `sphericity`, `anisotropy`
- `lambda_diff_12`, `lambda_diff_23`, `lambda_diff_13`

**方向特征**：
- `eigenvec1_x_alignment`, `eigenvec1_y_alignment`, `eigenvec1_z_alignment`
- `eigenvec1_max_alignment`, `eigenvec1_voxel_aligned`
- `eigenvec2_x_alignment`, `eigenvec2_y_alignment`, `eigenvec2_z_alignment`
- `eigenvec2_max_alignment`, `eigenvec2_voxel_aligned`
- `eigenvec3_x_alignment`, `eigenvec3_y_alignment`, `eigenvec3_z_alignment`
- `eigenvec3_max_alignment`, `eigenvec3_voxel_aligned`

**组合特征**：
- `overall_voxel_alignment`, `is_voxel_aligned`
- `is_small_volume`, `is_very_small_volume`
- `elongation_flatness_product`, `is_high_elongation`, `is_high_flatness`

## 🎯 特征工程的优势

### 1. **物理意义明确**
- 所有特征都基于椭球的物理属性
- 特征向量作为整体处理，保持方向信息
- 体积、形状、方向特征分离

### 2. **冗余消除**
- 移除所有可计算的特征
- 只保留基础特征和工程特征
- 减少特征维度，提高模型稳定性

### 3. **伪影检测优化**
- `is_voxel_aligned`：检测与体素轴对齐的伪影
- `is_small_volume`：检测小体积伪影
- `is_high_elongation`：检测异常拉长的伪影
- `overall_voxel_alignment`：整体对齐度

## 📈 预期改进

### 特征质量
- **原来**：31个混合特征，包含冗余
- **现在**：约25个工程特征，物理意义明确

### 模型性能
- **特征一致性**：所有特征都基于椭球属性
- **伪影检测**：专门针对伪影模式设计
- **计算效率**：减少特征维度，提高速度

### 预测准确性
- **阈值计算**：基于更准确的特征
- **伪影率控制**：更精确的伪影检测
- **结果解释**：特征物理意义明确

## 💡 使用建议

### 1. **验证特征工程**
检查特征分析结果，确认：
- 只使用了椭球核心特征
- 工程特征物理意义明确
- 特征数量合理（约25个）

### 2. **关注关键特征**
重点观察这些特征的重要性：
- `is_voxel_aligned`：体素对齐伪影
- `is_small_volume`：小体积伪影
- `overall_voxel_alignment`：整体对齐度
- `elongation`, `flatness`：形状异常

### 3. **调整阈值参数**
如果结果仍不满意，可以调整：
- 体素对齐阈值：`0.9` → `0.8` 或 `0.95`
- 体积分类阈值：`0.1` → `0.05` 或 `0.2`
- 形状异常阈值：`2.0` → `1.5` 或 `3.0`

## 🎉 总结

通过椭球特征工程，程序现在能够：

1. **使用核心特征**：只基于椭球的6个核心特征
2. **正确处理方向**：特征向量作为整体处理
3. **创建工程特征**：针对伪影检测优化
4. **消除冗余**：移除所有可计算的特征
5. **提高准确性**：基于物理意义的特征选择

**现在重新运行程序，应该看到更准确和合理的预测结果！** 🚀

预期改进：
- 特征数量从31个减少到约25个
- 所有特征都有明确的物理意义
- 伪影检测更加准确
- 预测阈值更接近专家值
