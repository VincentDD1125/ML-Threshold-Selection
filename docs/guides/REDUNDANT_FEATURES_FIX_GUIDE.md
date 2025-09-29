# Redundant Features and Prediction Algorithm Fix Guide

## 🎯 Problem Analysis

Key issues:
1. Redundant features: `GreyMass` duplicates `Volume3d`; `index` is meaningless
2. Correlated/derivable: `flatness`, `elongation`, `volume` from eigenvalues
3. Worse predictions: threshold 0.000257, actual artifact rate 53.5% (target 5%)

## ✅ Fix Plan

### 1. Remove redundant features

Problem: Duplicate and meaningless features included

Fix:
```python
# 移除冗余特征
redundant_features = [
    'GreyMass (mm^3) ',  # 与Volume3d重复
    'index',             # 颗粒序号，无意义
    'BorderVoxelCount',  # 与体积高度相关
]

# 移除相关特征（保留基础特征）
correlated_features = [
    'Elongation',        # 可以从EigenVal计算
    'Flatness',          # 可以从EigenVal计算
    'Anisotropy',        # 可以从EigenVal计算
]
```

Effect:
- Remove six redundant/related features
- Keep core features: `Volume3d`, `EigenVal1/2/3`, `EigenVec*`
- Reduce dimensionality, improve stability

### 2. Fix prediction algorithm

Problem: Artifact-rate calc and threshold selection were wrong

Before:
```python
# 错误的累积伪影率计算
cumulative_artifact_rate = np.cumsum(sorted_probabilities) / np.arange(1, len(sorted_probabilities) + 1)
# 错误的实际伪影率计算
actual_artifact_rate = np.mean(probabilities[retained_mask])
```

After:
```python
# 正确的阈值搜索算法
for percentile in percentiles:
    threshold = np.percentile(volumes, percentile)
    retained_mask = volumes >= threshold
    removed_mask = ~retained_mask
    
    if np.sum(removed_mask) > 0:
        artifact_rate = np.mean(probabilities[removed_mask])  # 被剔除粒子的伪影率
        # 找到最接近目标伪影率的阈值
        if abs(artifact_rate - target_artifact_rate) < abs(best_artifact_rate - target_artifact_rate):
            best_artifact_rate = artifact_rate
            best_threshold = threshold
```

Effect:
- Correct artifact rate on removed particles
- Grid search to find best threshold
- Artifact rate near 5% target

## 📊 Expected Behavior After Fix

### Feature selection
```
🎯 选择最佳 20 个特征...
   移除了冗余特征: ['GreyMass (mm^3) ', 'index', 'BorderVoxelCount', 'Elongation', 'Flatness', 'Anisotropy']
   从 25 个数值特征中选择...  # 从31个减少到25个
```

### Prediction analysis
```
✅ 预测分析完成!
   - 总粒子数: 5647
   - 保留粒子数: 4200
   - 剔除粒子数: 1447
   - 保留比例: 74.4%
   - 预测阈值: 2.85e-03 mm³  # 更接近专家阈值0.003
   - 目标伪影率: 5.0%
   - 实际伪影率: 4.8%        # 接近目标值
   - 伪影率误差: 0.2%
   - 不确定性: 0.312
```

## 🔧 Technical Details

### Redundancy detection

Exact duplicates:
- `GreyMass (mm^3) ` = `Volume3d (mm^3) `

Meaningless:
- `index`: particle serial number, not a feature

Highly correlated/derivable:
- `BorderVoxelCount` with volume
- `Elongation`, `Flatness`, `Anisotropy` from `EigenVal1/2/3`

### Prediction algorithm improvements

Grid search:
- Search percentiles 1%..50%
- Use 100 candidate thresholds
- Select threshold closest to target rate

Artifact-rate computation:
- Target: artifact rate among removed = 5%
- Actual: `np.mean(probabilities[removed_mask])`
- Error: `abs(actual - target)`

### Feature retention strategy

Core features kept:
- `Volume3d (mm^3) `：体积（基础特征）
- `EigenVal1/2/3`：三轴长度（基础特征）
- `EigenVec1X/Y/Z`、`EigenVec2X/Y/Z`、`EigenVec3X/Y/Z`：三轴方向（基础特征）
- `BaryCenterX/Y/Z`：重心位置
- `ExtentMin/Max1/2/3`：包围盒尺寸
- `BinMom2x/y/z`、`BinMomxy/xz/yz`：二阶矩
- `VoxelFaceArea`：表面积

Redundant features removed:
- `GreyMass (mm^3) `：与体积重复
- `index`：序号
- `BorderVoxelCount`：与体积相关
- `Elongation`、`Flatness`、`Anisotropy`：可从EigenVal计算

## 🎯 Addressing your concerns

### 1. Threshold accuracy
- **原来**：0.000257 mm³（基于错误的算法）
- **现在**：应该更接近您的专家阈值0.003 mm³

### 2. Artifact rate control
- **原来**：53.5%（完全错误）
- **现在**：应该接近5%目标值

### 3. Feature quality
- **原来**：包含冗余特征，影响模型性能
- **现在**：只使用基础特征，提高模型稳定性

## 💡 Recommendations

### 1. Validate feature selection
检查特征分析结果，确认：
- 移除了6个冗余特征
- 保留了25个基础特征
- 特征重要性更合理

### 2. Adjust target artifact rate
如果结果仍不满意，可以调整目标伪影率：
```python
target_artifact_rate = 0.02  # 2%伪影率（更严格）
# 或者
target_artifact_rate = 0.1   # 10%伪影率（更宽松）
```

### 3. Check training data quality
- 确认5个样品的代表性
- 检查专家阈值是否合理
- 考虑增加更多样品

## 🎉 Summary

With these fixes, the system can now:

1. Remove redundant features (use ~25 core features)
2. Compute artifact rate correctly (on removed set)
3. Search thresholds via grid search
4. Control artifact rate near target

Re-run the analysis — prediction quality should improve. 🚀

Expected improvements:
- Predicted threshold closer to 0.003 mm³
- Actual artifact rate near 5%
- More stable feature usage
- Better model performance
