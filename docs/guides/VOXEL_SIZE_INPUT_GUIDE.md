# Voxel Size Input Guide

## 🎯 Voxel Size Input

We added a dedicated voxel size input module to use your real values instead of estimates.

## 🚀 Workflow

### Step 1: Load training data
1. Click "1. Load Training Data"
2. Select your XLSX/CSV files
3. The system automatically extracts sample names

### Step 2: Input voxel sizes
1. Click "2.5. Input Voxel Sizes"
2. Enter voxel size for each sample in the dialog
3. Double-click the "Voxel Size (mm)" cell to edit
4. Click "Save"

### Step 3: Input expert thresholds
1. Click "2. Input Expert Thresholds"
2. Enter thresholds for each sample

### Step 4: Feature analysis
1. Click "🔍 Feature Analysis"
2. The system normalizes using your voxel sizes

## 📊 体素尺寸输入界面

### 界面特点
- **表格形式**：清晰显示每个样品ID
- **双击编辑**：双击体素尺寸列进行编辑
- **实时验证**：输入时自动验证格式和数值
- **保存确认**：保存后显示所有输入的体素尺寸

### 输入格式
- **单位**：毫米 (mm)
- **格式**：数字，如 `0.0025` 或 `2.5e-3`
- **精度**：支持小数点后4位精度

### 示例输入
```
Sample ID          Voxel Size (mm)
totalLE19          0.0025
totalAKAN20        0.0018
totalLE03          0.0023
totalHL19335       0.0021
totalANA16937      0.0019
```

## 🔧 体素尺寸标准化

### 标准化规则
系统会根据体素尺寸对几何特征进行标准化：

#### 体积特征
- `Volume3d (mm^3) ` → 除以 `voxel_size³`
- `GreyMass (mm^3) ` → 除以 `voxel_size³`

#### 面积特征
- `BinMom2x (mm^2) ` → 除以 `voxel_size²`
- `BinMom2y (mm^2) ` → 除以 `voxel_size²`
- `BinMom2z (mm^2) ` → 除以 `voxel_size²`
- `BinMomxy (mm^2) ` → 除以 `voxel_size²`
- `BinMomxz (mm^2) ` → 除以 `voxel_size²`
- `BinMomyz (mm^2) ` → 除以 `voxel_size²`

#### 长度特征
- `BaryCenterX (mm) ` → 除以 `voxel_size`
- `BaryCenterY (mm) ` → 除以 `voxel_size`
- `BaryCenterZ (mm) ` → 除以 `voxel_size`
- `ExtentMin1 (mm) ` → 除以 `voxel_size`
- `ExtentMin2 (mm) ` → 除以 `voxel_size`
- `ExtentMin3 (mm) ` → 除以 `voxel_size`
- `ExtentMax1 (mm) ` → 除以 `voxel_size`
- `ExtentMax2 (mm) ` → 除以 `voxel_size`
- `ExtentMax3 (mm) ` → 除以 `voxel_size`

## 💡 使用建议

### 1. 获取体素尺寸
- 从CT扫描参数中获取
- 从重建软件中获取
- 从数据文件中获取
- 从扫描报告中获取

### 2. 输入注意事项
- 确保单位是毫米 (mm)
- 保持足够的精度（建议4位小数）
- 验证数值的合理性
- 确保所有样品都有体素尺寸

### 3. 验证输入
- 保存后检查日志中的确认信息
- 确保所有样品都显示正确的体素尺寸
- 如有错误，可以重新编辑

## 🎯 优势

### 1. 准确性
- 使用真实的体素尺寸数据
- 避免估算误差
- 确保标准化准确性

### 2. 灵活性
- 支持不同样品的不同体素尺寸
- 适应各种扫描参数
- 处理复杂的实验设置

### 3. 可追溯性
- 记录所有输入的体素尺寸
- 便于验证和审计
- 支持重复实验

## 🔍 特征分析改进

使用真实体素尺寸后，特征分析会更加准确：

### 1. 标准化效果
- 消除体素尺寸差异的影响
- 确保特征的可比性
- 提高模型泛化能力

### 2. 特征选择
- 基于标准化后的特征进行选择
- 识别真正重要的特征
- 避免体素尺寸偏差

### 3. 模型性能
- 提高预测准确率
- 增强模型稳定性
- 改善跨样品一致性

## 📋 完整工作流程

1. **Load Training Data** → 加载数据文件
2. **Input Voxel Sizes** → 输入体素尺寸
3. **Input Expert Thresholds** → 输入专家阈值
4. **Feature Analysis** → 进行特征分析
5. **Train Model** → 训练模型
6. **Load Test Data** → 加载测试数据
7. **Predict Analysis** → 预测分析

## 🎉 总结

通过使用真实的体素尺寸数据，您可以：

- **提高准确性**：使用真实数据而非估算
- **增强可靠性**：确保标准化的正确性
- **改善性能**：提高模型预测准确率
- **保持一致性**：确保跨样品的可比性

**开始使用：运行 `python main.py` 并按照新的工作流程操作！** 🚀
