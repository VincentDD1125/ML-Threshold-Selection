# Visualization Guide

## 🎯 Features

### 📊 Training visualization
Click "📊 Training Visualization" to view detailed training results:

#### 1. ROC curve
- Classification performance
- AUC measures discrimination
- Diagonal represents random classifier

#### 2. Feature importance
- Top 10 features
- Understand model decisions
- Guide feature selection

#### 3. Predicted probability distribution
- Distributions for normal vs artifact
- Evaluate decision boundary
- Reference line at threshold=0.5

#### 4. Metrics
- Accuracy, Precision, Recall, F1
- Bar charts with annotations

### 📈 Prediction visualization
Click "📈 Prediction Visualization" to view prediction analysis:

#### 1. Volume distribution
- Histogram of test data volumes

#### 2. Probability vs volume
- Scatter: probability vs volume
- Color encodes probability

#### 3. Probability distribution
- Predicted probabilities for all particles
- Assess confidence

#### 4. Threshold analysis
- Retain rate and artifact rate vs thresholds
- Choose optimal threshold
- Dual Y-axes

### 💾 Model management
- Save model (.pkl)
- Export results (CSV)

## 🚀 Workflow

### 1. Training phase
1. Load training data
2. Input expert thresholds
3. Train model
4. Click "📊 Training Visualization"

### 2. Prediction phase
1. Load test data
2. Predict analysis
3. Click "📈 Prediction Visualization"
4. Save/export as needed

## 📋 Metrics

### Training
- AUC: closer to 1 is better
- Accuracy: ratio of correct predictions
- Precision: artifact prediction quality
- Recall: fraction of artifacts identified
- F1: harmonic mean of precision and recall

### Prediction
- Retain rate: proportion kept under threshold
- Artifact rate: artifacts among kept
- Threshold: suggested volume threshold

## 💡 Tips

### Improve accuracy
1. Add more training data
2. Optimize features (see importance chart)
3. Tune thresholds via analysis chart
4. Try different model parameters

### Visualization analysis
1. ROC: if AUC<0.7, add data
2. Feature importance: inspect high-importance features
3. Probability distribution: heavy overlap → add features
4. Threshold analysis: balance retain vs artifact rate

## 🔧 故障排除

### 可视化窗口问题
- 如果窗口不显示，检查matplotlib是否正确安装
- 确保有足够的屏幕空间显示1200x800窗口

### 性能问题
- 大数据集可能需要较长时间生成可视化
- 可以关闭不需要的可视化窗口释放内存

## 📊 图表解读

### ROC曲线
- 曲线越靠近左上角越好
- AUC>0.8表示模型性能良好
- AUC<0.6表示模型需要改进

### 特征重要性
- 重要性>0.1的特征值得关注
- 体积相关特征通常最重要
- 方向特征可能指示伪影模式

### 阈值分析
- 保留率下降过快表示阈值过严格
- 伪影率过高表示阈值过宽松
- 寻找保留率>80%且伪影率<5%的阈值

## 🎉 总结

新增的可视化功能让您能够：
- **深入了解模型性能** - 通过多种图表评估训练效果
- **优化预测结果** - 通过阈值分析选择最佳参数
- **提高工作效率** - 直观的图表替代复杂的数值分析
- **支持决策制定** - 基于可视化结果调整策略

**开始使用：运行 `python main.py` 并探索新的可视化功能！** 🚀
