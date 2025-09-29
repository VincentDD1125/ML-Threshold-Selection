# Advanced Feature Analysis Guide

## 🎯 Deep Feature Analysis System

We provide a comprehensive feature analysis system designed to address the following key areas:

### 1. Significance of feature differences
- Use statistical tests (t-test) to determine which features differ significantly between removed and kept particles
- Compute effect size (Cohen's d) to assess practical significance
- Identify truly important features and avoid non-informative ones

### 2. Voxel size normalization
- Automatically estimate voxel size for each sample
- Normalize geometric features to remove voxel-size bias
- Ensure comparability across samples

### 3. Redundant feature handling
- Detect highly correlated features (e.g., volume and grey_mass)
- Use correlation analysis to identify multicollinearity
- Provide feature selection recommendations

## 🚀 Workflow

### Step 1: Data preparation
1. Load training data files
2. Input expert thresholds
3. Click "🔍 Feature Analysis"

### Step 2: Feature analysis
The system will automatically perform the following:

#### A. Voxel size estimation
```
📏 Sample totalLE19 estimated voxel size: 0.0023 mm
📏 Sample totalAKAN20 estimated voxel size: 0.0018 mm
```

#### B. Feature difference analysis
```
🎯 Significant features (p<0.05, Cohen's d>0.2):
   - Volume3d (mm^3) : d=2.456, p=1.23e-45
   - Elongation: d=1.234, p=2.34e-12
   - Flatness: d=1.567, p=1.45e-15
   - EigenVec1X: d=0.789, p=3.21e-08
```

#### C. Feature selection results
```
🔝 F-test selected: 15
🔝 Mutual information selected: 18
🔝 Combined: 22
```

### Step 3: Visualization
The system generates four key charts:

#### 1. Effect size distribution
- Cohen's d values for the top 15 features
- Thresholds for small/medium/large effects
- Helps identify the most important features

#### 2. Significant features
- Shows all statistically significant features
- Sorted by effect size
- Provides feature importance ranking

#### 3. Correlation heatmap
- Correlations among the top 10 features
- Identify highly correlated pairs
- Helps avoid multicollinearity

#### 4. Feature selection comparison
- Compare F-test and mutual information methods
- Show overlaps and differences
- Guide the final selection

### Step 4: Model training
- The best features from analysis are used automatically
- Significantly improves performance and accuracy

## 📊 Metrics

### Statistical significance
- **p-value < 0.05**: significant
- **Cohen's d > 0.2**: small effect
- **Cohen's d > 0.5**: medium effect  
- **Cohen's d > 0.8**: large effect

### Feature selection methods
- **F-test**: ANOVA-based, suitable for linear relationships
- **Mutual information**: information-theoretic, suitable for non-linear relationships
- **Combined**: union of both to avoid missing important features

### Voxel normalization
- **Volume features**: divide by voxel volume (voxel_size³)
- **Area features**: divide by voxel area (voxel_size²)
- **Length features**: divide by voxel size (voxel_size)

## 🎯 Key advantages

### 1. Scientific rigor
- Statistics-driven feature selection
- Avoid subjective judgment and using all features blindly
- Ensure selected features are truly effective

### 2. Voxel-size invariance
- Automatically handle voxel-size differences across samples
- Ensure comparability and consistency
- Improve model generalization

### 3. Multicollinearity handling
- Automatically detect highly correlated features
- Reduce overfitting
- Improve model stability

### 4. Visualization
- Intuitive charts for analysis results
- Easier understanding and interpretation
- Support evidence-based decisions

## 💡 Best practices

### 1. Analyze before training
- Perform feature analysis before training
- Understand key data patterns
- Optimize feature selection based on analysis

### 2. Focus on effect size
- Consider effect size beyond statistical significance
- Prefer large-effect features
- Avoid small-effect yet significant features

### 3. Voxel-size consistency
- Ensure the same voxel size across samples
- Or use normalized features
- Avoid voxel-size bias in results

### 4. Correlation checks
- Check correlations regularly
- Remove highly redundant features
- Keep features independent

## 🔬 Technical details

### Voxel size estimation algorithm
```python
# Estimate from the cube root of the minimal positive volume
min_volume = np.min(volumes[volumes > 0])
estimated_voxel_size = min_volume ** (1/3)
```

### Effect size computation
```python
# Cohen's d
pooled_std = np.sqrt(((n1-1)*std1² + (n2-1)*std2²) / (n1+n2-2))
cohens_d = abs(mean1 - mean2) / pooled_std
```

### Feature selection algorithms
```python
# F-test selection
f_selector = SelectKBest(score_func=f_classif, k=20)
# Mutual information selection  
mi_selector = SelectKBest(score_func=mutual_info_classif, k=20)
```

## 🎉 Expected outcomes

With this deep feature analysis system, you can expect:

1. Higher predictive accuracy through principled feature selection
2. Enhanced stability by avoiding multicollinearity
3. Better interpretability of which features matter and why
4. Cross-sample consistency via voxel normalization

Get started: run `python main.py` and click "🔍 Feature Analysis". 🚀
