#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正后的Joshua方法 - 解决体素尺寸和特征缩放问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from joshua_feature_engineering_fixed import JoshuaFeatureEngineerFixed
from src.ml_threshold_selection.supervised_learner import SupervisedThresholdLearner


def create_synthetic_data(n_particles=2000, noise_level=0.1):
    """创建合成数据用于测试"""
    print("🔬 创建合成测试数据...")
    
    np.random.seed(42)
    
    # 创建真实颗粒（正常）
    n_normal = int(n_particles * 0.8)
    normal_data = {
        'Volume3d (mm^3) ': np.random.lognormal(-12, 0.8, n_normal),
        'EigenVal1': np.random.lognormal(-6, 0.3, n_normal),
        'EigenVal2': np.random.lognormal(-6, 0.3, n_normal),
        'EigenVal3': np.random.lognormal(-6, 0.3, n_normal),
    }
    
    # 创建伪影颗粒（异常）
    n_artifact = n_particles - n_normal
    artifact_data = {
        'Volume3d (mm^3) ': np.random.lognormal(-13, 1.2, n_artifact),  # 更小，更分散
        'EigenVal1': np.random.lognormal(-5.5, 0.6, n_artifact),  # 更大，更分散
        'EigenVal2': np.random.lognormal(-6.5, 0.4, n_artifact),
        'EigenVal3': np.random.lognormal(-6.5, 0.4, n_artifact),
    }
    
    # 合并数据
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = np.concatenate([normal_data[key], artifact_data[key]])
    
    # 创建特征向量（单位向量）
    for i in range(1, 4):
        # 正常颗粒：随机方向
        normal_vec = np.random.normal(0, 1, (n_normal, 3))
        normal_vec = normal_vec / np.linalg.norm(normal_vec, axis=1, keepdims=True)
        
        # 伪影颗粒：倾向于与体素网格对齐
        artifact_vec = np.random.normal(0, 0.3, (n_artifact, 3))
        # 增加对齐概率
        alignment_mask = np.random.random(n_artifact) < 0.3
        artifact_vec[alignment_mask] = np.eye(3)[np.random.choice(3, alignment_mask.sum())]
        artifact_vec = artifact_vec / np.linalg.norm(artifact_vec, axis=1, keepdims=True)
        
        all_vec = np.concatenate([normal_vec, artifact_vec])
        all_data[f'EigenVec{i}X'] = all_vec[:, 0]
        all_data[f'EigenVec{i}Y'] = all_vec[:, 1]
        all_data[f'EigenVec{i}Z'] = all_vec[:, 2]
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 创建标签
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_artifact)])
    
    # 添加噪声
    if noise_level > 0:
        for col in df.columns:
            if col.startswith('EigenVec'):
                noise = np.random.normal(0, noise_level, len(df))
                df[col] += noise
                # 重新归一化
                for i in range(1, 4):
                    vec = df[[f'EigenVec{i}X', f'EigenVec{i}Y', f'EigenVec{i}Z']].values
                    vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)
                    vec_normalized = vec / vec_norm
                    df[f'EigenVec{i}X'] = vec_normalized[:, 0]
                    df[f'EigenVec{i}Y'] = vec_normalized[:, 1]
                    df[f'EigenVec{i}Z'] = vec_normalized[:, 2]
    
    print(f"   - 总颗粒数: {len(df)}")
    print(f"   - 正常颗粒: {n_normal} ({n_normal/len(df)*100:.1f}%)")
    print(f"   - 伪影颗粒: {n_artifact} ({n_artifact/len(df)*100:.1f}%)")
    
    return df, labels


def test_fixed_joshua():
    """测试修正后的Joshua方法"""
    print("🚀 测试修正后的Joshua方法")
    print("=" * 60)
    
    # 1. 创建测试数据
    df, labels = create_synthetic_data(n_particles=2000, noise_level=0.1)
    
    print(f"📊 原始数据统计:")
    print(f"   - 体积范围: {df['Volume3d (mm^3) '].min():.2e} - {df['Volume3d (mm^3) '].max():.2e} mm³")
    print(f"   - 特征值范围: EigenVal1={df['EigenVal1'].min():.2e}-{df['EigenVal1'].max():.2e}")
    
    # 2. 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"📊 数据分割:")
    print(f"   - 训练集: {len(X_train)} 样本")
    print(f"   - 测试集: {len(X_test)} 样本")
    
    # 3. 测试修正后的Joshua方法
    print("\n🔬 测试修正后的Joshua方法...")
    joshua_engineer = JoshuaFeatureEngineerFixed(voxel_size_um=50)  # 50微米分辨率
    
    # 提取Joshua特征（训练时拟合标准化器）
    joshua_features_train = joshua_engineer.extract_joshua_features(X_train, fit_scaler=True)
    
    # 提取Joshua特征（测试时使用已拟合的标准化器）
    joshua_features_test = joshua_engineer.extract_joshua_features(X_test, fit_scaler=False)
    
    print(f"   - Joshua特征数: {len(joshua_features_train.columns)}")
    print(f"   - 特征列表: {list(joshua_features_train.columns)}")
    
    # 训练Joshua模型
    joshua_learner = SupervisedThresholdLearner()
    joshua_results = joshua_learner.train(joshua_features_train, y_train)
    joshua_probabilities = joshua_learner.predict_proba(joshua_features_test)
    joshua_predictions = (joshua_probabilities > 0.5).astype(int)
    
    # 4. 性能评估
    print("\n📈 修正后的Joshua方法性能:")
    print("=" * 40)
    
    # Joshua方法性能
    joshua_auc = roc_auc_score(y_test, joshua_probabilities)
    joshua_report = classification_report(y_test, joshua_predictions, output_dict=True)
    
    print("🔬 修正后的Joshua方法:")
    print(f"   - AUC: {joshua_auc:.4f}")
    print(f"   - 准确率: {joshua_report['accuracy']:.4f}")
    
    # 检查是否有类别1
    if '1' in joshua_report:
        print(f"   - 精确率: {joshua_report['1']['precision']:.4f}")
        print(f"   - 召回率: {joshua_report['1']['recall']:.4f}")
        print(f"   - F1分数: {joshua_report['1']['f1-score']:.4f}")
    else:
        print("   - 无伪影预测")
    
    # 5. 可视化结果
    print("\n📊 生成修正后的可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fixed Joshua Method Results', fontsize=16, fontweight='bold')
    
    # ROC曲线
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, joshua_probabilities)
    
    axes[0, 0].plot(fpr, tpr, label=f'Fixed Joshua (AUC={joshua_auc:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve (Fixed Joshua Method)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, joshua_predictions)
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix (Fixed Joshua Method)')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 特征重要性（如果有的话）
    if hasattr(joshua_learner, 'model') and hasattr(joshua_learner.model, 'feature_importances_'):
        feature_importance = joshua_learner.model.feature_importances_
        feature_names = joshua_features_train.columns
        
        bars = axes[1, 0].bar(range(len(feature_names)), feature_importance)
        axes[1, 0].set_xticks(range(len(feature_names)))
        axes[1, 0].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Feature Importance')
        axes[1, 0].set_title('Feature Importance (Fixed Joshua Method)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 显示数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        axes[1, 0].text(0.5, 0.5, 'Feature importance not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Feature Importance')
    
    # 特征分布对比
    normal_mask = y_test == 0
    artifact_mask = y_test == 1
    
    for i, col in enumerate(joshua_features_test.columns):
        if i < 3:  # 只显示前3个特征
            normal_values = joshua_features_test[col][normal_mask]
            artifact_values = joshua_features_test[col][artifact_mask]
            
            axes[1, 1].hist(normal_values, alpha=0.5, label=f'{col} (Normal)', bins=20)
            axes[1, 1].hist(artifact_values, alpha=0.5, label=f'{col} (Artifact)', bins=20)
    
    axes[1, 1].set_xlabel('Feature Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Feature Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fixed_joshua_method_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 总结
    print("\n🎯 修正后的测试总结:")
    print("=" * 40)
    
    print(f"🔬 修正后的Joshua方法性能:")
    print(f"   - AUC: {joshua_auc:.4f}")
    print(f"   - 准确率: {joshua_report['accuracy']:.4f}")
    
    if joshua_auc > 0.9:
        print("🏆 修正后的Joshua方法表现优秀!")
    elif joshua_auc > 0.8:
        print("✅ 修正后的Joshua方法表现良好!")
    else:
        print("⚠️ 修正后的Joshua方法仍有改进空间")
    
    print(f"\n🔧 关键修正:")
    print(f"   - 体素尺寸归一化: 50μm分辨率")
    print(f"   - 特征标准化: 所有特征均值≈0，标准差≈1")
    print(f"   - 特征数量: {len(joshua_features_train.columns)}个")
    
    return {
        'auc': joshua_auc,
        'accuracy': joshua_report['accuracy'],
        'features': len(joshua_features_train.columns),
        'report': joshua_report
    }


def main():
    """主函数"""
    print("🚀 修正后的Joshua方法测试套件")
    print("=" * 60)
    
    try:
        # 测试修正后的Joshua方法
        results = test_fixed_joshua()
        
        print("\n✅ 修正后的测试完成!")
        print("=" * 60)
        
        # 最终总结
        print("🎯 最终结论:")
        print(f"🔬 修正后的Joshua方法: AUC={results['auc']:.4f}")
        print(f"📊 准确率: {results['accuracy']:.4f}")
        print(f"🔧 特征数: {results['features']}")
        
        if results['auc'] > 0.9:
            print("🏆 修正成功！Joshua方法现在表现优秀!")
        elif results['auc'] > 0.8:
            print("✅ 修正有效！Joshua方法现在表现良好!")
        else:
            print("⚠️ 仍有改进空间，但修正方向正确")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
