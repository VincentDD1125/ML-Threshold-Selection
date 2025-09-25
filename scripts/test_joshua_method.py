#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Joshua方法的性能和准确性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from joshua_feature_engineering import JoshuaFeatureEngineer
from joshua_feature_analyzer import JoshuaFeatureAnalyzer
from feature_analysis_tool import FeatureAnalyzer
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


def test_joshua_vs_traditional():
    """对比Joshua方法和传统方法"""
    print("🚀 开始Joshua方法 vs 传统方法对比测试")
    print("=" * 60)
    
    # 1. 创建测试数据
    df, labels = create_synthetic_data(n_particles=2000, noise_level=0.1)
    
    # 2. 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"📊 数据分割:")
    print(f"   - 训练集: {len(X_train)} 样本")
    print(f"   - 测试集: {len(X_test)} 样本")
    
    # 3. 测试Joshua方法
    print("\n🔬 测试Joshua方法...")
    joshua_engineer = JoshuaFeatureEngineer()
    joshua_analyzer = JoshuaFeatureAnalyzer()
    
    # 提取Joshua特征
    joshua_features_train = joshua_engineer.extract_joshua_features(X_train)
    joshua_features_test = joshua_engineer.extract_joshua_features(X_test)
    
    print(f"   - Joshua特征数: {len(joshua_features_train.columns)}")
    print(f"   - 特征列表: {list(joshua_features_train.columns)}")
    
    # 训练Joshua模型
    joshua_learner = SupervisedThresholdLearner()
    joshua_results = joshua_learner.train(joshua_features_train, y_train)
    joshua_probabilities = joshua_learner.predict_proba(joshua_features_test)
    joshua_predictions = (joshua_probabilities > 0.5).astype(int)
    
    # 4. 测试传统方法
    print("\n📊 测试传统方法...")
    traditional_analyzer = FeatureAnalyzer()
    
    # 执行传统特征分析
    traditional_results = traditional_analyzer.analyze_feature_differences(
        X_train, y_train, voxel_sizes={'sample1': 0.03}
    )
    
    # 使用最佳特征
    best_features = traditional_results['selected_features']['combined']
    available_features = [f for f in best_features if f in X_train.columns]
    
    if available_features:
        traditional_features_train = X_train[available_features]
        traditional_features_test = X_test[available_features]
    else:
        # 使用所有数值特征
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['SampleID', 'label']]
        traditional_features_train = X_train[feature_columns]
        traditional_features_test = X_test[feature_columns]
    
    print(f"   - 传统特征数: {len(traditional_features_train.columns)}")
    print(f"   - 特征列表: {list(traditional_features_train.columns)}")
    
    # 训练传统模型
    traditional_learner = SupervisedThresholdLearner()
    traditional_results = traditional_learner.train(traditional_features_train, y_train)
    traditional_probabilities = traditional_learner.predict_proba(traditional_features_test)
    traditional_predictions = (traditional_probabilities > 0.5).astype(int)
    
    # 5. 性能对比
    print("\n📈 性能对比结果:")
    print("=" * 40)
    
    # Joshua方法性能
    joshua_auc = roc_auc_score(y_test, joshua_probabilities)
    joshua_report = classification_report(y_test, joshua_predictions, output_dict=True)
    
    print("🔬 Joshua方法:")
    print(f"   - AUC: {joshua_auc:.4f}")
    print(f"   - 准确率: {joshua_report['accuracy']:.4f}")
    
    # 检查是否有类别1
    if '1' in joshua_report:
        print(f"   - 精确率: {joshua_report['1']['precision']:.4f}")
        print(f"   - 召回率: {joshua_report['1']['recall']:.4f}")
        print(f"   - F1分数: {joshua_report['1']['f1-score']:.4f}")
    else:
        print("   - 无伪影预测")
    
    # 传统方法性能
    traditional_auc = roc_auc_score(y_test, traditional_probabilities)
    traditional_report = classification_report(y_test, traditional_predictions, output_dict=True)
    
    print("\n📊 传统方法:")
    print(f"   - AUC: {traditional_auc:.4f}")
    print(f"   - 准确率: {traditional_report['accuracy']:.4f}")
    
    # 检查是否有类别1
    if '1' in traditional_report:
        print(f"   - 精确率: {traditional_report['1']['precision']:.4f}")
        print(f"   - 召回率: {traditional_report['1']['recall']:.4f}")
        print(f"   - F1分数: {traditional_report['1']['f1-score']:.4f}")
    else:
        print("   - 无伪影预测")
    
    # 6. 可视化对比
    print("\n📊 生成对比可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Joshua Method vs Traditional Method Comparison', fontsize=16, fontweight='bold')
    
    # ROC曲线对比
    from sklearn.metrics import roc_curve
    fpr_joshua, tpr_joshua, _ = roc_curve(y_test, joshua_probabilities)
    fpr_traditional, tpr_traditional, _ = roc_curve(y_test, traditional_probabilities)
    
    axes[0, 0].plot(fpr_joshua, tpr_joshua, label=f'Joshua (AUC={joshua_auc:.3f})', linewidth=2)
    axes[0, 0].plot(fpr_traditional, tpr_traditional, label=f'Traditional (AUC={traditional_auc:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 混淆矩阵对比
    cm_joshua = confusion_matrix(y_test, joshua_predictions)
    cm_traditional = confusion_matrix(y_test, traditional_predictions)
    
    import seaborn as sns
    sns.heatmap(cm_joshua, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Joshua Method Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    sns.heatmap(cm_traditional, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0])
    axes[1, 0].set_title('Traditional Method Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # 性能指标对比
    metrics = ['AUC', 'Accuracy']
    joshua_scores = [joshua_auc, joshua_report['accuracy']]
    traditional_scores = [traditional_auc, traditional_report['accuracy']]
    
    # 添加其他指标（如果存在）
    if '1' in joshua_report and '1' in traditional_report:
        metrics.extend(['Precision', 'Recall', 'F1-Score'])
        joshua_scores.extend([joshua_report['1']['precision'], joshua_report['1']['recall'], joshua_report['1']['f1-score']])
        traditional_scores.extend([traditional_report['1']['precision'], traditional_report['1']['recall'], traditional_report['1']['f1-score']])
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, joshua_scores, width, label='Joshua', alpha=0.8)
    axes[1, 1].bar(x + width/2, traditional_scores, width, label='Traditional', alpha=0.8)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('joshua_vs_traditional_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 总结
    print("\n🎯 测试总结:")
    print("=" * 40)
    
    if joshua_auc > traditional_auc:
        print("🏆 Joshua方法在AUC上表现更好!")
        improvement = (joshua_auc - traditional_auc) / traditional_auc * 100
        print(f"   改进幅度: {improvement:.2f}%")
    else:
        print("📊 传统方法在AUC上表现更好")
        improvement = (traditional_auc - joshua_auc) / joshua_auc * 100
        print(f"   优势幅度: {improvement:.2f}%")
    
    print(f"\n🔬 Joshua方法优势:")
    print(f"   - 特征数量: {len(joshua_features_train.columns)} (vs {len(traditional_features_train.columns)})")
    print(f"   - 数学严谨性: 基于椭球体几何理论")
    print(f"   - 特征紧凑性: 无冗余特征")
    print(f"   - 几何不变性: 保持Frobenius范数")
    
    print(f"\n📊 传统方法优势:")
    print(f"   - 特征丰富性: 更多工程特征")
    print(f"   - 领域知识: 包含专家经验")
    print(f"   - 灵活性: 可调整特征选择")
    
    return {
        'joshua_auc': joshua_auc,
        'traditional_auc': traditional_auc,
        'joshua_features': len(joshua_features_train.columns),
        'traditional_features': len(traditional_features_train.columns),
        'joshua_report': joshua_report,
        'traditional_report': traditional_report
    }


def test_feature_interpretability():
    """测试特征可解释性"""
    print("\n🔍 测试特征可解释性...")
    
    # 创建测试数据
    df, labels = create_synthetic_data(n_particles=1000, noise_level=0.05)
    
    # 添加SampleID列
    df['SampleID'] = 'sample1'
    
    # Joshua特征分析
    joshua_analyzer = JoshuaFeatureAnalyzer()
    joshua_results = joshua_analyzer.analyze_feature_differences(
        df, labels, voxel_sizes={'sample1': 0.03}
    )
    
    # 生成可视化
    joshua_analyzer.visualize_joshua_feature_analysis(joshua_results)
    
    # 生成报告
    report = joshua_analyzer.generate_joshua_feature_report(joshua_results)
    
    print("📝 Joshua特征分析报告:")
    print(report)
    
    return joshua_results


def main():
    """主函数"""
    print("🚀 Joshua方法测试套件")
    print("=" * 60)
    
    try:
        # 1. 性能对比测试
        results = test_joshua_vs_traditional()
        
        # 2. 特征可解释性测试
        interpretability_results = test_feature_interpretability()
        
        print("\n✅ 所有测试完成!")
        print("=" * 60)
        
        # 最终总结
        print("🎯 最终结论:")
        if results['joshua_auc'] > results['traditional_auc']:
            print("🏆 Joshua方法在性能上优于传统方法!")
        else:
            print("📊 传统方法在性能上优于Joshua方法")
        
        print(f"🔬 Joshua方法特征数: {results['joshua_features']}")
        print(f"📊 传统方法特征数: {results['traditional_features']}")
        print(f"📈 特征减少: {(results['traditional_features'] - results['joshua_features']) / results['traditional_features'] * 100:.1f}%")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
