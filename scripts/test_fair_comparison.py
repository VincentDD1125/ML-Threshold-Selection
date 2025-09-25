#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公平对比测试 - 确保两种方法使用相同的特征空间
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from joshua_feature_engineering import JoshuaFeatureEngineer
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


def test_fair_comparison():
    """公平对比测试 - 两种方法都使用Joshua特征空间"""
    print("🚀 公平对比测试 - 两种方法都使用Joshua特征空间")
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
    
    # 3. 提取Joshua特征（两种方法都使用）
    print("\n🔬 提取Joshua特征（两种方法都使用）...")
    joshua_engineer = JoshuaFeatureEngineer(voxel_size_mm=None)  # 不使用体素尺寸归一化
    
    joshua_features_train = joshua_engineer.extract_joshua_features(X_train)
    joshua_features_test = joshua_engineer.extract_joshua_features(X_test)
    
    print(f"   - Joshua特征数: {len(joshua_features_train.columns)}")
    print(f"   - 特征列表: {list(joshua_features_train.columns)}")
    
    # 4. 方法1：使用所有7个Joshua特征
    print("\n🔬 方法1：使用所有7个Joshua特征...")
    learner_all = SupervisedThresholdLearner()
    results_all = learner_all.train(joshua_features_train, y_train)
    probabilities_all = learner_all.predict_proba(joshua_features_test)
    predictions_all = (probabilities_all > 0.5).astype(int)
    
    # 5. 方法2：使用部分Joshua特征（模拟传统方法的选择）
    print("\n📊 方法2：使用部分Joshua特征（模拟特征选择）...")
    
    # 选择最重要的特征（基于方差）
    feature_vars = joshua_features_train.var()
    selected_features = feature_vars.nlargest(4).index.tolist()  # 选择4个最重要的特征
    
    print(f"   - 选择的特征: {selected_features}")
    
    joshua_features_train_selected = joshua_features_train[selected_features]
    joshua_features_test_selected = joshua_features_test[selected_features]
    
    learner_selected = SupervisedThresholdLearner()
    results_selected = learner_selected.train(joshua_features_train_selected, y_train)
    probabilities_selected = learner_selected.predict_proba(joshua_features_test_selected)
    predictions_selected = (probabilities_selected > 0.5).astype(int)
    
    # 6. 性能对比
    print("\n📈 公平对比结果:")
    print("=" * 40)
    
    # 方法1性能
    auc_all = roc_auc_score(y_test, probabilities_all)
    report_all = classification_report(y_test, predictions_all, output_dict=True)
    
    print("🔬 方法1（所有7个Joshua特征）:")
    print(f"   - AUC: {auc_all:.4f}")
    print(f"   - 准确率: {report_all['accuracy']:.4f}")
    
    # 检查是否有类别1
    if '1' in report_all:
        print(f"   - 精确率: {report_all['1']['precision']:.4f}")
        print(f"   - 召回率: {report_all['1']['recall']:.4f}")
        print(f"   - F1分数: {report_all['1']['f1-score']:.4f}")
    else:
        print("   - 无伪影预测")
    
    # 方法2性能
    auc_selected = roc_auc_score(y_test, probabilities_selected)
    report_selected = classification_report(y_test, predictions_selected, output_dict=True)
    
    print("\n📊 方法2（选择的4个Joshua特征）:")
    print(f"   - AUC: {auc_selected:.4f}")
    print(f"   - 准确率: {report_selected['accuracy']:.4f}")
    
    # 检查是否有类别1
    if '1' in report_selected:
        print(f"   - 精确率: {report_selected['1']['precision']:.4f}")
        print(f"   - 召回率: {report_selected['1']['recall']:.4f}")
        print(f"   - F1分数: {report_selected['1']['f1-score']:.4f}")
    else:
        print("   - 无伪影预测")
    
    # 7. 可视化对比
    print("\n📊 生成公平对比可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fair Comparison: All vs Selected Joshua Features', fontsize=16, fontweight='bold')
    
    # ROC曲线对比
    from sklearn.metrics import roc_curve
    fpr_all, tpr_all, _ = roc_curve(y_test, probabilities_all)
    fpr_selected, tpr_selected, _ = roc_curve(y_test, probabilities_selected)
    
    axes[0, 0].plot(fpr_all, tpr_all, label=f'All 7 Features (AUC={auc_all:.3f})', linewidth=2)
    axes[0, 0].plot(fpr_selected, tpr_selected, label=f'Selected 4 Features (AUC={auc_selected:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves Comparison (Fair)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 混淆矩阵对比
    cm_all = confusion_matrix(y_test, predictions_all)
    cm_selected = confusion_matrix(y_test, predictions_selected)
    
    import seaborn as sns
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('All 7 Features Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    sns.heatmap(cm_selected, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0])
    axes[1, 0].set_title('Selected 4 Features Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # 性能指标对比
    metrics = ['AUC', 'Accuracy']
    scores_all = [auc_all, report_all['accuracy']]
    scores_selected = [auc_selected, report_selected['accuracy']]
    
    # 添加其他指标（如果存在）
    if '1' in report_all and '1' in report_selected:
        metrics.extend(['Precision', 'Recall', 'F1-Score'])
        scores_all.extend([report_all['1']['precision'], report_all['1']['recall'], report_all['1']['f1-score']])
        scores_selected.extend([report_selected['1']['precision'], report_selected['1']['recall'], report_selected['1']['f1-score']])
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, scores_all, width, label='All 7 Features', alpha=0.8)
    axes[1, 1].bar(x + width/2, scores_selected, width, label='Selected 4 Features', alpha=0.8)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics Comparison (Fair)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fair_comparison_joshua_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. 总结
    print("\n🎯 公平对比总结:")
    print("=" * 40)
    
    if auc_all > auc_selected:
        print("🏆 使用所有7个Joshua特征表现更好!")
        improvement = (auc_all - auc_selected) / auc_selected * 100
        print(f"   改进幅度: {improvement:.2f}%")
    else:
        print("📊 使用选择的4个Joshua特征表现更好")
        improvement = (auc_selected - auc_all) / auc_all * 100
        print(f"   优势幅度: {improvement:.2f}%")
    
    print(f"\n🔬 关键发现:")
    print(f"   - 所有7个特征: AUC={auc_all:.4f}")
    print(f"   - 选择4个特征: AUC={auc_selected:.4f}")
    print(f"   - 特征选择的影响: {abs(auc_all - auc_selected):.4f}")
    
    return {
        'auc_all': auc_all,
        'auc_selected': auc_selected,
        'features_all': 7,
        'features_selected': 4,
        'report_all': report_all,
        'report_selected': report_selected
    }


def main():
    """主函数"""
    print("🚀 公平对比测试套件")
    print("=" * 60)
    
    try:
        # 公平对比测试
        results = test_fair_comparison()
        
        print("\n✅ 公平对比测试完成!")
        print("=" * 60)
        
        # 最终总结
        print("🎯 最终结论:")
        if results['auc_all'] > results['auc_selected']:
            print("🏆 使用所有Joshua特征在性能上更优!")
        else:
            print("📊 特征选择在Joshua特征空间中也有价值")
        
        print(f"🔬 所有7个特征: AUC={results['auc_all']:.4f}")
        print(f"📊 选择4个特征: AUC={results['auc_selected']:.4f}")
        print(f"📈 性能差异: {abs(results['auc_all'] - results['auc_selected']):.4f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
