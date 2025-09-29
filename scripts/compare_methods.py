#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare three approaches on synthetic data:
1) Fixed Joshua (voxel rounding used only for thresholds/visualization, not as feature)
2) Resolution-aware features (log1p(VoxelCount_cont) + Joshua6)
3) Traditional engineered features

Outputs ROC/AUC and confusion matrices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

from joshua_feature_engineering_fixed import JoshuaFeatureEngineerFixed
from res_aware_feature_engineering import ResolutionAwareFeatureEngineer
from feature_analysis_tool import FeatureAnalyzer
from src.ml_threshold_selection.supervised_learner import SupervisedThresholdLearner


def create_synthetic_data(n_particles=2000, noise_level=0.1):
    np.random.seed(42)
    n_normal = int(n_particles * 0.8)
    n_artifact = n_particles - n_normal

    normal_data = {
        'Volume3d (mm^3) ': np.random.lognormal(-12, 0.8, n_normal),
        'EigenVal1': np.random.lognormal(-6, 0.3, n_normal),
        'EigenVal2': np.random.lognormal(-6, 0.3, n_normal),
        'EigenVal3': np.random.lognormal(-6, 0.3, n_normal),
    }
    artifact_data = {
        'Volume3d (mm^3) ': np.random.lognormal(-13, 1.2, n_artifact),
        'EigenVal1': np.random.lognormal(-5.5, 0.6, n_artifact),
        'EigenVal2': np.random.lognormal(-6.5, 0.4, n_artifact),
        'EigenVal3': np.random.lognormal(-6.5, 0.4, n_artifact),
    }

    all_data = {}
    for k in normal_data:
        all_data[k] = np.concatenate([normal_data[k], artifact_data[k]])

    for i in range(1, 4):
        normal_vec = np.random.normal(0, 1, (n_normal, 3))
        normal_vec /= np.linalg.norm(normal_vec, axis=1, keepdims=True)
        artifact_vec = np.random.normal(0, 0.3, (n_artifact, 3))
        alignment_mask = np.random.random(n_artifact) < 0.3
        artifact_vec[alignment_mask] = np.eye(3)[np.random.choice(3, alignment_mask.sum())]
        artifact_vec /= np.linalg.norm(artifact_vec, axis=1, keepdims=True)
        vec = np.concatenate([normal_vec, artifact_vec])
        all_data[f'EigenVec{i}X'] = vec[:, 0]
        all_data[f'EigenVec{i}Y'] = vec[:, 1]
        all_data[f'EigenVec{i}Z'] = vec[:, 2]

    df = pd.DataFrame(all_data)

    if noise_level > 0:
        for i in range(1, 4):
            vec = df[[f'EigenVec{i}X', f'EigenVec{i}Y', f'EigenVec{i}Z']].values
            vec += np.random.normal(0, noise_level, vec.shape)
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            df[f'EigenVec{i}X'], df[f'EigenVec{i}Y'], df[f'EigenVec{i}Z'] = vec[:,0], vec[:,1], vec[:,2]

    labels = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_artifact, dtype=int)])
    return df, labels


def main():
    print("🚀 Compare methods on synthetic data")
    df, labels = create_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, stratify=labels, random_state=42)

    voxel_um = 50
    # Joshua-Fixed
    jf = JoshuaFeatureEngineerFixed(voxel_size_um=voxel_um)
    Xtr_j = jf.extract_joshua_features(X_train, fit_scaler=True)
    Xte_j = jf.extract_joshua_features(X_test, fit_scaler=False)
    learner_j = SupervisedThresholdLearner(); learner_j.train(Xtr_j, y_train)
    prob_j = learner_j.predict_proba(Xte_j); auc_j = roc_auc_score(y_test, prob_j)
    cm_j = confusion_matrix(y_test, (prob_j>0.5).astype(int))

    # Resolution-aware
    ra = ResolutionAwareFeatureEngineer()
    Xtr_ra = ra.extract(X_train, voxel_size_um=voxel_um, fit_scaler=True)
    Xte_ra = ra.extract(X_test, voxel_size_um=voxel_um, fit_scaler=False)
    learner_ra = SupervisedThresholdLearner(); learner_ra.train(Xtr_ra, y_train)
    prob_ra = learner_ra.predict_proba(Xte_ra); auc_ra = roc_auc_score(y_test, prob_ra)
    cm_ra = confusion_matrix(y_test, (prob_ra>0.5).astype(int))

    # Traditional
    analyzer = FeatureAnalyzer()
    trad = analyzer.analyze_feature_differences(X_train, y_train, voxel_sizes={'sample1': 0.03})
    best = trad['selected_features']['combined']
    use_cols = [c for c in best if c in X_train.columns]
    if not use_cols:
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        use_cols = [c for c in num_cols if c not in ['SampleID','label']]
    Xtr_t, Xte_t = X_train[use_cols], X_test[use_cols]
    learner_t = SupervisedThresholdLearner(); learner_t.train(Xtr_t, y_train)
    prob_t = learner_t.predict_proba(Xte_t); auc_t = roc_auc_score(y_test, prob_t)
    cm_t = confusion_matrix(y_test, (prob_t>0.5).astype(int))

    print(f"Joshua-Fixed AUC={auc_j:.4f}")
    print(f"Res-Aware    AUC={auc_ra:.4f}")
    print(f"Traditional  AUC={auc_t:.4f}")

    fig, axes = plt.subplots(2,3, figsize=(20,12))
    fpr_j, tpr_j, _ = roc_curve(y_test, prob_j)
    fpr_ra, tpr_ra, _ = roc_curve(y_test, prob_ra)
    fpr_t, tpr_t, _ = roc_curve(y_test, prob_t)
    axes[0,0].plot(fpr_j,tpr_j,label=f'Joshua-Fixed(AUC={auc_j:.3f})'); axes[0,0].legend(); axes[0,0].set_title('ROC - Joshua')
    axes[0,1].plot(fpr_ra,tpr_ra,label=f'Res-Aware(AUC={auc_ra:.3f})'); axes[0,1].legend(); axes[0,1].set_title('ROC - Res-Aware')
    axes[0,2].plot(fpr_t,tpr_t,label=f'Traditional(AUC={auc_t:.3f})'); axes[0,2].legend(); axes[0,2].set_title('ROC - Traditional')
    import seaborn as sns
    sns.heatmap(cm_j, annot=True, fmt='d', cmap='Blues', ax=axes[1,0]); axes[1,0].set_title('CM - Joshua')
    sns.heatmap(cm_ra, annot=True, fmt='d', cmap='Greens', ax=axes[1,1]); axes[1,1].set_title('CM - Res-Aware')
    sns.heatmap(cm_t, annot=True, fmt='d', cmap='Oranges', ax=axes[1,2]); axes[1,2].set_title('CM - Traditional')
    plt.tight_layout(); plt.savefig('compare_methods.png', dpi=300, bbox_inches='tight'); plt.show()


if __name__ == '__main__':
    main()
