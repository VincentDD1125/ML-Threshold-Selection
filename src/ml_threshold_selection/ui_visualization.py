#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter visualization helpers extracted from main.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def save_chart(fig, base_name, format_type, log_func):
    from datetime import datetime
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.{format_type}"
    filepath = os.path.join(output_dir, filename)
    if format_type == "png":
        fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')
    elif format_type == "svg":
        fig.savefig(filepath, bbox_inches='tight', format='svg')
    log_func(f"✅ Chart saved as {format_type.upper()}: {filepath}")


def show_training_visualization(app):
    if app.training_results is None:
        app.log("❌ Please train model first")
        return
    if app.visualization_window is not None:
        app.visualization_window.destroy()
    import tkinter as tk
    app.visualization_window = tk.Toplevel(app.root)
    app.visualization_window.title("Training Results Visualization")
    app.visualization_window.geometry("1200x800")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Model Training Results Analysis', fontsize=16, fontweight='bold')

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(app.training_results['y'], app.training_results['train_proba'])
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {app.training_results['train_auc']:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1)
    axes[0, 0].set_xlabel('False Positive Rate (FPR)')
    axes[0, 0].set_ylabel('True Positive Rate (TPR)')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    feature_names = app.training_results['features'].columns
    if hasattr(app.model, 'feature_importances_'):
        importance = app.model.feature_importances_
    elif hasattr(app.model, 'feature_importance'):
        importance = app.model.feature_importance(importance_type='gain')
    else:
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(app.model, app.training_results['X'], app.training_results['y'], n_repeats=5, random_state=42)
        importance = perm_importance.importances_mean
    top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]
    if top_features:
        names, values = zip(*top_features)
        axes[0, 1].barh(range(len(names)), values)
        axes[0, 1].set_yticks(range(len(names)))
        axes[0, 1].set_yticklabels(names)
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Feature Importance (Top 10)')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No feature importance available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Feature Importance')

    normal_proba = app.training_results['train_proba'][app.training_results['y'] == 0]
    artifact_proba = app.training_results['train_proba'][app.training_results['y'] == 1]
    axes[1, 0].hist(normal_proba, bins=30, alpha=0.7, label='Normal Particles', color='blue')
    axes[1, 0].hist(artifact_proba, bins=30, alpha=0.7, label='Artifact Particles', color='red')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    axes[1, 0].set_xlabel('Prediction Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [app.training_results['train_accuracy'], app.training_results['precision'], app.training_results['recall'], app.training_results['f1']]
    bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, app.visualization_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    import tkinter as tk
    save_frame = tk.Frame(app.visualization_window)
    save_frame.pack(pady=10)
    tk.Button(save_frame, text="Save as PNG", command=lambda: save_chart(fig, "training_visualization", "png", app.log)).pack(side=tk.LEFT, padx=5)
    tk.Button(save_frame, text="Save as SVG", command=lambda: save_chart(fig, "training_visualization", "svg", app.log)).pack(side=tk.LEFT, padx=5)
    app.log("✅ Training visualization displayed")


def show_prediction_visualization(app):
    if app.test_data is None or app.probabilities is None:
        app.log("❌ Please perform prediction analysis first")
        return
    if app.visualization_window is not None:
        app.visualization_window.destroy()
    import tkinter as tk
    app.visualization_window = tk.Toplevel(app.root)
    app.visualization_window.title("Prediction Results Visualization")
    app.visualization_window.geometry("1200x800")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Prediction Results Analysis', fontsize=16, fontweight='bold')

    volumes = app.test_data['Volume3d (mm^3) '].values
    if not app.voxel_sizes:
        app.log("❌ Please input voxel sizes first (mm)")
        return
    first_sample = app.sample_list[0] if app.sample_list else list(app.voxel_sizes.keys())[0]
    voxel_mm = float(app.voxel_sizes[first_sample])
    voxel_vol = voxel_mm ** 3
    voxels_cont = np.clip(volumes / voxel_vol, a_min=1e-12, a_max=None)

    axes[0, 0].hist(np.log10(voxels_cont), bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('log10(Voxel Count)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Voxel Count Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    xticks = axes[0, 0].get_xticks()
    mm3_labels = [f"{(10**t)*voxel_vol:.1e}" for t in xticks]
    ax_top = axes[0, 0].secondary_xaxis('top')
    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels(mm3_labels, rotation=0)
    ax_top.set_xlabel('Equivalent Volume (mm³)')

    scatter = axes[0, 1].scatter(np.log10(voxels_cont), app.probabilities, c=app.probabilities, cmap='RdYlBu_r', alpha=0.6)
    axes[0, 1].set_xlabel('log10(Voxel Count)')
    axes[0, 1].set_ylabel('Artifact Probability')
    axes[0, 1].set_title('Prediction Probability vs Voxel Count')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1], label='Artifact Probability')

    axes[1, 0].hist(app.probabilities, bins=30, alpha=0.7, color='lightgreen')
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
    axes[1, 0].set_xlabel('Prediction Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    thresholds = np.logspace(np.log10(max(voxels_cont.min(), 1e-12)), np.log10(voxels_cont.max()), 50)
    retention_rates, artifact_rates = [], []
    for t in thresholds:
        retained = voxels_cont >= t
        retention_rates.append(float(np.mean(retained)))
        artifact_rates.append(float(np.mean(app.probabilities[retained])) if np.sum(retained) > 0 else 0.0)
    ax2 = axes[1, 1].twinx()
    line1 = axes[1, 1].plot(np.log10(thresholds), retention_rates, 'b-', label='Retention Rate')
    line2 = ax2.plot(np.log10(thresholds), artifact_rates, 'r-', label='Artifact Rate')
    
    # Add threshold lines if available
    if hasattr(app, 'loose_threshold_vox') and app.loose_threshold_vox is not None:
        loose_log = np.log10(app.loose_threshold_vox)
        axes[1, 1].axvline(x=loose_log, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Loose Threshold')
        ax2.axvline(x=loose_log, color='green', linestyle='--', linewidth=2, alpha=0.8)
    
    if hasattr(app, 'strict_threshold_vox') and app.strict_threshold_vox is not None:
        strict_log = np.log10(app.strict_threshold_vox)
        strict_label = f'Strict Threshold (P>{app.strict_probability_threshold})'
        axes[1, 1].axvline(x=strict_log, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=strict_label)
        ax2.axvline(x=strict_log, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    
    axes[1, 1].set_xlabel('log10(Voxel Threshold)')
    axes[1, 1].set_ylabel('Retention Rate', color='b')
    ax2.set_ylabel('Artifact Rate', color='r')
    axes[1, 1].set_title('Dual Threshold Analysis (Voxel Domain)')
    axes[1, 1].grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1, 1].legend(lines, labels, loc='center right')

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, app.visualization_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    import tkinter as tk
    save_frame = tk.Frame(app.visualization_window)
    save_frame.pack(pady=10)
    tk.Button(save_frame, text="Save as PNG", command=lambda: save_chart(fig, "prediction_visualization", "png", app.log)).pack(side=tk.LEFT, padx=5)
    tk.Button(save_frame, text="Save as SVG", command=lambda: save_chart(fig, "prediction_visualization", "svg", app.log)).pack(side=tk.LEFT, padx=5)
    app.log("✅ Prediction visualization displayed")


