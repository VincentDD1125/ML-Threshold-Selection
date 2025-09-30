#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application controller: minimal GUI class that delegates all logic to modules.
"""

from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Optional
import numpy as np
import pandas as pd

from src.ml_threshold_selection.ui_layout import build_main_ui
from src.ml_threshold_selection.data_io import (
    load_multiple_training_data as io_load_multiple_training_data,
    input_expert_thresholds as io_input_expert_thresholds,
    load_test_data as io_load_test_data,
    input_voxel_sizes as io_input_voxel_sizes,
    validate_training_data as io_validate_training_data,
)
from src.ml_threshold_selection.ui_visualization import (
    show_training_visualization as ui_show_training,
    show_prediction_visualization as ui_show_prediction,
    save_chart as ui_save_chart,
)
from src.ml_threshold_selection.labeling import generate_labels_from_thresholds as gen_labels_from_thresholds
from src.ml_threshold_selection.training_pipeline import train_model_pipeline
from src.ml_threshold_selection.io_persistence import auto_save as persist_auto_save, load_last as persist_load_last
from src.ml_threshold_selection.prediction_analysis import compute_dual_thresholds
from src.ml_threshold_selection.export_results import export_filtered_results, export_threshold_report
from src.ml_threshold_selection.feature_utils import extract_simple_features as util_extract_simple_features
from src.ml_threshold_selection.fabric_logging import UILogger
from src.ml_threshold_selection.fabric_pipeline import run_fabric_boxplots

# Optional project modules
try:
    from ml_threshold_selection.feature_engineering import FeatureEngineer
    from ml_threshold_selection.threshold_finder import AdaptiveThresholdFinder
    from ml_threshold_selection.semi_supervised_learner import SemiSupervisedThresholdLearner
    FULL_MODULES_AVAILABLE = True
except Exception:
    FULL_MODULES_AVAILABLE = False

try:
    import lightgbm as lgb  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

from src.analysis.ellipsoid_feature_analyzer import JoshuaFeatureAnalyzer as EllipsoidFeatureAnalyzer
from src.features.ellipsoid_feature_engineering import JoshuaFeatureEngineerFixed as EllipsoidFeatureEngineer
from src.features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer


class FixedMLGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ML Threshold Selection - Enhanced Version")
        self.root.geometry("1400x900")

        # Expose availability flags for UI
        self.LIGHTGBM_AVAILABLE = LIGHTGBM_AVAILABLE
        self.FULL_MODULES_AVAILABLE = FULL_MODULES_AVAILABLE

        # State
        self.model = None
        self.feature_engineer = FeatureEngineer() if FULL_MODULES_AVAILABLE else None
        self.threshold_finder = AdaptiveThresholdFinder() if FULL_MODULES_AVAILABLE else None
        self.training_data = None
        self.test_data = None
        self.features = None
        self.probabilities = None
        self.training_files = []
        self.expert_thresholds = {}
        self.sample_list = []
        self.threshold_input_window = None
        self.training_results = None
        self.visualization_window = None
        self.ellipsoid_feature_analyzer = EllipsoidFeatureAnalyzer()
        self.ellipsoid_feature_engineer = EllipsoidFeatureEngineer()
        self.resolution_aware_engineer = ResolutionAwareFeatureEngineer()
        self.ellipsoid_analysis_results = None
        self.voxel_sizes = {}
        # Configuration parameters
        try:
            from config import STRICT_PROBABILITY_THRESHOLD
            self.strict_probability_threshold = STRICT_PROBABILITY_THRESHOLD
        except ImportError:
            self.strict_probability_threshold = 0.01  # Default fallback

        # Build UI
        build_main_ui(self)

    # Basic logging to UI
    def log(self, message: str):
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()

    # IO delegates
    def load_multiple_training_data(self):
        io_load_multiple_training_data(self)

    def validate_training_data(self, df):
        return io_validate_training_data(self, df)

    def input_expert_thresholds(self):
        io_input_expert_thresholds(self)

    def load_test_data(self):
        io_load_test_data(self)

    def input_voxel_sizes(self):
        io_input_voxel_sizes(self)

    # Labeling
    def generate_labels_from_thresholds(self):
        if self.training_data is None:
            self.log("❌ Please load training data first")
            return
        self.training_data = gen_labels_from_thresholds(
            training_data=self.training_data,
            expert_thresholds=self.expert_thresholds,
            voxel_sizes=self.voxel_sizes,
            sample_list=self.sample_list,
            log=self.log,
        )

    # Training
    def train_model(self):
        if self.training_data is None:
            self.log("❌ Please load training data first")
            return
        if not self.expert_thresholds:
            self.log("❌ Please enter expert thresholds first")
            return
        try:
            self.log("🔄 Training model...")
            self.root.update()
            self.generate_labels_from_thresholds()
            model, features, training_results = train_model_pipeline(
                training_data=self.training_data,
                voxel_sizes=self.voxel_sizes,
                resolution_aware_engineer=self.resolution_aware_engineer,
                lightgbm_available=LIGHTGBM_AVAILABLE,
            )
            self.model = model
            self.features = features
            self.training_results = training_results
            self.log("✅ Training complete!")
            self.log(f"   - Num features: {len(self.features.columns)}")
            self.log(f"   - Num samples: {len(self.training_results['X'])}")
            self.log(f"   - Train AUC: {self.training_results['train_auc']:.3f}")
            self.log(f"   - Train accuracy: {self.training_results['train_accuracy']:.3f}")
            self.log(f"   - Precision: {self.training_results['precision']:.3f}")
            self.log(f"   - Recall: {self.training_results['recall']:.3f}")
            self.log(f"   - F1 score: {self.training_results['f1']:.3f}")
            # Auto-save
            persist_auto_save(
                model=self.model,
                training_data=self.training_data,
                expert_thresholds=self.expert_thresholds,
                voxel_sizes=self.voxel_sizes,
                training_files=self.training_files,
                features=self.features,
                training_results=self.training_results,
                ellipsoid_analysis_results=self.ellipsoid_analysis_results,
                resolution_aware_engineer=self.resolution_aware_engineer,
            )
            self.log("💾 Model automatically saved for next session")
        except Exception as e:
            self.log(f"❌ Training failed: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

    # Prediction
    def predict_analysis(self):
        if self.test_data is None:
            self.log("❌ Please load test data first")
            return
        if self.model is None:
            self.log("❌ Please train the model first")
            return
        try:
            self.log("🔄 Starting prediction analysis...")
            if not self.voxel_sizes:
                self.log("❌ Please input voxel sizes first (mm)")
                return
            first_sample = self.sample_list[0] if self.sample_list else list(self.voxel_sizes.keys())[0]
            voxel_mm = float(self.voxel_sizes[first_sample])
            test_features = self.resolution_aware_engineer.extract(self.test_data, voxel_size_mm=voxel_mm, fit_scaler=False)
            self.log(f"🔧 Resolution-aware prediction features: {test_features.shape[1]}")
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(test_features.values)[:, 1]
            else:
                probabilities = self.model.predict(test_features.values)
            volumes = self.test_data['Volume3d (mm^3) '].values
            voxel_vol = voxel_mm ** 3
            voxels_cont = volumes / voxel_vol
            inflection_threshold, noise_removal_threshold = compute_dual_thresholds(
                voxels_cont, probabilities, self.strict_probability_threshold
            )
            self.log("✅ Prediction analysis complete!")
            self.log(f"   - Total particles: {len(volumes)}")
            if inflection_threshold is not None:
                self.log(f"   - Loose threshold (Inflection): {int(np.ceil(inflection_threshold))} vox | {(int(np.ceil(inflection_threshold))*voxel_vol):.2e} mm³")
            if noise_removal_threshold is not None:
                self.log(f"   - Strict threshold (P>{self.strict_probability_threshold}): {int(np.ceil(noise_removal_threshold))} vox | {(int(np.ceil(noise_removal_threshold))*voxel_vol):.2e} mm³")
            self.probabilities = probabilities
            self.loose_threshold_vox = int(np.ceil(inflection_threshold)) if inflection_threshold is not None else None
            self.strict_threshold_vox = int(np.ceil(noise_removal_threshold)) if noise_removal_threshold is not None else None
            self.test_voxel_size_mm = voxel_mm
        except Exception as e:
            self.log(f"❌ Prediction analysis failed: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            self.loose_threshold_vox = None
            self.strict_threshold_vox = None
            self.test_voxel_size_mm = None

    # Visualization
    def show_training_visualization(self):
        ui_show_training(self)

    def show_prediction_visualization(self):
        ui_show_prediction(self)

    def save_chart(self, fig, base_name, format_type):
        try:
            ui_save_chart(fig, base_name, format_type, self.log)
        except Exception as e:
            self.log(f"❌ Chart save failed: {e}")

    # Help / User Guide
    def open_user_guide(self):
        try:
            import webbrowser
            from pathlib import Path
            guide = Path(__file__).resolve().parents[2] / 'docs' / 'USER_GUIDE_MODEL_AND_FEATURES_EN.md'
            if guide.exists():
                webbrowser.open(guide.as_uri())
                self.log("📖 Opened User Guide (EN)")
            else:
                self.log("⚠️ User Guide not found under docs/")
        except Exception as e:
            self.log(f"❌ Failed to open User Guide: {e}")

    def configure_strict_threshold(self):
        """Configure the strict probability threshold (P>threshold)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Strict Threshold")
        dialog.geometry("450x250")
        dialog.grab_set()
        dialog.resizable(False, False)

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text="Configure Strict Threshold", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))

        info_label = ttk.Label(main_frame, text="Set the probability threshold for strict filtering (P > threshold)", font=("Arial", 10))
        info_label.pack(pady=(0, 15))

        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.pack(pady=(0, 20))
        
        ttk.Label(threshold_frame, text="Probability Threshold:").pack(side=tk.LEFT, padx=(0, 10))
        threshold_var = tk.StringVar(value=str(self.strict_probability_threshold))
        threshold_entry = ttk.Entry(threshold_frame, textvariable=threshold_var, width=15)
        threshold_entry.pack(side=tk.LEFT, padx=(0, 10))
        threshold_entry.focus()

        # Add some spacing
        ttk.Label(main_frame, text="").pack(pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))
        
        def save_threshold():
            try:
                new_threshold = float(threshold_var.get())
                if 0.0 <= new_threshold <= 1.0:
                    self.strict_probability_threshold = new_threshold
                    self.log(f"✅ Strict probability threshold updated to: {new_threshold}")
                    dialog.destroy()
                else:
                    self.log("❌ Threshold must be between 0.0 and 1.0")
            except ValueError:
                self.log("❌ Invalid threshold value. Please enter a number between 0.0 and 1.0")

        ttk.Button(button_frame, text="Save", command=save_threshold, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=15).pack(side=tk.LEFT, padx=(0, 10))
        
        # Add keyboard shortcuts
        dialog.bind('<Return>', lambda e: save_threshold())
        dialog.bind('<Escape>', lambda e: dialog.destroy())

    # Config loader for thresholds & voxel sizes
    def load_thresholds_config(self):
        try:
            from tkinter import filedialog
            import pandas as pd
            path = filedialog.askopenfilename(title='Select thresholds_voxels.csv', filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
            if not path:
                return
            df = pd.read_csv(path)
            required = {'SampleID', 'ExpertThreshold_mm3', 'VoxelSize_mm'}
            if not required.issubset(set(df.columns)):
                self.log(f"❌ Invalid config. Required columns: {sorted(list(required))}")
                return
            # Update in-memory config
            self.expert_thresholds = {str(r['SampleID']): float(r['ExpertThreshold_mm3']) for _, r in df.iterrows()}
            self.voxel_sizes = {str(r['SampleID']): float(r['VoxelSize_mm']) for _, r in df.iterrows()}
            self.sample_list = sorted(list(self.voxel_sizes.keys()))
            self.log(f"✅ Loaded thresholds & voxels for {len(self.sample_list)} samples")
            for sid in self.sample_list:
                self.log(f"   - {sid}: threshold={self.expert_thresholds.get(sid, 'NA'):.2e} mm³, voxel={self.voxel_sizes.get(sid, 'NA'):.4f} mm")
        except Exception as e:
            self.log(f"❌ Failed to load thresholds config: {e}")

    # Fabric boxplots
    def generate_fabric_boxplots(self):
        try:
            if self.test_data is None:
                self.log("❌ Please load test data first")
                return
            required_cols = [
                'EigenVal1', 'EigenVal2', 'EigenVal3',
                'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',
                'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',
                'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z',
            ]
            for c in required_cols:
                if c not in self.test_data.columns:
                    self.log(f"❌ Missing required column for fabric analysis: {c}")
                    return
            if not hasattr(self, 'loose_threshold_vox') or not hasattr(self, 'strict_threshold_vox'):
                self.log("❌ Please perform prediction analysis first to get thresholds")
                return
            if self.loose_threshold_vox is None or self.strict_threshold_vox is None:
                self.log("❌ Thresholds are not ready. Run prediction analysis again.")
                return
            if not self.voxel_sizes:
                self.log("❌ Please input voxel sizes first (mm)")
                return
            first_sample = self.sample_list[0] if self.sample_list else list(self.voxel_sizes.keys())[0]
            voxel_mm = float(self.voxel_sizes[first_sample])
            logger = UILogger(self.log)
            run_fabric_boxplots(
                df=self.test_data,
                voxel_size_mm=voxel_mm,
                loose_threshold_vox=self.loose_threshold_vox,
                strict_threshold_vox=self.strict_threshold_vox,
                logger=logger,
                outputs_dir='outputs',
                n_bootstrap=1000,
                min_particles=50,
            )
        except Exception as e:
            self.log(f"❌ Fabric boxplots failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")

    # Export
    def export_results(self):
        if self.test_data is None or self.probabilities is None:
            self.log("❌ Please perform prediction analysis first")
            return
        try:
            if not hasattr(self, 'loose_threshold_vox') or not hasattr(self, 'strict_threshold_vox'):
                self.log("❌ Please perform prediction analysis to get thresholds first")
                return
            if self.loose_threshold_vox is None or self.strict_threshold_vox is None:
                self.log("❌ Thresholds not calculated properly. Please run prediction analysis again.")
                return
            if not hasattr(self, 'test_voxel_size_mm') or self.test_voxel_size_mm is None:
                self.log("❌ Test voxel size not available")
                return
            voxel_size_mm = self.test_voxel_size_mm
            loose_file, strict_file = export_filtered_results(
                results_df=self.test_data,
                probabilities=self.probabilities,
                loose_threshold_vox=self.loose_threshold_vox,
                strict_threshold_vox=self.strict_threshold_vox,
                voxel_size_mm=voxel_size_mm,
                outputs_dir='outputs',
                strict_probability_threshold=self.strict_probability_threshold,
            )
            voxel_vol = voxel_size_mm ** 3
            loose_threshold_mm = self.loose_threshold_vox * voxel_vol
            strict_threshold_mm = self.strict_threshold_vox * voxel_vol
            loose_kept = int((self.test_data['Volume3d (mm^3) '] >= loose_threshold_mm).sum())
            strict_kept = int((self.test_data['Volume3d (mm^3) '] >= strict_threshold_mm).sum())
            report_filename = f"outputs/Threshold_Report_{self.loose_threshold_vox:.0f}vox_{self.strict_threshold_vox:.0f}vox.txt"
            export_threshold_report(
                out_path=report_filename,
                total_rows=len(self.test_data),
                voxel_size_mm=voxel_size_mm,
                loose_threshold_vox=self.loose_threshold_vox,
                strict_threshold_vox=self.strict_threshold_vox,
                loose_threshold_mm=loose_threshold_mm,
                strict_threshold_mm=strict_threshold_mm,
                loose_kept=loose_kept,
                strict_kept=strict_kept,
            )
            self.log(f"✅ Loose threshold results exported to: {loose_file}")
            self.log(f"✅ Strict threshold results exported to: {strict_file}")
            self.log(f"✅ Threshold report exported to: {report_filename}")
            self.log("📊 Export complete: 2 XLSX files + 1 TXT report generated")
        except Exception as e:
            self.log(f"❌ Export results failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")

    # Analysis
    def analyze_ellipsoid_features(self):
        from src.ml_threshold_selection.analysis_pipeline import run_feature_analysis
        run_feature_analysis(self)

    def display_ellipsoid_analysis_results(self):
        from src.ml_threshold_selection.analysis_pipeline import display_feature_analysis_results
        display_feature_analysis_results(self)

    def extract_simple_features(self, df):
        return util_extract_simple_features(df)

    # Persistence
    def load_last_time_model(self):
        try:
            model_data = persist_load_last('outputs')
            self.model = model_data['model']
            self.training_data = model_data['training_data']
            self.expert_thresholds = model_data['expert_thresholds']
            self.voxel_sizes = model_data['voxel_sizes']
            self.training_files = model_data['training_files']
            self.features = model_data.get('features', None)
            self.training_results = model_data.get('training_results', None)
            self.ellipsoid_analysis_results = model_data.get('ellipsoid_analysis_results', None) or model_data.get('joshua_analysis_results', None)
            if 'resolution_aware_engineer' in model_data:
                self.resolution_aware_engineer = model_data['resolution_aware_engineer']
                self.log("   - Resolution-aware engineer with fitted scaler restored")
            else:
                self.resolution_aware_engineer = ResolutionAwareFeatureEngineer()
                self.log("   - New resolution-aware engineer created (scaler not fitted)")
            self.log("✅ Last time model loaded successfully!")
            self.log(f"   - Training data: {len(self.training_data)} particles")
            self.log(f"   - Expert thresholds: {len(self.expert_thresholds)} samples")
            self.log(f"   - Voxel sizes: {len(self.voxel_sizes)} samples")
            self.log(f"   - Training files: {len(self.training_files)} files")
            self.log("   - Model ready for prediction and visualization")
        except Exception as e:
            self.log(f"❌ Load last time model failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")


