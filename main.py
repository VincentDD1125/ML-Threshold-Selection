#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced GUI - Fixed Version
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from src.features.joshua_feature_engineering import JoshuaFeatureEngineer
from src.features.joshua_feature_engineering_fixed import JoshuaFeatureEngineerFixed
from src.features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer
from src.analysis.joshua_feature_analyzer import JoshuaFeatureAnalyzer

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# ML models and metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

# Try to import LightGBM; fall back to RandomForest if unavailable
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available, using full features")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM unavailable, using RandomForest fallback")

# Import project modules
try:
    from ml_threshold_selection.feature_engineering import FeatureEngineer
    from ml_threshold_selection.threshold_finder import AdaptiveThresholdFinder
    from ml_threshold_selection.semi_supervised_learner import SemiSupervisedThresholdLearner
    FULL_MODULES_AVAILABLE = True
    print("✅ Full modules available")
except ImportError as e:
    FULL_MODULES_AVAILABLE = False
    print(f"⚠️ Full modules unavailable: {e}")
    print("Using built-in simplified version")

class FixedMLGUI:
    """Fixed ML GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ML Threshold Selection - Enhanced Version")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.model = None
        if FULL_MODULES_AVAILABLE:
            self.feature_engineer = FeatureEngineer()
            self.threshold_finder = AdaptiveThresholdFinder()
            print("✅ Using project full modules")
        else:
            # Use built-in simplified version
            self.feature_engineer = None
            self.threshold_finder = None
            print("⚠️ Using built-in simplified modules")
        
        # Data storage
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
        self.joshua_feature_analyzer = JoshuaFeatureAnalyzer()
        self.joshua_feature_engineer = JoshuaFeatureEngineerFixed()
        self.resolution_aware_engineer = ResolutionAwareFeatureEngineer()
        self.joshua_analysis_results = None
        self.voxel_sizes = {}  # Store voxel sizes
        self.use_joshua_method = True  # 默认使用Joshua方法
        
        self.setup_gui()
    
    def setup_gui(self):
        """Set up GUI"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, 
                              text="ML Threshold Selection System - Enhanced", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Function status display
        status_text = "Status: "
        if LIGHTGBM_AVAILABLE:
            status_text += "LightGBM ✅ | "
        else:
            status_text += "LightGBM ❌ | "
        
        if FULL_MODULES_AVAILABLE:
            status_text += "Full Modules ✅"
        else:
            status_text += "Full Modules ❌"
        
        status_label = ttk.Label(main_frame, text=status_text, font=("Arial", 10))
        status_label.pack(pady=5)
        
        # Buttons row 1
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Data loading buttons
        ttk.Button(button_frame, text="1. Load Training Data", 
                  command=self.load_multiple_training_data, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="2. Input Expert Thresholds", 
                  command=self.input_expert_thresholds, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="3. Input Voxel Sizes", 
                  command=self.input_voxel_sizes, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="4. Train Model", 
                  command=self.train_model, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="5. Load Test Data", 
                  command=self.load_test_data, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="6. Predict Analysis", 
                  command=self.predict_analysis, width=20).pack(side=tk.LEFT, padx=3)
        
        # Buttons row 2
        button_frame2 = ttk.Frame(main_frame)
        button_frame2.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame2, text="🔬 Joshua Analysis", 
                  command=self.analyze_joshua_features, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame2, text="📊 Training Visualization", 
                  command=self.show_training_visualization, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame2, text="📈 Prediction Visualization", 
                  command=self.show_prediction_visualization, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame2, text="💾 Save Model", 
                  command=self.save_model, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame2, text="📂 Load Model", 
                  command=self.load_model, width=20).pack(side=tk.LEFT, padx=3)
        
        # Buttons row 3
        button_frame3 = ttk.Frame(main_frame)
        button_frame3.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame3, text="📤 Export Results", 
                  command=self.export_results, width=20).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame3, text="🔄 Switch Method", 
                  command=self.switch_method, width=20).pack(side=tk.LEFT, padx=3)
        
        # Status display
        self.status_label = ttk.Label(main_frame, text="Waiting for operation...", 
                                     font=("Arial", 12))
        self.status_label.pack(pady=10)
        
        # Results area
        self.results_text = tk.Text(main_frame, height=25, width=120, 
                                   font=("Consolas", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def log(self, message):
        """Log message to UI"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def load_file(self, filepath):
        """Load file (CSV/XLSX supported)"""
        try:
            file_ext = Path(filepath).suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(filepath)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            self.log(f"✅ File loaded: {Path(filepath).name}")
            self.log(f"   - Type: {file_ext}")
            self.log(f"   - Rows: {len(df)}")
            self.log(f"   - Columns: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            self.log(f"❌ File load failed: {e}")
            return None
    
    def load_multiple_training_data(self):
        """Load multiple training data files"""
        filepaths = filedialog.askopenfilenames(
            title="Select multiple training data files",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filepaths:
            self.log(f"🔄 Loading {len(filepaths)} files...")
            all_data = []
            successful_files = []
            sample_names = set()
            
            for filepath in filepaths:
                df = self.load_file(filepath)
                if df is not None:
                    # Add source file column
                    df['source_file'] = Path(filepath).name
                    all_data.append(df)
                    successful_files.append(filepath)
                    
                    # Extract sample name
                    if 'SampleID' in df.columns:
                        sample_names.update(df['SampleID'].unique())
                    else:
                        # If no SampleID column, use file stem as sample name
                        sample_name = Path(filepath).stem
                        sample_names.add(sample_name)
                        df['SampleID'] = sample_name
            
            if all_data:
                # Concatenate all data
                self.training_data = pd.concat(all_data, ignore_index=True)
                self.training_files = successful_files
                self.sample_list = sorted(list(sample_names))
                
                self.log(f"✅ Batch load complete: {len(self.training_data)} grains")
                self.log(f"📁 Loaded files: {[Path(f).name for f in successful_files]}")
                self.log(f"🔍 Samples: {self.sample_list}")
                
                self.validate_training_data(self.training_data)
            else:
                self.log("❌ No files loaded successfully")
    
    def validate_training_data(self, df):
        """Validate training data"""
        # Check required columns
        required_cols = ['Volume3d (mm^3) ', 'EigenVal1', 'EigenVal2', 'EigenVal3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.log(f"⚠️ Missing required columns: {missing_cols}")
            return False
        
        # Show columns
        self.log(f"📋 Columns: {list(df.columns)}")
        
        return True
    
    def input_expert_thresholds(self):
        """Interactive input of expert thresholds"""
        if not self.sample_list:
            self.log("❌ Please import data files first")
            return
        
        # Simplified threshold input
        self.log("📝 Please input expert thresholds (format: SampleID:Threshold)")
        self.log("Example: Quantity_LE03:1.0e-06")
        
        # Create simple input dialog
        self.create_simple_threshold_input()
    
    def create_simple_threshold_input(self):
        """Create simple threshold input dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Input Expert Thresholds")
        dialog.geometry("600x400")
        dialog.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Input Expert Thresholds", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Description
        info_label = ttk.Label(main_frame, 
                              text="Enter a volume threshold per sample, one per line (SampleID:Threshold)", 
                              font=("Arial", 10))
        info_label.pack(pady=5)
        
        # Input box
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.threshold_text = tk.Text(text_frame, height=15, width=60, font=("Consolas", 10))
        self.threshold_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.threshold_text.yview)
        self.threshold_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Prefill example
        example_text = ""
        for sample_id in self.sample_list:
            example_text += f"{sample_id}:1.0e-06\n"
        self.threshold_text.insert(tk.END, example_text)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save", 
                  command=lambda: self.save_simple_thresholds(dialog), width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=dialog.destroy, width=15).pack(side=tk.LEFT, padx=5)
    
    def save_simple_thresholds(self, dialog):
        """Save thresholds"""
        text_content = self.threshold_text.get("1.0", tk.END).strip()
        lines = text_content.split('\n')
        
        self.expert_thresholds = {}
        valid_count = 0
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                try:
                    sample_id, threshold_str = line.split(':', 1)
                    sample_id = sample_id.strip()
                    threshold = float(threshold_str.strip())
                    self.expert_thresholds[sample_id] = threshold
                    valid_count += 1
                except ValueError:
                    self.log(f"⚠️ Invalid format: {line}")
        
        self.log(f"✅ Saved expert thresholds for {valid_count} samples")
        dialog.destroy()
    
    def train_model(self):
        """Train model"""
        if self.training_data is None:
            self.log("❌ Please load training data first")
            return
        
        if not self.expert_thresholds:
            self.log("❌ Please enter expert thresholds first")
            return
        
        try:
            self.log("🔄 Training model...")
            self.root.update()
            
            # Generate labels from expert thresholds
            self.generate_labels_from_thresholds()
            
            # Resolution-aware unified features: log1p(V_vox_cont) + Joshua6, standardized
            missing = [sid for sid in self.sample_list if sid not in self.voxel_sizes]
            if missing:
                self.log(f"❌ Missing voxel sizes (mm) for samples: {missing}")
                return
            first_sample = self.sample_list[0]
            voxel_mm = float(self.voxel_sizes[first_sample])
            self.features = self.resolution_aware_engineer.extract(self.training_data, voxel_size_mm=voxel_mm, fit_scaler=True)
            self.log(f"🔧 Resolution-aware features prepared: {self.features.shape[1]} columns")
            
            self.log(f"📈 Extracted {len(self.features.columns)} features")
            
            # Prepare training data
            X = self.features.fillna(0)
            y = self.training_data['label'].values
            
            # Choose model
            if LIGHTGBM_AVAILABLE:
                self.log("🚀 Training with LightGBM...")
                # LightGBM
                train_data = lgb.Dataset(X, label=y)
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
                self.model = lgb.train(params, train_data, num_boost_round=100)
            else:
                self.log("🌲 Training with RandomForest...")
                # RandomForest
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                self.model.fit(X, y)
            
            # Compute training scores
            if hasattr(self.model, 'predict_proba'):
                train_proba = self.model.predict_proba(X)[:, 1]
            else:
                train_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
            
            train_auc = roc_auc_score(y, train_proba)
            
            # Compute accuracy - convert probabilities to binary
            if hasattr(self.model, 'predict'):
                # LightGBM predict returns probabilities; convert to binary
                train_pred_proba = self.model.predict(X)
                train_pred = (train_pred_proba > 0.5).astype(int)
            else:
                train_pred = (train_proba > 0.5).astype(int)
            
            train_accuracy = accuracy_score(y, train_pred)
            
            # Precision and recall
            precision = precision_score(y, train_pred)
            recall = recall_score(y, train_pred)
            f1 = f1_score(y, train_pred)
            
            self.log(f"✅ Training complete!")
            self.log(f"   - Num features: {len(self.features.columns)}")
            self.log(f"   - Num samples: {len(X)}")
            self.log(f"   - Train AUC: {train_auc:.3f}")
            self.log(f"   - Train accuracy: {train_accuracy:.3f}")
            self.log(f"   - Precision: {precision:.3f}")
            self.log(f"   - Recall: {recall:.3f}")
            self.log(f"   - F1 score: {f1:.3f}")
            
            # Show training files info
            if len(self.training_files) > 1:
                self.log(f"   - Num training files: {len(self.training_files)}")
                self.log(f"   - Sources: {[Path(f).name for f in self.training_files]}")
            
            # Show feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = self.features.columns
                top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5]
                self.log(f"🔝 Top features: {[f'{name}({imp:.3f})' for name, imp in top_features]}")
            elif hasattr(self.model, 'feature_importance'):
                importance = self.model.feature_importance(importance_type='gain')
                feature_names = self.features.columns
                top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5]
                self.log(f"🔝 Top features: {[f'{name}({imp:.3f})' for name, imp in top_features]}")
            
            # Save results for visualization
            self.training_results = {
                'X': X,
                'y': y,
                'train_proba': train_proba,
                'train_pred': train_pred,
                'train_auc': train_auc,
                'train_accuracy': train_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'features': self.features
            }
            
            # Show visualization button
            self.show_training_visualization_button()
            
        except Exception as e:
            self.log(f"❌ Training failed: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
    
    def extract_simple_features(self, df):
        """Enhanced feature extraction focusing on artifact patterns"""
        features = {}
        
        # Basic geometric features
        features['log_volume'] = np.log10(df['Volume3d (mm^3) '].values)
        features['volume'] = df['Volume3d (mm^3) '].values
        
        # Ellipsoid axis lengths
        if 'EigenVal1' in df.columns:
            eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
            a, b, c = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
            
            features['a'] = a
            features['b'] = b
            features['c'] = c
            
            # Axis ratios (critical for artifact detection)
            features['a_b_ratio'] = a / b
            features['a_c_ratio'] = a / c
            features['b_c_ratio'] = b / c
            
            # Sphericity
            features['sphericity'] = c / a
            
            # Anisotropy indicators
            lambda_sum = np.sum(eigenvals, axis=1)
            features['anisotropy'] = (eigenvals[:, 0] - eigenvals[:, 2]) / lambda_sum
            features['lambda_diff_12'] = np.abs(eigenvals[:, 0] - eigenvals[:, 1])
            features['lambda_diff_23'] = np.abs(eigenvals[:, 1] - eigenvals[:, 2])
        
        # Shape features (critical for artifact detection)
        if 'Elongation' in df.columns:
            features['elongation'] = df['Elongation'].values
        if 'Flatness' in df.columns:
            features['flatness'] = df['Flatness'].values
        
        # Eigenvector alignment features (key for voxel alignment artifacts)
        if all(col in df.columns for col in ['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']):
            # First eigenvector (longest axis)
            eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
            
            # Calculate angles with voxel axes (X, Y, Z)
            # Artifacts tend to align with voxel grid
            x_axis = np.array([1, 0, 0])
            y_axis = np.array([0, 1, 0])
            z_axis = np.array([0, 0, 1])
            
            # Dot product to get alignment
            features['eigenvec1_x_alignment'] = np.abs(np.dot(eigenvec1, x_axis))
            features['eigenvec1_y_alignment'] = np.abs(np.dot(eigenvec1, y_axis))
            features['eigenvec1_z_alignment'] = np.abs(np.dot(eigenvec1, z_axis))
            
            # Maximum alignment (artifacts often align with one axis)
            features['eigenvec1_max_alignment'] = np.maximum.reduce([
                features['eigenvec1_x_alignment'],
                features['eigenvec1_y_alignment'],
                features['eigenvec1_z_alignment']
            ])
        
        if all(col in df.columns for col in ['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']):
            # Second eigenvector
            eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
            features['eigenvec2_x_alignment'] = np.abs(np.dot(eigenvec2, x_axis))
            features['eigenvec2_y_alignment'] = np.abs(np.dot(eigenvec2, y_axis))
            features['eigenvec2_z_alignment'] = np.abs(np.dot(eigenvec2, z_axis))
            features['eigenvec2_max_alignment'] = np.maximum.reduce([
                features['eigenvec2_x_alignment'],
                features['eigenvec2_y_alignment'],
                features['eigenvec2_z_alignment']
            ])
        
        if all(col in df.columns for col in ['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']):
            # Third eigenvector (shortest axis)
            eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
            features['eigenvec3_x_alignment'] = np.abs(np.dot(eigenvec3, x_axis))
            features['eigenvec3_y_alignment'] = np.abs(np.dot(eigenvec3, y_axis))
            features['eigenvec3_z_alignment'] = np.abs(np.dot(eigenvec3, z_axis))
            features['eigenvec3_max_alignment'] = np.maximum.reduce([
                features['eigenvec3_x_alignment'],
                features['eigenvec3_y_alignment'],
                features['eigenvec3_z_alignment']
            ])
        
        # Volume-based features (small volumes often artifacts)
        features['is_small_volume'] = (features['volume'] < 1e-6).astype(int)
        features['is_very_small_volume'] = (features['volume'] < 1e-7).astype(int)
        
        # Combined artifact indicators
        if 'elongation' in features and 'flatness' in features:
            # High elongation + high flatness often indicates artifacts
            features['elongation_flatness_product'] = features['elongation'] * features['flatness']
            features['is_high_elongation'] = (features['elongation'] > 0.8).astype(int)
            features['is_high_flatness'] = (features['flatness'] > 0.8).astype(int)
        
        # Voxel alignment artifact indicator
        if 'eigenvec1_max_alignment' in features:
            features['is_voxel_aligned'] = (features['eigenvec1_max_alignment'] > 0.9).astype(int)
        
        return pd.DataFrame(features)
    
    def generate_labels_from_thresholds(self):
        """Generate labels based on expert thresholds (voxel-domain)"""
        if 'SampleID' not in self.training_data.columns:
            self.log("❌ SampleID column missing in training data")
            return
        
        # Ensure voxel sizes are present
        if not self.voxel_sizes:
            self.log("❌ Please input voxel sizes first (mm)")
            return
        
        # Convert absolute thresholds (mm^3) to voxel thresholds per sample (voxel size stored in mm)
        sample_threshold_vox = {}
        for sample_id, t_abs in self.expert_thresholds.items():
            if sample_id in self.voxel_sizes:
                try:
                    voxel_mm = float(self.voxel_sizes[sample_id])
                    voxel_vol = voxel_mm ** 3
                    sample_threshold_vox[sample_id] = int(np.ceil(float(t_abs) / voxel_vol))
                except Exception:
                    continue
        
        # Generate labels in voxel-domain
        labels = []
        for _, row in self.training_data.iterrows():
            sample_id = row.get('SampleID')
            volume_mm3 = row.get('Volume3d (mm^3) ', None)
            if sample_id in sample_threshold_vox and volume_mm3 is not None and sample_id in self.voxel_sizes:
                voxel_mm = float(self.voxel_sizes[sample_id])
                voxel_vol = voxel_mm ** 3
                v_vox = float(volume_mm3) / voxel_vol
                t_vox = sample_threshold_vox[sample_id]
                label = 1 if v_vox < t_vox else 0
            else:
                # If no expert threshold, mark unknown(-1)
                label = -1
            labels.append(label)
        
        self.training_data['label'] = labels
        label_counts = self.training_data['label'].value_counts().to_dict()
        
        # Label distribution
        self.log(f"📊 Label distribution:")
        self.log(f"   - Normal (0): {label_counts.get(0, 0)}")
        self.log(f"   - Artifact (1): {label_counts.get(1, 0)}")
        self.log(f"   - Unknown (-1): {label_counts.get(-1, 0)}")
        
        # Remove unknowns
        if -1 in label_counts and label_counts[-1] > 0:
            self.training_data = self.training_data[self.training_data['label'] != -1].copy()
            self.log(f"⚠️ Removed {label_counts[-1]} particles without expert thresholds")
    
    def load_test_data(self):
        """Load test data"""
        filepath = filedialog.askopenfilename(
            title="Select Test Data File",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            df = self.load_file(filepath)
            if df is not None:
                self.test_data = df
                self.log(f"✅ Test data loaded successfully: {len(self.test_data)} particles")
                
                # Extract sample ID from filename
                sample_id = os.path.splitext(os.path.basename(filepath))[0]
                
                # Ask for voxel size for test data
                self.log("📏 Please input voxel size for test data (mm/voxel):")
                self.log("   Example: 0.03 means each voxel edge length is 0.03mm")
                self.log("   If unknown, you can use 0.03 as default value")
                
                # Create voxel size input dialog
                voxel_window = tk.Toplevel(self.root)
                voxel_window.title("Input Test Data Voxel Size")
                voxel_window.geometry("400x200")
                voxel_window.transient(self.root)
                voxel_window.grab_set()
                
                tk.Label(voxel_window, text=f"Voxel size for test data: {sample_id}", 
                        font=("Arial", 12, "bold")).pack(pady=10)
                
                tk.Label(voxel_window, text="Voxel size (mm/voxel):", 
                        font=("Arial", 10)).pack(pady=5)
                
                voxel_entry = tk.Entry(voxel_window, font=("Arial", 10), width=20)
                voxel_entry.pack(pady=5)
                voxel_entry.insert(0, "0.03")  # Default value
                
                def save_voxel_size():
                    try:
                        voxel_size = float(voxel_entry.get())
                        if sample_id not in self.voxel_sizes:
                            self.voxel_sizes[sample_id] = voxel_size
                        self.log(f"✅ Test data voxel size: {sample_id} = {voxel_size} mm")
                        voxel_window.destroy()
                    except ValueError:
                        tk.messagebox.showerror("Error", "Please enter a valid number")
                
                tk.Button(voxel_window, text="Save", command=save_voxel_size, 
                         font=("Arial", 10), width=10).pack(pady=10)
    
    def predict_analysis(self):
        """Predict analysis"""
        if self.test_data is None:
            self.log("❌ Please load test data first")
            return
        
        if self.model is None:
            self.log("❌ Please train the model first")
            return
        
        try:
            self.log("🔄 Starting prediction analysis...")
            
            # Feature extraction - resolution-aware, same scaler as training
            if not self.voxel_sizes:
                self.log("❌ Please input voxel sizes first (mm)")
                return
            # Use first sample voxel size for transform (scaler already fit at training)
            first_sample = self.sample_list[0] if self.sample_list else list(self.voxel_sizes.keys())[0]
            voxel_mm = float(self.voxel_sizes[first_sample])
            test_features = self.resolution_aware_engineer.extract(self.test_data, voxel_size_mm=voxel_mm, fit_scaler=False)
            self.log(f"🔧 Resolution-aware prediction features: {test_features.shape[1]}")
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(test_features.values)[:, 1]
            else:
                # LightGBM prediction with shape check disabled
                probabilities = self.model.predict(test_features.values, 
                                                 num_iteration=self.model.best_iteration,
                                                 predict_disable_shape_check=True)
            
            # Calculate threshold - use model predicted probabilities
            volumes = self.test_data['Volume3d (mm^3) '].values
            voxel_vol = voxel_mm ** 3
            voxel_counts_cont = volumes / voxel_vol
            voxel_counts_int = np.ceil(voxel_counts_cont).astype(int)
            
            # Compute thresholds in voxel domain EXACTLY as in visualization
            thresholds = np.logspace(
                np.log10(max(voxel_counts_cont.min(), 1e-12)),
                np.log10(voxel_counts_cont.max()),
                50,
            )
            artifact_rates = []
            for t in thresholds:
                retained = voxel_counts_cont >= t
                # Artifact rate = mean predicted artifact probability on retained set
                artifact_rates.append(np.mean(probabilities[retained]) if np.sum(retained) > 0 else 0.0)
            inflection_threshold = self.find_inflection_threshold(thresholds, artifact_rates)
            # Strict: remove ALL particles with P>0.05
            strict_mask = probabilities > 0.05
            noise_removal_threshold = float(np.max(voxel_counts_cont[strict_mask])) if np.any(strict_mask) else None
            if inflection_threshold is not None and noise_removal_threshold is not None:
                if noise_removal_threshold < inflection_threshold:
                    noise_removal_threshold = inflection_threshold

            total_count = len(volumes)
            uncertainty = np.std(probabilities)
            self.log(f"✅ Prediction analysis complete!")
            self.log(f"   - Total particles: {total_count}")
            if inflection_threshold is not None:
                self.log(f"   - Loose threshold (Inflection): {int(np.ceil(inflection_threshold))} vox | {(int(np.ceil(inflection_threshold))*voxel_vol):.2e} mm³")
            if noise_removal_threshold is not None:
                self.log(f"   - Strict threshold (P>0.05): {int(np.ceil(noise_removal_threshold))} vox | {(int(np.ceil(noise_removal_threshold))*voxel_vol):.2e} mm³")
            
            # Save results
            self.probabilities = probabilities
            
        except Exception as e:
            self.log(f"❌ Prediction analysis failed: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
    
    def show_training_visualization_button(self):
        """Show training visualization button prompt"""
        self.log("📊 Click 'Training Visualization' button to view detailed training results")
    
    def show_training_visualization(self):
        """Display training visualization"""
        if self.training_results is None:
            self.log("❌ Please train model first")
            return
        
        try:
            # Create visualization window
            if self.visualization_window is not None:
                self.visualization_window.destroy()
            
            # Display training results
            self.visualization_window = tk.Toplevel(self.root)
            self.visualization_window.title("Training Results Visualization")
            self.visualization_window.geometry("1200x800")
            
            # Create matplotlib figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Model Training Results Analysis', fontsize=16, fontweight='bold')
            
            # 1. ROC Curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(self.training_results['y'], self.training_results['train_proba'])
            axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {self.training_results["train_auc"]:.3f}')
            axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1)
            axes[0, 0].set_xlabel('False Positive Rate (FPR)')
            axes[0, 0].set_ylabel('True Positive Rate (TPR)')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Feature Importance
            feature_names = self.training_results['features'].columns
            
            if hasattr(self.model, 'feature_importances_'):
                # RandomForest
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'feature_importance'):
                # LightGBM
                importance = self.model.feature_importance(importance_type='gain')
            else:
                # Fallback: use permutation importance
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(self.model, self.training_results['X'], 
                                                       self.training_results['y'], n_repeats=5, random_state=42)
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
                axes[0, 1].text(0.5, 0.5, 'No feature importance available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Feature Importance')
            
            # 3. Prediction Probability Distribution
            normal_proba = self.training_results['train_proba'][self.training_results['y'] == 0]
            artifact_proba = self.training_results['train_proba'][self.training_results['y'] == 1]
            
            axes[1, 0].hist(normal_proba, bins=30, alpha=0.7, label='Normal Particles', color='blue')
            axes[1, 0].hist(artifact_proba, bins=30, alpha=0.7, label='Artifact Particles', color='red')
            axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
            axes[1, 0].set_xlabel('Prediction Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Prediction Probability Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Performance Metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [self.training_results['train_accuracy'], 
                     self.training_results['precision'],
                     self.training_results['recall'],
                     self.training_results['f1']]
            
            bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Display values on bars
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Embed in tkinter window
            canvas = FigureCanvasTkAgg(fig, self.visualization_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add save buttons
            save_frame = tk.Frame(self.visualization_window)
            save_frame.pack(pady=10)
            
            tk.Button(save_frame, text="Save as PNG", 
                     command=lambda: self.save_chart(fig, "training_visualization", "png")).pack(side=tk.LEFT, padx=5)
            tk.Button(save_frame, text="Save as SVG", 
                     command=lambda: self.save_chart(fig, "training_visualization", "svg")).pack(side=tk.LEFT, padx=5)
            
            self.log("✅ Training visualization displayed")
            
        except Exception as e:
            self.log(f"❌ Visualization failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")
    
    def show_prediction_visualization(self):
        """Display prediction visualization"""
        if self.test_data is None or self.probabilities is None:
            self.log("❌ Please perform prediction analysis first")
            return
        
        try:
            # Create visualization window
            if self.visualization_window is not None:
                self.visualization_window.destroy()
            
            self.visualization_window = tk.Toplevel(self.root)
            self.visualization_window.title("Prediction Results Visualization")
            self.visualization_window.geometry("1200x800")
            
            # Create matplotlib figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Prediction Results Analysis', fontsize=16, fontweight='bold')
            
            volumes = self.test_data['Volume3d (mm^3) '].values
            # voxel info
            if not self.voxel_sizes:
                self.log("❌ Please input voxel sizes first (mm)")
                return
            first_sample = self.sample_list[0] if self.sample_list else list(self.voxel_sizes.keys())[0]
            voxel_mm = float(self.voxel_sizes[first_sample])
            voxel_vol = voxel_mm ** 3
            voxels_cont = np.clip(volumes / voxel_vol, a_min=1e-12, a_max=None)
            voxels_int = np.ceil(voxels_cont).astype(int)
            
            # 1. Voxel distribution (primary) with mm^3 annotation
            axes[0, 0].hist(np.log10(voxels_cont), bins=50, alpha=0.7, color='skyblue')
            axes[0, 0].set_xlabel('log10(Voxel Count)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Voxel Count Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            # annotate equivalent mm^3 for a few ticks
            xticks = axes[0, 0].get_xticks()
            mm3_labels = []
            for t in xticks:
                vc = 10**t
                mm3_labels.append(f"{vc*voxel_vol:.1e}")
            ax_top = axes[0, 0].secondary_xaxis('top')
            ax_top.set_xticks(xticks)
            ax_top.set_xticklabels(mm3_labels, rotation=0)
            ax_top.set_xlabel('Equivalent Volume (mm³)')
            
            # 2. Prediction probability vs voxel count
            scatter = axes[0, 1].scatter(np.log10(voxels_cont), self.probabilities, 
                                       c=self.probabilities, cmap='RdYlBu_r', alpha=0.6)
            axes[0, 1].set_xlabel('log10(Voxel Count)')
            axes[0, 1].set_ylabel('Artifact Probability')
            axes[0, 1].set_title('Prediction Probability vs Voxel Count')
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 1], label='Artifact Probability')
            
            # 3. Prediction probability distribution
            axes[1, 0].hist(self.probabilities, bins=30, alpha=0.7, color='lightgreen')
            axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
            axes[1, 0].set_xlabel('Prediction Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Prediction Probability Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Threshold analysis in voxel domain (with mm³ overlay)
            thresholds = np.logspace(np.log10(max(voxels_cont.min(), 1e-12)), np.log10(voxels_cont.max()), 50)
            retention_rates = []
            artifact_rates = []
            
            for thresh in thresholds:
                retained = voxels_cont >= thresh
                retention_rate = np.mean(retained)
                # Artifact rate = mean probability on retained set (restore previous definition)
                if np.sum(retained) > 0:
                    artifact_rate = np.mean(self.probabilities[retained])
                else:
                    artifact_rate = 0.0
                
                retention_rates.append(retention_rate)
                artifact_rates.append(artifact_rate)
            
            ax2 = axes[1, 1].twinx()
            line1 = axes[1, 1].plot(np.log10(thresholds), retention_rates, 'b-', label='Retention Rate')
            line2 = ax2.plot(np.log10(thresholds), artifact_rates, 'r-', label='Artifact Rate')
            
            # Calculate dual thresholds
            # 1. Inflection point threshold (use artifact-rate curvature on removed set)
            inflection_threshold = self.find_inflection_threshold(thresholds, artifact_rates)
            
            # 2. Strict threshold: remove all P>0.05
            strict_mask = self.probabilities > 0.05
            noise_removal_threshold = float(np.max(voxels_cont[strict_mask])) if np.any(strict_mask) else None

            # Enforce strict >= loose when both exist
            if inflection_threshold is not None and noise_removal_threshold is not None:
                if noise_removal_threshold < inflection_threshold:
                    noise_removal_threshold = inflection_threshold
            
            # Add threshold lines to the plot
            if inflection_threshold is not None:
                inf_vx = int(np.ceil(inflection_threshold))
                inf_mm3 = inf_vx * voxel_vol
                axes[1, 1].axvline(x=np.log10(inflection_threshold), color='green', linestyle='--', 
                                 linewidth=2, label=f'Inflection: {inf_vx} vox | {inf_mm3:.2e} mm³')
            
            if noise_removal_threshold is not None:
                nr_vx = int(np.ceil(noise_removal_threshold))
                nr_mm3 = nr_vx * voxel_vol
                axes[1, 1].axvline(x=np.log10(noise_removal_threshold), color='orange', linestyle='--', 
                                 linewidth=2, label=f'Strict (cliff): {nr_vx} vox | {nr_mm3:.2e} mm³')
            
            axes[1, 1].set_xlabel('log10(Voxel Threshold)')
            axes[1, 1].set_ylabel('Retention Rate', color='b')
            ax2.set_ylabel('Artifact Rate', color='r')
            axes[1, 1].set_title('Dual Threshold Analysis (Voxel Domain)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[1, 1].legend(lines, labels, loc='center right')
            
            # Log threshold information
            self.log(f"📊 Dual Threshold Analysis:")
            if inflection_threshold is not None:
                self.log(f"   - Inflection: {int(np.ceil(inflection_threshold))} vox | {(int(np.ceil(inflection_threshold))*voxel_vol):.2e} mm³")
            if noise_removal_threshold is not None:
                self.log(f"   - Strict (cliff): {int(np.ceil(noise_removal_threshold))} vox | {(int(np.ceil(noise_removal_threshold))*voxel_vol):.2e} mm³")
            
            plt.tight_layout()
            
            # Embed in tkinter window
            canvas = FigureCanvasTkAgg(fig, self.visualization_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add save buttons
            save_frame = tk.Frame(self.visualization_window)
            save_frame.pack(pady=10)
            
            tk.Button(save_frame, text="Save as PNG", 
                     command=lambda: self.save_chart(fig, "prediction_visualization", "png")).pack(side=tk.LEFT, padx=5)
            tk.Button(save_frame, text="Save as SVG", 
                     command=lambda: self.save_chart(fig, "prediction_visualization", "svg")).pack(side=tk.LEFT, padx=5)
            
            self.log("✅ Prediction visualization displayed")
            
        except Exception as e:
            self.log(f"❌ Prediction visualization failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")
    
    def find_inflection_threshold(self, thresholds, artifact_rates):
        """Find inflection point threshold using second derivative"""
        try:
            # Calculate second derivative to find inflection points
            if len(artifact_rates) < 3:
                return None
            
            # Smooth the data first
            from scipy.ndimage import gaussian_filter1d
            smoothed_rates = gaussian_filter1d(artifact_rates, sigma=1.0)
            
            # Calculate second derivative
            second_derivative = np.gradient(np.gradient(smoothed_rates))
            
            # Find the most significant inflection point (maximum second derivative)
            inflection_idx = np.argmax(second_derivative)
            
            if inflection_idx > 0 and inflection_idx < len(thresholds) - 1:
                return thresholds[inflection_idx]
            else:
                # Fallback: find where artifact rate starts to increase significantly
                for i in range(1, len(artifact_rates)):
                    if artifact_rates[i] - artifact_rates[i-1] > 0.1:
                        return thresholds[i]
                return None
        except Exception as e:
            self.log(f"Warning: Could not find inflection threshold: {e}")
            return None

    def find_cliff_threshold(self, voxels_cont: np.ndarray, probabilities: np.ndarray) -> float:
        """Find cliff position from scatter (voxel_count vs probability).

        Strategy:
        - Bin along log10(voxel_count), compute median probability per bin
        - Prefer the first crossing where median drops from >high to <low
        - Fallback: use steepest negative slope of the smoothed median curve
        Returns voxel_count threshold (float) or None.
        """
        try:
            x = np.log10(np.clip(voxels_cont.astype(float), 1e-12, None))
            p = probabilities.astype(float)
            if len(x) < 10:
                return None
            bins = np.linspace(np.min(x), np.max(x), 60)
            idx = np.digitize(x, bins) - 1
            centers = []
            med = []
            for b in range(len(bins) - 1):
                m = (idx == b)
                if np.any(m):
                    centers.append(0.5 * (bins[b] + bins[b + 1]))
                    med.append(np.median(p[m]))
            if len(med) < 5:
                return None
            centers = np.array(centers)
            med = np.array(med)
            # Crossing heuristic
            high, low = 0.6, 0.05
            for i in range(1, len(med)):
                if med[i] <= low and np.max(med[:i]) >= high:
                    return float(10 ** centers[i])
            # Fallback: steepest descent
            try:
                from scipy.ndimage import gaussian_filter1d
                med_s = gaussian_filter1d(med, sigma=1.0)
            except Exception:
                med_s = med
            deriv = np.gradient(med_s, centers)
            cliff_idx = int(np.argmin(deriv))
            return float(10 ** centers[cliff_idx])
        except Exception as e:
            self.log(f"Warning: find_cliff_threshold failed: {e}")
            return None
    
    def find_noise_removal_threshold(self, thresholds, artifact_rates, volumes, probabilities):
        """Find threshold that removes all high-probability noise particles"""
        try:
            # Define high noise probability threshold (e.g., >0.8)
            noise_prob_threshold = 0.8
            
            # Find particles with high artifact probability
            high_noise_mask = probabilities > noise_prob_threshold
            
            if np.sum(high_noise_mask) == 0:
                # No high-noise particles found, use a conservative threshold
                return thresholds[int(len(thresholds) * 0.1)]  # 10th percentile
            
            # Find the minimum volume among high-noise particles
            high_noise_volumes = volumes[high_noise_mask]
            min_noise_volume = np.min(high_noise_volumes)
            
            # Find the threshold that would remove all high-noise particles
            # Use a slightly higher threshold to ensure complete removal
            safety_factor = 1.1
            noise_removal_threshold = min_noise_volume * safety_factor
            
            # Ensure the threshold is within our range
            if noise_removal_threshold < thresholds[0]:
                noise_removal_threshold = thresholds[0]
            elif noise_removal_threshold > thresholds[-1]:
                noise_removal_threshold = thresholds[-1]
            
            return noise_removal_threshold
            
        except Exception as e:
            self.log(f"Warning: Could not find noise removal threshold: {e}")
            return None
    
    def save_model(self):
        """Save model"""
        if self.model is None:
            self.log("❌ Please train model first")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                title="Save Model"
            )
            
            if filename:
                import pickle
                # Save model and related data
                model_data = {
                    'model': self.model,
                    'features': self.features.columns.tolist() if self.features is not None else None,
                    'expert_thresholds': self.expert_thresholds,
                    'voxel_sizes': self.voxel_sizes,
                    'training_files': self.training_files
                }
                with open(filename, 'wb') as f:
                    pickle.dump(model_data, f)
                self.log(f"✅ Model saved to: {filename}")
                
        except Exception as e:
            self.log(f"❌ Model save failed: {e}")
    
    def load_model(self):
        """Load model"""
        try:
            # Load model
            model_path = filedialog.askopenfilename(
                title="Load Model",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if model_path:
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Check if model_data is a dictionary or direct model object
                if isinstance(model_data, dict):
                    # Load model and related data from dictionary
                    self.model = model_data['model']
                    if 'features' in model_data and model_data['features'] is not None:
                        # Recreate features DataFrame structure
                        import pandas as pd
                        self.features = pd.DataFrame(columns=model_data['features'])
                    if 'expert_thresholds' in model_data:
                        self.expert_thresholds = model_data['expert_thresholds']
                    if 'voxel_sizes' in model_data:
                        self.voxel_sizes = model_data['voxel_sizes']
                    if 'training_files' in model_data:
                        self.training_files = model_data['training_files']
                    
                    self.log(f"✅ Model loaded from: {model_path}")
                    self.log(f"   - Features: {len(model_data.get('features', []))}")
                    self.log(f"   - Expert thresholds: {len(model_data.get('expert_thresholds', {}))}")
                    self.log(f"   - Voxel sizes: {len(model_data.get('voxel_sizes', {}))}")
                    
                    # Set training results to enable visualization
                    if 'features' in model_data and model_data['features'] is not None:
                        self.training_results = {
                            'model': self.model,
                            'features': self.features,
                            'train_auc': 0.95,  # Placeholder
                            'train_accuracy': 0.90,  # Placeholder
                            'precision': 0.90,  # Placeholder
                            'recall': 0.90,  # Placeholder
                            'f1': 0.90,  # Placeholder
                            'y': None,  # Will be set when needed
                            'train_proba': None,  # Will be set when needed
                            'X': None  # Will be set when needed
                        }
                else:
                    # Direct model object (old format)
                    self.model = model_data
                    self.log(f"✅ Model loaded from: {model_path}")
                    self.log("   - Note: This is an old format model, some features may not be available")
                    
                    # For old format models, we need to extract features from the model
                    if hasattr(self.model, 'feature_name'):
                        # LightGBM model
                        feature_names = self.model.feature_name()
                        import pandas as pd
                        self.features = pd.DataFrame(columns=feature_names)
                        self.log(f"   - Extracted {len(feature_names)} features from model")
                    elif hasattr(self.model, 'feature_importances_'):
                        # RandomForest model
                        n_features = len(self.model.feature_importances_)
                        feature_names = [f'feature_{i}' for i in range(n_features)]
                        import pandas as pd
                        self.features = pd.DataFrame(columns=feature_names)
                        self.log(f"   - Extracted {n_features} features from model")
                
        except Exception as e:
            self.log(f"❌ Model load failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")
    
    def multi_sample_test(self):
        """Multi-sample test with threshold calculation for each sample"""
        if self.model is None:
            self.log("❌ Please train or load model first")
            return
        
        try:
            # Select multiple test files
            test_files = filedialog.askopenfilenames(
                title="Select Multiple Test Files",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not test_files:
                return
            
            self.log(f"🔄 Loading {len(test_files)} test files...")
            
            # Process each test file
            results = []
            for i, file_path in enumerate(test_files):
                self.log(f"📁 Processing file {i+1}/{len(test_files)}: {os.path.basename(file_path)}")
                
                try:
                    # Load test data
                    if file_path.endswith('.xlsx'):
                        test_data = pd.read_excel(file_path)
                    else:
                        test_data = pd.read_csv(file_path)
                    
                    # Extract sample ID from filename
                    sample_id = os.path.splitext(os.path.basename(file_path))[0]
                    test_data['SampleID'] = sample_id
                    
                    # Extract features
                    if self.features is not None:
                        feature_columns = self.features.columns.tolist()
                    else:
                        # Use simple feature extraction
                        feature_columns = self.extract_simple_features(test_data).columns.tolist()
                    
                    # Prepare features for prediction
                    X_test = test_data[feature_columns].fillna(0)
                    
                    # Make predictions
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(X_test)[:, 1]
                    else:
                        probabilities = self.model.predict(X_test)
                    
                    # Calculate threshold for this sample
                    volumes = test_data['Volume3d (mm^3) '].values
                    threshold_result = self.calculate_adaptive_threshold(volumes, probabilities)
                    
                    # Store results
                    result = {
                        'sample_id': sample_id,
                        'file_path': file_path,
                        'total_particles': len(test_data),
                        'retained_particles': threshold_result['retained_count'],
                        'removed_particles': threshold_result['removed_count'],
                        'retention_rate': threshold_result['retention_rate'],
                        'predicted_threshold': threshold_result['threshold'],
                        'target_artifact_rate': threshold_result['target_artifact_rate'],
                        'actual_artifact_rate': threshold_result['actual_artifact_rate'],
                        'artifact_rate_error': threshold_result['artifact_rate_error']
                    }
                    results.append(result)
                    
                    self.log(f"   ✅ {sample_id}: {result['retained_particles']}/{result['total_particles']} retained, threshold={result['predicted_threshold']:.6f}")
                    
                except Exception as e:
                    self.log(f"   ❌ Error processing {file_path}: {e}")
                    continue
            
            # Display summary
            self.log(f"\n📊 Multi-Sample Test Results Summary:")
            self.log(f"   Total samples processed: {len(results)}")
            self.log(f"   Average retention rate: {np.mean([r['retention_rate'] for r in results]):.1%}")
            self.log(f"   Average threshold: {np.mean([r['predicted_threshold'] for r in results]):.6f}")
            
            # Save results to CSV
            if results:
                results_df = pd.DataFrame(results)
                output_path = filedialog.asksaveasfilename(
                    title="Save Multi-Sample Results",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                if output_path:
                    results_df.to_csv(output_path, index=False)
                    self.log(f"✅ Results saved to: {output_path}")
            
        except Exception as e:
            self.log(f"❌ Multi-sample test failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")
    
    def calculate_adaptive_threshold(self, volumes, probabilities, target_artifact_rate=0.05):
        """Calculate adaptive threshold for a sample"""
        # Sort by volume
        volume_indices = np.argsort(volumes)
        sorted_volumes = volumes[volume_indices]
        sorted_probabilities = probabilities[volume_indices]
        
        # Search for threshold that achieves target artifact rate
        best_threshold = None
        best_error = float('inf')
        
        # Grid search over volume percentiles
        percentiles = np.linspace(1, 50, 50)  # 1% to 50%
        
        for percentile in percentiles:
            threshold = np.percentile(sorted_volumes, percentile)
            
            # Calculate artifact rate for removed particles
            removed_mask = sorted_volumes < threshold
            if np.sum(removed_mask) > 0:
                actual_artifact_rate = np.mean(sorted_probabilities[removed_mask])
                error = abs(actual_artifact_rate - target_artifact_rate)
                
                if error < best_error:
                    best_error = error
                    best_threshold = threshold
        
        # Calculate final results
        retained_mask = volumes >= best_threshold
        retained_count = np.sum(retained_mask)
        removed_count = len(volumes) - retained_count
        retention_rate = retained_count / len(volumes)
        
        if removed_count > 0:
            actual_artifact_rate = np.mean(probabilities[~retained_mask])
        else:
            actual_artifact_rate = 0
        
        return {
            'threshold': best_threshold,
            'retained_count': retained_count,
            'removed_count': removed_count,
            'retention_rate': retention_rate,
            'target_artifact_rate': target_artifact_rate,
            'actual_artifact_rate': actual_artifact_rate,
            'artifact_rate_error': abs(actual_artifact_rate - target_artifact_rate)
        }
    
    def export_results(self):
        """Export results"""
        if self.test_data is None or self.probabilities is None:
            self.log("❌ Please perform prediction analysis first")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Results"
            )
            
            if filename:
                # Create results DataFrame
                results_df = self.test_data.copy()
                results_df['predicted_probability'] = self.probabilities
                results_df['predicted_label'] = (self.probabilities > 0.5).astype(int)
                
                # Calculate threshold
                volumes = self.test_data['Volume3d (mm^3) '].values
                v_min_star = np.percentile(volumes, 10)
                results_df['keep_flag'] = volumes >= v_min_star
                
                results_df.to_csv(filename, index=False)
                self.log(f"✅ Results exported to: {filename}")
                
        except Exception as e:
            self.log(f"❌ Export results failed: {e}")
    
    def analyze_features(self):
        """Analyze feature differences (传统方法)"""
        if self.training_data is None:
            self.log("❌ Please load training data first")
            return
        
        if not self.expert_thresholds:
            self.log("❌ Please enter expert thresholds first")
            return
        
        try:
            self.log("🔍 Starting traditional feature difference analysis...")
            
            # Generate labels
            self.generate_labels_from_thresholds()
            
            # Prepare data - clean non-numeric columns
            df = self.training_data.copy()
            
            # Remove possible string columns (like filenames), but keep SampleID and label
            string_columns = []
            for col in df.columns:
                if col in ['SampleID', 'label']:  # Keep these important columns
                    continue
                if df[col].dtype == 'object':
                    # Check if contains non-numeric data
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        string_columns.append(col)
            
            if string_columns:
                self.log(f"   Removed string columns: {string_columns}")
                df = df.drop(columns=string_columns)
            
            labels = df['label'].values
            sample_ids = df['SampleID'].values if 'SampleID' in df.columns else None
            
            # Use user-input voxel sizes
            if not self.voxel_sizes:
                self.log("❌ Please input voxel sizes first")
                return
            
            voxel_sizes = self.voxel_sizes
            
            # Traditional feature analysis is deprecated - use Joshua method instead
            self.log("📊 Traditional feature analysis is deprecated. Please use Joshua analysis instead.")
            
            self.log("✅ Traditional feature analysis completed")
            
        except Exception as e:
            self.log(f"❌ Feature analysis failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")
    
    def analyze_joshua_features(self):
        """Analyze features using Joshua method"""
        if self.training_data is None:
            self.log("❌ Please load training data first")
            return
        
        if not self.expert_thresholds:
            self.log("❌ Please enter expert thresholds first")
            return
        
        try:
            self.log("🔬 Starting Joshua feature analysis...")
            
            # Generate labels
            self.generate_labels_from_thresholds()
            
            # Prepare data
            df = self.training_data.copy()
            
            # Remove possible string columns
            string_columns = []
            for col in df.columns:
                if col in ['SampleID', 'label']:
                    continue
                if df[col].dtype == 'object':
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        string_columns.append(col)
            
            if string_columns:
                self.log(f"   Removed string columns: {string_columns}")
                df = df.drop(columns=string_columns)
            
            labels = df['label'].values
            sample_ids = df['SampleID'].values if 'SampleID' in df.columns else None
            
            # Use user-input voxel sizes (in mm)
            if not self.voxel_sizes:
                self.log("❌ Please input voxel sizes first")
                return
            
            # Set voxel size for Joshua engineer (use first sample's voxel size in mm)
            first_sample_id = list(self.voxel_sizes.keys())[0]
            self.joshua_feature_engineer.voxel_size_mm = float(self.voxel_sizes[first_sample_id])
            self.log(f"🔧 设置Joshua方法体素尺寸: {self.joshua_feature_engineer.voxel_size_mm:.4f} mm")
            
            # Perform Joshua feature analysis
            self.joshua_analysis_results = self.joshua_feature_analyzer.analyze_feature_differences(
                df, labels, sample_ids, self.voxel_sizes
            )
            
            # Display analysis results
            self.display_joshua_analysis_results()
            
            # Ask if to generate visualization
            self.log("📊 Joshua feature analysis completed! Generate visualization charts?")
            self.log("   Click '📊 Training Visualization' to view detailed charts")
            
            self.log("✅ Joshua feature analysis completed")
            
        except Exception as e:
            self.log(f"❌ Joshua feature analysis failed: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")
    
    def display_joshua_analysis_results(self):
        """Display Joshua analysis results"""
        if self.joshua_analysis_results is None:
            return
        
        feature_stats = self.joshua_analysis_results['feature_stats']
        joshua_features = self.joshua_analysis_results['joshua_features']
        
        # Display significant features
        significant_features = [(name, stats) for name, stats in feature_stats.items() 
                              if stats['is_significant']]
        significant_features.sort(key=lambda x: x[1]['cohens_d'], reverse=True)
        
        self.log("🎯 Joshua方法显著特征 (p<0.05, Cohen's d>0.2):")
        for name, stats in significant_features:
            self.log(f"   - {name}: d={stats['cohens_d']:.3f}, p={stats['p_value']:.2e}")
        
        # Display all features
        self.log(f"🔬 Joshua方法特征 (共{len(joshua_features.columns)}个):")
        for col in joshua_features.columns:
            stats = feature_stats[col]
            significance = "显著" if stats['is_significant'] else "不显著"
            self.log(f"   - {col}: d={stats['cohens_d']:.3f}, {significance}")
        
        # Display feature descriptions
        feature_descriptions = self.joshua_feature_engineer.get_feature_descriptions()
        self.log("📋 特征描述:")
        for feature, description in feature_descriptions.items():
            self.log(f"   - {feature}: {description}")
    
    def switch_method(self):
        """Switch between traditional and Joshua methods"""
        self.use_joshua_method = not self.use_joshua_method
        method_name = "Joshua方法" if self.use_joshua_method else "传统方法"
        self.log(f"🔄 已切换到: {method_name}")
        self.log(f"   - Joshua方法: 7个核心特征，基于对数-椭球张量")
        self.log(f"   - 传统方法: 25-30个特征，基于特征工程")
        
        # Update status
        if self.use_joshua_method:
            self.status_label.config(text="当前方法: Joshua (7特征)")
        else:
            self.status_label.config(text="当前方法: 传统 (25-30特征)")
    
    
    
    def save_chart(self, fig, base_name, format_type):
        """Save chart in specified format"""
        try:
            from datetime import datetime
            import os
            
            # Create output directory
            output_dir = "analysis_results"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save chart
            filename = f"{base_name}_{timestamp}.{format_type}"
            filepath = os.path.join(output_dir, filename)
            
            if format_type == "png":
                fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')
            elif format_type == "svg":
                fig.savefig(filepath, bbox_inches='tight', format='svg')
            
            self.log(f"✅ Chart saved as {format_type.upper()}: {filepath}")
            
        except Exception as e:
            self.log(f"❌ Chart save failed: {e}")
    
    def input_voxel_sizes(self):
        """Input voxel sizes"""
        if not self.sample_list:
            self.log("❌ Please load training data first")
            return
        
        try:
            # Create voxel size input window
            voxel_window = tk.Toplevel(self.root)
            voxel_window.title("Input Voxel Sizes")
            voxel_window.geometry("600x500")
            voxel_window.transient(self.root)
            voxel_window.grab_set()
            
            # Title
            title_label = ttk.Label(voxel_window, text="Input Voxel Sizes for Each Sample", 
                                  font=("Arial", 14, "bold"))
            title_label.pack(pady=10)
            
            # Description
            info_label = ttk.Label(voxel_window, 
                                 text="Enter voxel size in mm for each sample.\nThis is used for feature normalization.\nExample: 0.03 for 30μm resolution",
                                 font=("Arial", 10))
            info_label.pack(pady=5)
            
            # Create table frame
            table_frame = ttk.Frame(voxel_window)
            table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Create table
            columns = ('Sample ID', 'Voxel Size (mm)')
            tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
            
            # Set column headers
            tree.heading('Sample ID', text='Sample ID')
            tree.heading('Voxel Size (mm)', text='Voxel Size (mm)')
            
            # Set column widths
            tree.column('Sample ID', width=200)
            tree.column('Voxel Size (mm)', width=200)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            # Layout
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Populate data
            for sample_id in self.sample_list:
                current_value = self.voxel_sizes.get(sample_id, "0.03")  # Default: 0.03mm
                tree.insert('', 'end', values=(sample_id, current_value))
            
            # Double-click editing
            def on_double_click(event):
                item = tree.selection()[0]
                column = tree.identify_column(event.x)
                
                if column == '#2':  # Voxel Size column
                    self.edit_voxel_cell(tree, item, column)
            
            tree.bind('<Double-1>', on_double_click)
            
            # Button frame
            button_frame = ttk.Frame(voxel_window)
            button_frame.pack(fill=tk.X, padx=20, pady=10)
            
            def save_voxel_sizes():
                """Save voxel sizes"""
                try:
                    # Read data from table
                    for item in tree.get_children():
                        values = tree.item(item)['values']
                        sample_id = values[0]
                        # Values may already be numeric; convert to string safely before strip
                        voxel_size_str = str(values[1]) if values[1] is not None else ""
                        
                        if voxel_size_str.strip():
                            try:
                                voxel_size = float(voxel_size_str)
                                if voxel_size > 0:
                                    self.voxel_sizes[sample_id] = voxel_size
                                else:
                                    self.log(f"⚠️ Voxel size for sample {sample_id} must be greater than 0")
                                    return
                            except ValueError:
                                self.log(f"⚠️ Voxel size for sample {sample_id} is in an incorrect format")
                                return
                        else:
                            self.log(f"⚠️ Voxel size for sample {sample_id} cannot be empty")
                            return
                    
                    self.log(f"✅ Saved voxel sizes for {len(self.voxel_sizes)} samples")
                    for sample_id, size in self.voxel_sizes.items():
                        self.log(f"   - {sample_id}: {size:.4f} mm")
                    
                    voxel_window.destroy()
                    
                except Exception as e:
                    self.log(f"❌ Failed to save voxel sizes: {e}")
            
            def clear_all_voxel_sizes():
                """Clear all voxel sizes"""
                for item in tree.get_children():
                    tree.item(item, values=(tree.item(item)['values'][0], ""))
                self.voxel_sizes.clear()
                self.log("🗑️ Cleared all voxel sizes")
            
            # Buttons
            ttk.Button(button_frame, text="Save", command=save_voxel_sizes, width=15).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Clear All", command=clear_all_voxel_sizes, width=15).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=voxel_window.destroy, width=15).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            self.log(f"❌ Failed to create voxel size input window: {e}")
    
    def edit_voxel_cell(self, tree, item, column):
        """Edit voxel size cell"""
        # Get current value
        values = tree.item(item)['values']
        current_value = values[1] if len(values) > 1 else ""
        
        # Create entry box
        entry = ttk.Entry(tree)
        entry.insert(0, current_value)
        entry.select_range(0, tk.END)
        
        def save_edit(event=None):
            new_value = entry.get()
            sample_id = values[0]
            tree.item(item, values=(sample_id, new_value))
            entry.destroy()
        
        def cancel_edit(event=None):
            entry.destroy()
        
        entry.bind('<Return>', save_edit)
        entry.bind('<Escape>', cancel_edit)
        entry.bind('<FocusOut>', save_edit)
        
        # Position and display entry box
        bbox = tree.bbox(item, column)
        if bbox:
            entry.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
            entry.focus_set()

def main():
    """Main function"""
    root = tk.Tk()
    app = FixedMLGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
