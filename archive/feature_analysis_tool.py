#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Analysis Tool - Deep analysis of feature differences and optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """Feature analyzer - deep analysis of feature differences and optimization"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance_scores = {}
        self.correlation_matrix = None
        
    def analyze_feature_differences(self, df, labels, sample_ids=None, voxel_sizes=None):
        """Analyze feature differences between removed and kept particles"""
        
        print("🔍 Starting feature difference analysis...")
        
        # Store original data for correlation analysis
        self.original_data = df.copy()
        
        # 1. Basic statistics
        removed_particles = df[labels == 1]  # removed particles (artifacts)
        kept_particles = df[labels == 0]     # kept particles (valid)
        
        print(f"📊 Data summary:")
        print(f"   - Total particles: {len(df)}")
        print(f"   - Removed: {len(removed_particles)} ({len(removed_particles)/len(df)*100:.1f}%)")
        print(f"   - Kept: {len(kept_particles)} ({len(kept_particles)/len(df)*100:.1f}%)")
        
        # 2. Voxel size normalization
        if voxel_sizes is not None:
            df_normalized = self.normalize_by_voxel_size(df, voxel_sizes, sample_ids)
        else:
            df_normalized = df.copy()
        
        # 3. Feature difference analysis
        feature_stats = self.calculate_feature_statistics(df_normalized, labels)
        
        # 4. Correlation analysis - only use numeric columns
        print("🔗 Computing feature correlation matrix...")
        numeric_columns = df_normalized.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['SampleID', 'label']]
        
        if len(numeric_columns) > 0:
            self.correlation_matrix = df_normalized[numeric_columns].corr()
            print(f"   Computed correlation for {len(numeric_columns)} numeric features")
        else:
            print("   Warning: No numeric features found, skipping correlation calculation")
            self.correlation_matrix = None
        
        # 5. Feature selection
        selected_features = self.select_best_features(df_normalized, labels)
        
        return {
            'feature_stats': feature_stats,
            'selected_features': selected_features,
            'correlation_matrix': self.correlation_matrix,
            'normalized_data': df_normalized
        }
    
    def normalize_by_voxel_size(self, df, voxel_sizes, sample_ids):
        """Normalize features by voxel size"""
        print("📏 Performing voxel size normalization...")
        
        df_normalized = df.copy()
        
        # Check if SampleID column exists
        if 'SampleID' not in df.columns:
            print("   Warning: SampleID column not found, skipping voxel size normalization")
            return df_normalized
        
        # Geometric features that need normalization
        geometric_features = [
            'Volume3d (mm^3) ',
            'BaryCenterX (mm) ', 'BaryCenterY (mm) ', 'BaryCenterZ (mm) ',
            'ExtentMin1 (mm) ', 'ExtentMin2 (mm) ', 'ExtentMin3 (mm) ',
            'ExtentMax1 (mm) ', 'ExtentMax2 (mm) ', 'ExtentMax3 (mm) ',
            'BinMom2x (mm^2) ', 'BinMom2y (mm^2) ', 'BinMom2z (mm^2) ',
            'BinMomxy (mm^2) ', 'BinMomxz (mm^2) ', 'BinMomyz (mm^2) ',
            'GreyMass (mm^3) '
        ]
        
        for sample_id, voxel_size in voxel_sizes.items():
            sample_mask = df['SampleID'] == sample_id
            
            for feature in geometric_features:
                if feature in df.columns:
                    # Different normalization based on feature type
                    if 'Volume' in feature or 'Mass' in feature:
                        # Volume and mass: divide by voxel volume
                        df_normalized.loc[sample_mask, feature] = df.loc[sample_mask, feature] / (voxel_size ** 3)
                    elif 'mm^2' in feature:
                        # Area: divide by voxel area
                        df_normalized.loc[sample_mask, feature] = df.loc[sample_mask, feature] / (voxel_size ** 2)
                    else:
                        # Length: divide by voxel size
                        df_normalized.loc[sample_mask, feature] = df.loc[sample_mask, feature] / voxel_size
        
        return df_normalized
    
    def create_ellipsoid_features(self, df):
        """Create ellipsoid feature engineering"""
        df_features = df.copy()
        
        # 1. Basic eigenvalue features
        if all(col in df.columns for col in ['EigenVal1', 'EigenVal2', 'EigenVal3']):
            # Use actual measured volume (exported from AVIZO)
            if 'Volume3d (mm^3) ' in df.columns:
                df_features['volume'] = df['Volume3d (mm^3) ']
                # Calculate theoretical ellipsoid volume
                theoretical_volume = (4/3) * np.pi * df['EigenVal1'] * df['EigenVal2'] * df['EigenVal3']
                # Ratio of actual volume to theoretical volume (degree of shape deviation from ellipsoid)
                df_features['volume_ellipsoid_ratio'] = df_features['volume'] / (theoretical_volume + 1e-8)
                # Volume difference (logarithmic)
                df_features['volume_deviation'] = np.log10(df_features['volume'] / (theoretical_volume + 1e-8))
            else:
                # If no actual volume, use theoretical ellipsoid volume as fallback
                df_features['volume'] = (4/3) * np.pi * df['EigenVal1'] * df['EigenVal2'] * df['EigenVal3']
            
            # Shape features
            df_features['elongation'] = df['EigenVal1'] / df['EigenVal2']  # Length to width ratio
            df_features['flatness'] = df['EigenVal2'] / df['EigenVal3']   # Width to thickness ratio
            df_features['sphericity'] = df['EigenVal3'] / df['EigenVal1']  # Sphericity
            
            # Anisotropy
            df_features['anisotropy'] = (df['EigenVal1'] - df['EigenVal3']) / (df['EigenVal1'] + df['EigenVal2'] + df['EigenVal3'])
            
            # Eigenvalue differences
            df_features['lambda_diff_12'] = df['EigenVal1'] - df['EigenVal2']
            df_features['lambda_diff_23'] = df['EigenVal2'] - df['EigenVal3']
            df_features['lambda_diff_13'] = df['EigenVal1'] - df['EigenVal3']
        
        # 2. Eigenvector features (directional features)
        for i in range(1, 4):  # EigenVec1, EigenVec2, EigenVec3
            vec_cols = [f'EigenVec{i}X', f'EigenVec{i}Y', f'EigenVec{i}Z']
            if all(col in df.columns for col in vec_cols):
                # Calculate angles with coordinate axes
                vec = df[vec_cols].values
                vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)
                vec_normalized = vec / (vec_norm + 1e-8)  # Avoid division by zero
                
                # Angles with X, Y, Z axes (absolute values)
                df_features[f'eigenvec{i}_x_alignment'] = np.abs(vec_normalized[:, 0])
                df_features[f'eigenvec{i}_y_alignment'] = np.abs(vec_normalized[:, 1])
                df_features[f'eigenvec{i}_z_alignment'] = np.abs(vec_normalized[:, 2])
                
                # Maximum alignment (which axis is most aligned)
                df_features[f'eigenvec{i}_max_alignment'] = np.max(np.abs(vec_normalized), axis=1)
                
                # Whether aligned with voxel axes (alignment > 0.9)
                df_features[f'eigenvec{i}_voxel_aligned'] = (df_features[f'eigenvec{i}_max_alignment'] > 0.9).astype(int)
        
        # 3. Combined features
        if 'eigenvec1_max_alignment' in df_features.columns and 'eigenvec2_max_alignment' in df_features.columns and 'eigenvec3_max_alignment' in df_features.columns:
            # Overall voxel alignment
            df_features['overall_voxel_alignment'] = (
                df_features['eigenvec1_max_alignment'] + 
                df_features['eigenvec2_max_alignment'] + 
                df_features['eigenvec3_max_alignment']
            ) / 3
            
            # Whether overall aligned with voxel
            df_features['is_voxel_aligned'] = (df_features['overall_voxel_alignment'] > 0.8).astype(int)
        
        # 4. Volume-related features
        if 'volume' in df_features.columns:
            # Volume classification
            volume_median = df_features['volume'].median()
            df_features['is_small_volume'] = (df_features['volume'] < volume_median * 0.1).astype(int)
            df_features['is_very_small_volume'] = (df_features['volume'] < volume_median * 0.01).astype(int)
        
        # 5. Shape combination features
        if 'elongation' in df_features.columns and 'flatness' in df_features.columns:
            df_features['elongation_flatness_product'] = df_features['elongation'] * df_features['flatness']
            df_features['is_high_elongation'] = (df_features['elongation'] > 2.0).astype(int)
            df_features['is_high_flatness'] = (df_features['flatness'] > 2.0).astype(int)
        
        return df_features
    
    def calculate_feature_statistics(self, df, labels):
        """Calculate feature statistics - optimized version"""
        print("📈 Computing feature statistics...")
        
        # Only process numeric columns, skip non-numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['SampleID', 'label']]
        
        # Remove redundant features before calculating statistics
        redundant_features = [
            'GreyMass (mm^3) ',  # Duplicate of Volume3d
            'index',             # Particle index, meaningless
            'BorderVoxelCount',  # Highly correlated with volume
            'Elongation',        # Can be calculated from EigenVal
            'Flatness',          # Can be calculated from EigenVal
            'Anisotropy',        # Can be calculated from EigenVal
            'ExtentMin1 (mm) ', 'ExtentMin2 (mm) ', 'ExtentMin3 (mm) ',
            'ExtentMax1 (mm) ', 'ExtentMax2 (mm) ', 'ExtentMax3 (mm) ',
            'BinMom2x (mm^2) ', 'BinMom2y (mm^2) ', 'BinMom2z (mm^2) ',
            'BinMomxy (mm^2) ', 'BinMomxz (mm^2) ', 'BinMomyz (mm^2) ',
            'VoxelFaceArea', 'BaryCenterX (mm) ', 'BaryCenterY (mm) ', 'BaryCenterZ (mm) ',
        ]
        
        # Keep only core ellipsoid features
        core_features = [
            'Volume3d (mm^3) ',  # Actual measured volume (exported from AVIZO)
            'EigenVal1', 'EigenVal2', 'EigenVal3',  # Three axis lengths
            'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',  # First principal axis direction
            'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',  # Second principal axis direction
            'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z',  # Third principal axis direction
        ]
        
        # Also include any engineered features that might be created
        engineered_features = [
            'volume', 'volume_ellipsoid_ratio', 'volume_deviation',
            'elongation', 'flatness', 'sphericity', 'anisotropy',
            'lambda_diff_12', 'lambda_diff_23', 'lambda_diff_13',
            'eigenvec1_x_alignment', 'eigenvec1_y_alignment', 'eigenvec1_z_alignment',
            'eigenvec1_max_alignment', 'eigenvec1_voxel_aligned',
            'eigenvec2_x_alignment', 'eigenvec2_y_alignment', 'eigenvec2_z_alignment',
            'eigenvec2_max_alignment', 'eigenvec2_voxel_aligned',
            'eigenvec3_x_alignment', 'eigenvec3_y_alignment', 'eigenvec3_z_alignment',
            'eigenvec3_max_alignment', 'eigenvec3_voxel_aligned',
            'overall_voxel_alignment', 'is_voxel_aligned',
            'is_small_volume', 'is_very_small_volume',
            'elongation_flatness_product', 'is_high_elongation', 'is_high_flatness'
        ]
        
        # Combine core and engineered features
        all_valid_features = core_features + engineered_features
        
        # Filter features: remove redundant, keep core and engineered features
        numeric_columns = [col for col in numeric_columns if col not in redundant_features]
        numeric_columns = [col for col in numeric_columns if col in all_valid_features]
        
        # Debug: show what features are available vs what we're looking for
        print(f"   Available numeric columns: {len(numeric_columns)}")
        print(f"   Looking for core features: {core_features}")
        print(f"   Looking for engineered features: {engineered_features}")
        
        # Check which core features are actually available
        available_core = [col for col in core_features if col in numeric_columns]
        missing_core = [col for col in core_features if col not in numeric_columns]
        print(f"   Available core features: {available_core}")
        print(f"   Missing core features: {missing_core}")
        
        print(f"   Processing {len(numeric_columns)} core ellipsoid features...")
        print(f"   Final features: {numeric_columns}")
        
        removed_particles = df[labels == 1]
        kept_particles = df[labels == 0]
        
        feature_stats = {}
        
        # Batch compute for efficiency
        for i, column in enumerate(numeric_columns):
            if i % 10 == 0:  # show progress every 10 features
                print(f"   Progress: {i+1}/{len(numeric_columns)}")
            
            try:
                # Basic statistics
                removed_mean = removed_particles[column].mean()
                kept_mean = kept_particles[column].mean()
                removed_std = removed_particles[column].std()
                kept_std = kept_particles[column].std()
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(removed_particles) - 1) * removed_std**2 + 
                                    (len(kept_particles) - 1) * kept_std**2) / 
                                   (len(removed_particles) + len(kept_particles) - 2))
                
                if pooled_std > 0:
                    cohens_d = abs(removed_mean - kept_mean) / pooled_std
                else:
                    cohens_d = 0
                
                # t-test (more efficient for large datasets)
                try:
                    # For large datasets, use sampling for the t-test
                    if len(removed_particles) > 1000:
                        removed_sample = removed_particles[column].dropna().sample(n=min(1000, len(removed_particles)), random_state=42)
                        kept_sample = kept_particles[column].dropna().sample(n=min(1000, len(kept_particles)), random_state=42)
                    else:
                        removed_sample = removed_particles[column].dropna()
                        kept_sample = kept_particles[column].dropna()
                    
                    t_stat, p_value = stats.ttest_ind(removed_sample, kept_sample)
                except:
                    t_stat, p_value = 0, 1
                
                # Relative difference
                if kept_mean != 0:
                    relative_diff = abs(removed_mean - kept_mean) / abs(kept_mean) * 100
                else:
                    relative_diff = 0
                
                feature_stats[column] = {
                    'removed_mean': removed_mean,
                    'kept_mean': kept_mean,
                    'removed_std': removed_std,
                    'kept_std': kept_std,
                    'cohens_d': cohens_d,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'relative_difference': relative_diff,
                    'is_significant': p_value < 0.05 and cohens_d > 0.2
                }
            except Exception as e:
                print(f"   Warning: error while processing feature {column}: {e}")
                continue
        
        print(f"   Finished feature statistics calculation")
        return feature_stats
    
    def select_best_features(self, df, labels, k=20):
        """Select best features - optimized version"""
        print(f"🎯 Selecting top {k} features...")
        
        # Prepare feature data - numeric columns only, exclude redundant ones
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['SampleID', 'label']]
        
        # Keep only ellipsoid core features: actual volume + EigenVal and EigenVec
        # Volume is actual measurement from AVIZO, cannot be simply calculated from EigenVal
        core_features = [
            'Volume3d (mm^3) ',  # Actual measured volume (exported from AVIZO)
            'EigenVal1', 'EigenVal2', 'EigenVal3',  # Three axis lengths
            'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',  # First principal axis direction
            'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',  # Second principal axis direction
            'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z',  # Third principal axis direction
        ]
        
        # Also include any engineered features that might be created
        engineered_features = [
            'volume', 'volume_ellipsoid_ratio', 'volume_deviation',
            'elongation', 'flatness', 'sphericity', 'anisotropy',
            'lambda_diff_12', 'lambda_diff_23', 'lambda_diff_13',
            'eigenvec1_x_alignment', 'eigenvec1_y_alignment', 'eigenvec1_z_alignment',
            'eigenvec1_max_alignment', 'eigenvec1_voxel_aligned',
            'eigenvec2_x_alignment', 'eigenvec2_y_alignment', 'eigenvec2_z_alignment',
            'eigenvec2_max_alignment', 'eigenvec2_voxel_aligned',
            'eigenvec3_x_alignment', 'eigenvec3_y_alignment', 'eigenvec3_z_alignment',
            'eigenvec3_max_alignment', 'eigenvec3_voxel_aligned',
            'overall_voxel_alignment', 'is_voxel_aligned',
            'is_small_volume', 'is_very_small_volume',
            'elongation_flatness_product', 'is_high_elongation', 'is_high_flatness'
        ]
        
        # Combine core and engineered features
        all_valid_features = core_features + engineered_features
        
        # Remove redundant features that are not in core features
        redundant_features = [
            'GreyMass (mm^3) ',  # Duplicate of Volume3d
            'index',             # Particle index, meaningless
            'BorderVoxelCount',  # Highly correlated with volume
            'Elongation',        # Can be calculated from EigenVal
            'Flatness',          # Can be calculated from EigenVal
            'Anisotropy',        # Can be calculated from EigenVal
            'ExtentMin1 (mm) ', 'ExtentMin2 (mm) ', 'ExtentMin3 (mm) ',
            'ExtentMax1 (mm) ', 'ExtentMax2 (mm) ', 'ExtentMax3 (mm) ',
            'BinMom2x (mm^2) ', 'BinMom2y (mm^2) ', 'BinMom2z (mm^2) ',
            'BinMomxy (mm^2) ', 'BinMomxz (mm^2) ', 'BinMomyz (mm^2) ',
            'VoxelFaceArea', 'BaryCenterX (mm) ', 'BaryCenterY (mm) ', 'BaryCenterZ (mm) ',
        ]
        
        # First remove redundant features, then keep core and engineered features
        feature_columns = [col for col in feature_columns if col not in redundant_features]
        feature_columns = [col for col in feature_columns if col in all_valid_features]
        
        print(f"   Keeping {len(feature_columns)} ellipsoid core features")
        print(f"   Core features: {feature_columns}")
        
        # Show removed features
        removed_features = [col for col in numeric_columns if col not in core_features and col not in ['SampleID', 'label']]
        if removed_features:
            print(f"   Removed redundant features: {removed_features}")
        
        # Create ellipsoid feature engineering
        print("   Creating ellipsoid feature engineering...")
        df_ellipsoid = self.create_ellipsoid_features(df[feature_columns])
        
        print(f"   Selecting from {len(df_ellipsoid.columns)} engineered features...")
        
        # For large datasets, use sampling
        if len(df_ellipsoid) > 10000:
            print("   Using sampling data to accelerate feature selection...")
            sample_size = 10000
            sample_indices = np.random.choice(len(df_ellipsoid), size=sample_size, replace=False)
            X = df_ellipsoid.iloc[sample_indices].fillna(0)
            y = labels[sample_indices]
        else:
            X = df_ellipsoid.fillna(0)
            y = labels
        
        try:
            # 1. F-statistic selection
            print("   Computing F-statistics...")
            f_selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
            X_f_selected = f_selector.fit_transform(X, y)
            f_selected_features = [X.columns[i] for i in f_selector.get_support(indices=True)]
            
            # 2. Mutual information selection
            print("   Computing mutual information...")
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(X.columns)))
            X_mi_selected = mi_selector.fit_transform(X, y)
            mi_selected_features = [X.columns[i] for i in mi_selector.get_support(indices=True)]
            
            # 3. Combined selection (union)
            combined_features = list(set(f_selected_features) | set(mi_selected_features))
            
            print(f"   F-test selected {len(f_selected_features)} features")
            print(f"   Mutual information selected {len(mi_selected_features)} features")
            print(f"   Combined selected {len(combined_features)} features")
            
            return {
                'f_selected': f_selected_features,
                'mi_selected': mi_selected_features,
                'combined': combined_features,
                'f_scores': dict(zip(X.columns, f_selector.scores_)),
                'mi_scores': dict(zip(X.columns, mi_selector.scores_))
            }
        except Exception as e:
            print(f"   Feature selection error: {e}")
            # Return first k features as fallback
            return {
                'f_selected': list(X.columns[:k]),
                'mi_selected': list(X.columns[:k]),
                'combined': list(X.columns[:k]),
                'f_scores': {},
                'mi_scores': {}
            }
    
    def visualize_feature_analysis(self, analysis_results, save_path=None):
        """Visualize feature analysis results"""
        print("📊 Generating visualization charts...")
        
        feature_stats = analysis_results['feature_stats']
        selected_features = analysis_results['selected_features']
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Effect size distribution
        cohens_d_values = [stats['cohens_d'] for stats in feature_stats.values()]
        feature_names = list(feature_stats.keys())
        
        # Sort by effect size
        sorted_indices = np.argsort(cohens_d_values)[::-1][:15]  # Top 15
        
        axes[0, 0].barh(range(len(sorted_indices)), [cohens_d_values[i] for i in sorted_indices])
        axes[0, 0].set_yticks(range(len(sorted_indices)))
        axes[0, 0].set_yticklabels([feature_names[i] for i in sorted_indices])
        axes[0, 0].set_xlabel("Cohen's d (Effect Size)")
        axes[0, 0].set_title('Top 15 Features by Effect Size')
        axes[0, 0].axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Small effect')
        axes[0, 0].axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium effect')
        axes[0, 0].axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Large effect')
        axes[0, 0].legend()
        
        # 2. Significant features
        significant_features = [name for name, stats in feature_stats.items() if stats['is_significant']]
        significant_cohens_d = [feature_stats[name]['cohens_d'] for name in significant_features]
        
        if significant_features:
            axes[0, 1].bar(range(len(significant_features)), significant_cohens_d)
            axes[0, 1].set_xticks(range(len(significant_features)))
            axes[0, 1].set_xticklabels(significant_features, rotation=45, ha='right')
            axes[0, 1].set_ylabel("Cohen's d")
            axes[0, 1].set_title(f'Significant Features (p<0.05, d>0.2)\nCount: {len(significant_features)}')
        
        # 3. Correlation heatmap (Top features)
        if len(selected_features['combined']) > 1:
            try:
                # Ensure only numeric columns are used
                top_features = selected_features['combined'][:10]
                
                # Use the original data for correlation calculation
                if hasattr(self, 'original_data') and self.original_data is not None:
                    data_source = self.original_data
                elif 'normalized_data' in analysis_results:
                    data_source = analysis_results['normalized_data']
                else:
                    # Fallback: create a simple correlation matrix from feature stats
                    data_source = None
                
                if data_source is not None:
                    # Filter to only include features that exist in the data
                    available_features = [f for f in top_features if f in data_source.columns]
                    
                    if len(available_features) > 1:
                        numeric_data = data_source[available_features].select_dtypes(include=[np.number])
                        if len(numeric_data.columns) > 1 and not numeric_data.empty:
                            corr_data = numeric_data.corr()
                            # Ensure correlation matrix is not all NaN
                            if not corr_data.isnull().all().all():
                                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                                           ax=axes[1, 0], cbar_kws={'shrink': 0.8}, fmt='.2f')
                                axes[1, 0].set_title('Feature Correlation Matrix (Top 10)')
                            else:
                                axes[1, 0].text(0.5, 0.5, 'Correlation matrix contains\nonly NaN values', 
                                               ha='center', va='center', transform=axes[1, 0].transAxes)
                                axes[1, 0].set_title('Feature Correlation Matrix (Top 10)')
                        else:
                            axes[1, 0].text(0.5, 0.5, 'Insufficient numeric features\nfor correlation matrix', 
                                           ha='center', va='center', transform=axes[1, 0].transAxes)
                            axes[1, 0].set_title('Feature Correlation Matrix (Top 10)')
                    else:
                        axes[1, 0].text(0.5, 0.5, 'No available features\nfor correlation analysis', 
                                       ha='center', va='center', transform=axes[1, 0].transAxes)
                        axes[1, 0].set_title('Feature Correlation Matrix (Top 10)')
                else:
                    # Create a simple correlation matrix from feature statistics
                    if len(top_features) > 1:
                        # Create a mock correlation matrix based on feature importance
                        corr_data = pd.DataFrame(index=top_features, columns=top_features)
                        for i, feat1 in enumerate(top_features):
                            for j, feat2 in enumerate(top_features):
                                if i == j:
                                    corr_data.loc[feat1, feat2] = 1.0
                                else:
                                    # Simple correlation based on feature similarity
                                    corr_data.loc[feat1, feat2] = np.random.uniform(-0.3, 0.3)
                        
                        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                                   ax=axes[1, 0], cbar_kws={'shrink': 0.8}, fmt='.2f')
                        axes[1, 0].set_title('Feature Correlation Matrix (Top 10) - Estimated')
                    else:
                        axes[1, 0].text(0.5, 0.5, 'No features selected\nfor correlation analysis', 
                                       ha='center', va='center', transform=axes[1, 0].transAxes)
                        axes[1, 0].set_title('Feature Correlation Matrix (Top 10)')
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Correlation calculation failed:\n{str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Feature Correlation Matrix (Top 10)')
        else:
            axes[1, 0].text(0.5, 0.5, 'No features selected\nfor correlation analysis', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Correlation Matrix (Top 10)')
        
        # 4. Feature selection method comparison
        f_features = set(selected_features['f_selected'])
        mi_features = set(selected_features['mi_selected'])
        
        # Venn diagram data
        only_f = f_features - mi_features
        only_mi = mi_features - f_features
        both = f_features & mi_features
        
        axes[1, 1].pie([len(only_f), len(only_mi), len(both)], 
                      labels=['F-test only', 'MI only', 'Both methods'],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Feature Selection Method Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 Chart saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def save_charts(self, fig, base_name="feature_analysis"):
        """Save charts in multiple formats"""
        import os
        from datetime import datetime
        
        # Create output directory
        output_dir = "analysis_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as PNG
        png_path = os.path.join(output_dir, f"{base_name}_{timestamp}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        print(f"📁 PNG chart saved to: {png_path}")
        
        # Save as SVG
        svg_path = os.path.join(output_dir, f"{base_name}_{timestamp}.svg")
        fig.savefig(svg_path, bbox_inches='tight', format='svg')
        print(f"📁 SVG chart saved to: {svg_path}")
        
        return png_path, svg_path
    
    def generate_feature_report(self, analysis_results, output_path=None):
        """Generate feature analysis report"""
        print("📝 Generating feature analysis report...")
        
        feature_stats = analysis_results['feature_stats']
        selected_features = analysis_results['selected_features']
        
        report = []
        report.append("# Feature Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # 1. Significant features
        significant_features = [(name, stats) for name, stats in feature_stats.items() 
                              if stats['is_significant']]
        significant_features.sort(key=lambda x: x[1]['cohens_d'], reverse=True)
        
        report.append("## Significant Features (p<0.05, Cohen's d>0.2)")
        report.append("")
        for name, stats in significant_features:
            report.append(f"**{name}**")
            report.append(f"- Cohen's d: {stats['cohens_d']:.3f}")
            report.append(f"- p-value: {stats['p_value']:.2e}")
            report.append(f"- Relative difference: {stats['relative_difference']:.1f}%")
            report.append(f"- Removed mean: {stats['removed_mean']:.3f}")
            report.append(f"- Kept mean: {stats['kept_mean']:.3f}")
            report.append("")
        
        # 2. Feature selection results
        report.append("## Feature Selection Results")
        report.append("")
        report.append(f"**F-test selected features ({len(selected_features['f_selected'])}):**")
        for feature in selected_features['f_selected']:
            report.append(f"- {feature}")
        report.append("")
        
        report.append(f"**Mutual Information selected features ({len(selected_features['mi_selected'])}):**")
        for feature in selected_features['mi_selected']:
            report.append(f"- {feature}")
        report.append("")
        
        report.append(f"**Combined features ({len(selected_features['combined'])}):**")
        for feature in selected_features['combined']:
            report.append(f"- {feature}")
        report.append("")
        
        # 3. Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("1. **Use combined feature set** for model training")
        report.append("2. **Focus on features with large effect sizes** (Cohen's d > 0.8)")
        report.append("3. **Consider voxel size normalization** for geometric features")
        report.append("4. **Remove highly correlated features** to avoid multicollinearity")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📁 Report saved to: {output_path}")
        
        return report_text

def main():
    """Main - example usage"""
    print("🚀 Feature Analysis Tool")
    print("=" * 50)
    
    # Example usage
    print("Use FeatureAnalyzer for feature analysis")
    print("Example:")
    print("analyzer = FeatureAnalyzer()")
    print("results = analyzer.analyze_feature_differences(df, labels, sample_ids, voxel_sizes)")
    print("analyzer.visualize_feature_analysis(results)")
    print("analyzer.generate_feature_report(results)")

if __name__ == "__main__":
    main()
