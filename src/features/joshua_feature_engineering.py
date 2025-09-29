#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering Module - Advanced feature extraction based on log-ellipsoid tensor
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class JoshuaFeatureEngineer:
    """Log-ellipsoid tensor feature engineering (7 core features)"""
    
    def __init__(self, voxel_size_mm: Optional[float] = None):
        """Initialize feature engineer
        
        Args:
            voxel_size_mm: Voxel size for normalization (optional)
        """
        self.voxel_size_mm = voxel_size_mm

    def extract_joshua_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract 7 core features based on log-ellipsoid tensor
        
        Args:
            df: DataFrame containing particle data
            
        Returns:
            DataFrame containing 7 features
        """
        print("🔬 Extracting features using log-ellipsoid tensor method...")
        
        # 1. Extract raw data
        volume = df['Volume3d (mm^3) '].values
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
        
        print(f"   - Processing {len(df)} particles")
        print(f"   - Volume range: {volume.min():.2e} - {volume.max():.2e} mm³")
        
        # 2. Calculate log of semi-axis lengths
        a1, a2, a3 = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        l1, l2, l3 = np.log(a1), np.log(a2), np.log(a3)
        
        print(f"   - Semi-axis log range: l1={l1.min():.3f}-{l1.max():.3f}, l2={l2.min():.3f}-{l2.max():.3f}, l3={l3.min():.3f}-{l3.max():.3f}")
        
        # 3. Build orientation matrix Q
        Q = np.stack([eigenvec1, eigenvec2, eigenvec3], axis=1)  # (n, 3, 3)
        
        # Verify eigenvectors are unit vectors
        vec_norms = np.linalg.norm(Q, axis=2)
        if not np.allclose(vec_norms, 1.0, atol=1e-6):
            print(f"   ⚠️ Warning: Eigenvectors are not unit vectors, normalizing")
            Q = Q / vec_norms[:, :, np.newaxis]
        
        # 4. Build log-ellipsoid tensor L
        log_E_tilde = np.zeros((len(df), 3, 3))
        log_E_tilde[:, 0, 0] = l1
        log_E_tilde[:, 1, 1] = l2
        log_E_tilde[:, 2, 2] = l3
        
        # Calculate L = Q^T · (log Ẽ) · Q
        L = np.zeros((len(df), 3, 3))
        for i in range(len(df)):
            try:
                L[i] = Q[i].T @ log_E_tilde[i] @ Q[i]
                # Check for NaN or inf values
                if np.isnan(L[i]).any() or np.isinf(L[i]).any():
                    print(f"   ⚠️ Warning: NaN/Inf detected for particle {i}, using zeros")
                    L[i] = np.zeros((3, 3))
            except Exception as e:
                print(f"   ⚠️ Warning: Matrix multiplication failed for particle {i}: {e}")
                L[i] = np.zeros((3, 3))
        
        # 5. Convert to 6D vector (maintaining geometric invariance)
        l_vector = np.zeros((len(df), 6))
        l_vector[:, 0] = L[:, 0, 0]  # L11
        l_vector[:, 1] = L[:, 1, 1]  # L22
        l_vector[:, 2] = L[:, 2, 2]  # L33
        l_vector[:, 3] = np.sqrt(2) * L[:, 0, 1]  # sqrt(2) * L12
        l_vector[:, 4] = np.sqrt(2) * L[:, 0, 2]  # sqrt(2) * L13
        l_vector[:, 5] = np.sqrt(2) * L[:, 1, 2]  # sqrt(2) * L23
        
        # 6. Build final features
        features = {}
        features['Volume'] = volume
        features['L11'] = l_vector[:, 0]
        features['L22'] = l_vector[:, 1]
        features['L33'] = l_vector[:, 2]
        features['sqrt2_L12'] = l_vector[:, 3]
        features['sqrt2_L13'] = l_vector[:, 4]
        features['sqrt2_L23'] = l_vector[:, 5]
        
        # 7. Voxel size normalization (if provided)
        if self.voxel_size_mm is not None:
            print(f"   - Applying voxel size normalization: {self.voxel_size_mm} mm")
            features['Volume'] = features['Volume'] / (self.voxel_size_mm ** 3)
        
        result_df = pd.DataFrame(features, index=df.index)
        
        print(f"   ✅ Successfully extracted 7 features: {list(result_df.columns)}")
        print(f"   - Feature statistics:")
        for col in result_df.columns:
            print(f"     {col}: mean={result_df[col].mean():.3f}, std={result_df[col].std():.3f}")
        
        return result_df

    def get_feature_names(self) -> list:
        """Get feature name list
        
        Returns:
            List of 7 feature names
        """
        return ['Volume', 'L11', 'L22', 'L33', 'sqrt2_L12', 'sqrt2_L13', 'sqrt2_L23']

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get feature descriptions
        
        Returns:
            Mapping from feature names to descriptions
        """
        return {
            'Volume': 'Particle volume (mm³)',
            'L11': 'Log-ellipsoid tensor diagonal element L₁₁',
            'L22': 'Log-ellipsoid tensor diagonal element L₂₂', 
            'L33': 'Log-ellipsoid tensor diagonal element L₃₃',
            'sqrt2_L12': 'Log-ellipsoid tensor off-diagonal element √2L₁₂',
            'sqrt2_L13': 'Log-ellipsoid tensor off-diagonal element √2L₁₃',
            'sqrt2_L23': 'Log-ellipsoid tensor off-diagonal element √2L₂₃'
        }

    def validate_features(self, features_df: pd.DataFrame) -> bool:
        """Validate feature validity
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            Whether valid
        """
        print("🔍 Validating feature validity...")
        
        # Check feature count
        if len(features_df.columns) != 7:
            print(f"   ❌ Wrong feature count: expected 7, got {len(features_df.columns)}")
            return False
        
        # Check feature names
        expected_features = self.get_feature_names()
        if not all(col in expected_features for col in features_df.columns):
            print(f"   ❌ Wrong feature names: {list(features_df.columns)}")
            return False
        
        # Check numerical validity
        if features_df.isnull().any().any():
            print(f"   ❌ Missing values found")
            return False
        
        if np.isinf(features_df.values).any():
            print(f"   ❌ Infinite values found")
            return False
        
        # Check volume feature
        if (features_df['Volume'] <= 0).any():
            print(f"   ❌ Non-positive volume values found")
            return False
        
        print("   ✅ Feature validation passed")
        return True

    def analyze_feature_distribution(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze feature distribution
        
        Args:
            features_df: Feature DataFrame
            labels: Label array (0=normal, 1=artifact)
            
        Returns:
            Analysis results dictionary
        """
        print("📊 Analyzing feature distribution...")
        
        normal_mask = labels == 0
        artifact_mask = labels == 1
        
        analysis = {}
        for col in features_df.columns:
            normal_data = features_df[col][normal_mask]
            artifact_data = features_df[col][artifact_mask]
            
            normal_mean = normal_data.mean()
            normal_std = normal_data.std()
            artifact_mean = artifact_data.mean()
            artifact_std = artifact_data.std()
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(normal_data) - 1) * normal_std**2 + 
                                (len(artifact_data) - 1) * artifact_std**2) / 
                               (len(normal_data) + len(artifact_data) - 2))
            
            if pooled_std > 0:
                effect_size = (artifact_mean - normal_mean) / pooled_std
            else:
                effect_size = 0
            
            analysis[col] = {
                'normal_mean': normal_mean,
                'normal_std': normal_std,
                'artifact_mean': artifact_mean,
                'artifact_std': artifact_std,
                'effect_size': effect_size
            }
        
        # Print analysis results
        print("   📈 Feature distribution analysis:")
        for col, stats in analysis.items():
            print(f"     {col}:")
            print(f"       Normal particles: mean={stats['normal_mean']:.3f}, std={stats['normal_std']:.3f}")
            print(f"       Artifact particles: mean={stats['artifact_mean']:.3f}, std={stats['artifact_std']:.3f}")
            print(f"       Effect size: {stats['effect_size']:.3f}")
        
        return analysis


def main():
    """Main function - example usage"""
    print("🚀 Feature Engineering Module")
    print("=" * 50)
    
    # Create example data
    np.random.seed(42)
    n_particles = 1000
    
    # Simulate data
    data = {
        'Volume3d (mm^3) ': np.random.lognormal(0, 1, n_particles),
        'EigenVal1': np.random.lognormal(0, 0.5, n_particles),
        'EigenVal2': np.random.lognormal(0, 0.5, n_particles),
        'EigenVal3': np.random.lognormal(0, 0.5, n_particles),
    }
    
    # Add eigenvector components
    for i in range(1, 4):
        for axis in ['X', 'Y', 'Z']:
            data[f'EigenVec{i}{axis}'] = np.random.normal(0, 1, n_particles)
    
    # Normalize eigenvectors
    for i in range(1, 4):
        vec = np.array([data[f'EigenVec{i}{axis}'] for axis in ['X', 'Y', 'Z']]).T
        norms = np.linalg.norm(vec, axis=1)
        for j, axis in enumerate(['X', 'Y', 'Z']):
            data[f'EigenVec{i}{axis}'] = vec[:, j] / norms
    
    df = pd.DataFrame(data)
    
    # Create feature engineer
    engineer = JoshuaFeatureEngineer(voxel_size_mm=0.03)
    
    # Extract features
    features = engineer.extract_joshua_features(df)
    
    # Validate features
    engineer.validate_features(features)
    
    # Simulate labels
    labels = np.random.choice([0, 1], size=n_particles, p=[0.8, 0.2])
    
    # Analyze feature distribution
    engineer.analyze_feature_distribution(features, labels)
    
    print("\n✅ Feature engineering completed!")
    print(f"Final feature shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")


if __name__ == "__main__":
    main()
