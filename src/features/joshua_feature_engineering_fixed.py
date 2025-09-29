#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Feature Engineering Module - Resolves voxel size normalization and feature scaling issues
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class JoshuaFeatureEngineerFixed:
    """Fixed feature engineer - resolves voxel size and feature scaling issues"""
    
    def __init__(self, voxel_size_um: Optional[float] = None):
        """Initialize feature engineer
        
        Args:
            voxel_size_um: Voxel size in micrometers (μm) for normalization (optional)
        """
        self.voxel_size_um = voxel_size_um
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_joshua_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Extract 7 core features based on log-ellipsoid tensor (fixed version)
        
        Args:
            df: DataFrame containing particle data
            fit_scaler: Whether to fit the scaler
            
        Returns:
            DataFrame containing 7 standardized features
        """
        print("🔬 Extracting features using fixed log-ellipsoid tensor method...")
        
        # 1. Extract raw data
        volume = df['Volume3d (mm^3) '].values
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
        
        print(f"   - Processing {len(df)} particles")
        print(f"   - Original volume range: {volume.min():.2e} - {volume.max():.2e} mm³")
        
        # 2. Correct voxel size normalization
        if self.voxel_size_um is not None:
            print(f"   - Applying voxel size normalization: {self.voxel_size_um} μm")
            # Convert to mm
            voxel_size_mm = self.voxel_size_um / 1000.0
            print(f"   - Voxel size: {voxel_size_mm:.6f} mm")
            # Normalize: calculate how many voxels the particle consists of (minimum 1 voxel, ceiling to integer voxel count)
            volume_voxels = np.ceil(volume / (voxel_size_mm ** 3)).astype(np.int64)
            # Ensure minimum 1 voxel
            volume_voxels = np.maximum(volume_voxels, 1)
            volume_normalized = volume_voxels.astype(float)
            print(f"   - Normalized volume (voxel count) range: {volume_normalized.min():.0f} - {volume_normalized.max():.0f}")
        else:
            # Without voxel size, cannot get voxel count, directly error to avoid producing <1 voxel volume
            raise ValueError("Voxel size (μm) is required to normalize volume into voxel counts (>=1). Please input voxel sizes.")
        
        # 3. Calculate log of semi-axis lengths
        a1, a2, a3 = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        l1, l2, l3 = np.log(a1), np.log(a2), np.log(a3)
        
        print(f"   - Semi-axis log range: l1={l1.min():.3f}-{l1.max():.3f}, l2={l2.min():.3f}-{l2.max():.3f}, l3={l3.min():.3f}-{l3.max():.3f}")
        
        # 4. Build orientation matrix Q
        Q = np.stack([eigenvec1, eigenvec2, eigenvec3], axis=1)  # (n, 3, 3)
        
        # Verify eigenvectors are unit vectors
        vec_norms = np.linalg.norm(Q, axis=2)
        if not np.allclose(vec_norms, 1.0, atol=1e-6):
            print(f"   ⚠️ Warning: Eigenvectors are not unit vectors, normalizing")
            Q = Q / vec_norms[:, :, np.newaxis]
        
        # 5. Build log-ellipsoid tensor L
        log_E_tilde = np.zeros((len(df), 3, 3))
        log_E_tilde[:, 0, 0] = l1
        log_E_tilde[:, 1, 1] = l2
        log_E_tilde[:, 2, 2] = l3
        
        # Calculate L = Q^T · (log Ẽ) · Q
        L = np.zeros((len(df), 3, 3))
        for i in range(len(df)):
            L[i] = Q[i].T @ log_E_tilde[i] @ Q[i]
        
        # 6. Convert to 6D vector (maintaining geometric invariance)
        l_vector = np.zeros((len(df), 6))
        l_vector[:, 0] = L[:, 0, 0]  # L11
        l_vector[:, 1] = L[:, 1, 1]  # L22
        l_vector[:, 2] = L[:, 2, 2]  # L33
        l_vector[:, 3] = np.sqrt(2) * L[:, 0, 1]  # sqrt(2) * L12
        l_vector[:, 4] = np.sqrt(2) * L[:, 0, 2]  # sqrt(2) * L13
        l_vector[:, 5] = np.sqrt(2) * L[:, 1, 2]  # sqrt(2) * L23
        
        # 7. Build final features
        features = {}
        features['Volume'] = volume_normalized
        features['L11'] = l_vector[:, 0]
        features['L22'] = l_vector[:, 1]
        features['L33'] = l_vector[:, 2]
        features['sqrt2_L12'] = l_vector[:, 3]
        features['sqrt2_L13'] = l_vector[:, 4]
        features['sqrt2_L23'] = l_vector[:, 5]
        
        result_df = pd.DataFrame(features, index=df.index)
        
        print(f"   ✅ Successfully extracted 7 features: {list(result_df.columns)}")
        print(f"   - Feature statistics (before scaling):")
        for col in result_df.columns:
            print(f"     {col}: mean={result_df[col].mean():.3f}, std={result_df[col].std():.3f}, range=[{result_df[col].min():.3f}, {result_df[col].max():.3f}]")
        
        # 8. Key fix: Feature scaling
        print(f"   🔧 Applying feature scaling...")
        if fit_scaler and not self.is_fitted:
            # Fit scaler
            result_df_scaled = pd.DataFrame(
                self.scaler.fit_transform(result_df),
                columns=result_df.columns,
                index=result_df.index
            )
            self.is_fitted = True
            print(f"   ✅ Scaler fitted")
        elif self.is_fitted:
            # Use fitted scaler
            result_df_scaled = pd.DataFrame(
                self.scaler.transform(result_df),
                columns=result_df.columns,
                index=result_df.index
            )
            print(f"   ✅ Using fitted scaler")
        else:
            result_df_scaled = result_df.copy()
            print(f"   ⚠️ No scaling applied")
        
        print(f"   - Feature statistics (after scaling):")
        for col in result_df_scaled.columns:
            print(f"     {col}: mean={result_df_scaled[col].mean():.3f}, std={result_df_scaled[col].std():.3f}, range=[{result_df_scaled[col].min():.3f}, {result_df_scaled[col].max():.3f}]")
        
        return result_df_scaled

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
            'Volume': 'Normalized particle volume (voxel count)',
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
        
        # Check volume feature (should be voxel count, integer and >=1)
        if (features_df['Volume'] < 1).any():
            print(f"   ❌ Volume less than 1 voxel found")
            return False
        # Volume should be integer (allow small numerical errors)
        if not np.allclose(features_df['Volume'].values, np.round(features_df['Volume'].values), atol=1e-6):
            print(f"   ❌ Normalized volume is not integer voxel count")
            return False
        
        # Check scaling effect
        means = features_df.mean()
        stds = features_df.std()
        if not np.allclose(means, 0, atol=0.1):
            print(f"   ⚠️ Feature means not close to 0: {means}")
        if not np.allclose(stds, 1, atol=0.1):
            print(f"   ⚠️ Feature stds not close to 1: {stds}")
        
        print("   ✅ Feature validation passed")
        return True


def main():
    """Main function - example usage"""
    print("🚀 Fixed Feature Engineering Module")
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
    
    # Create fixed feature engineer
    engineer = JoshuaFeatureEngineerFixed(voxel_size_um=50)  # 50 micrometer resolution
    
    # Extract features
    features = engineer.extract_joshua_features(df)
    
    # Validate features
    engineer.validate_features(features)
    
    print("\n✅ Fixed feature engineering completed!")
    print(f"Final feature shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")


if __name__ == "__main__":
    main()
