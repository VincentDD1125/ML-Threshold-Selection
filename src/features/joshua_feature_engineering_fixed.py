#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正的Joshua特征工程模块 - 解决体素尺寸归一化和特征缩放问题
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class JoshuaFeatureEngineerFixed:
    """修正的Joshua特征工程师 - 解决体素尺寸和特征缩放问题"""
    
    def __init__(self, voxel_size_um: Optional[float] = None):
        """初始化特征工程师
        
        Args:
            voxel_size_um: 体素尺寸，单位微米（um），用于归一化（可选）
        """
        self.voxel_size_um = voxel_size_um
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_joshua_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """基于Joshua论文提取7个核心特征（修正版）
        
        Args:
            df: 包含颗粒数据的DataFrame
            fit_scaler: 是否拟合标准化器
            
        Returns:
            包含7个标准化特征的DataFrame
        """
        print("🔬 使用修正的Joshua论文方法提取特征...")
        
        # 1. 提取原始数据
        volume = df['Volume3d (mm^3) '].values
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
        
        print(f"   - 处理 {len(df)} 个颗粒")
        print(f"   - 原始体积范围: {volume.min():.2e} - {volume.max():.2e} mm³")
        
        # 2. 正确的体素尺寸归一化
        if self.voxel_size_um is not None:
            print(f"   - 应用体素尺寸归一化: {self.voxel_size_um} μm")
            # 转换为mm
            voxel_size_mm = self.voxel_size_um / 1000.0
            print(f"   - 体素尺寸: {voxel_size_mm:.6f} mm")
            # 归一化：计算颗粒由多少个体素组成（最少为1个体素，取上取整为整数体素数）
            volume_voxels = np.ceil(volume / (voxel_size_mm ** 3)).astype(np.int64)
            volume_voxels = np.maximum(volume_voxels, 1)
            volume_normalized = volume_voxels.astype(float)
            print(f"   - 归一化后体积（体素数）范围: {volume_normalized.min():.0f} - {volume_normalized.max():.0f}")
        else:
            # 没有体素尺寸就无法得到体素数，直接报错以避免产生<1体素的体积
            raise ValueError("Voxel size (μm) is required to normalize volume into voxel counts (>=1). Please input voxel sizes.")
        
        # 3. 计算半轴长度的对数
        a1, a2, a3 = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        l1, l2, l3 = np.log(a1), np.log(a2), np.log(a3)
        
        print(f"   - 半轴长度对数范围: l1={l1.min():.3f}-{l1.max():.3f}, l2={l2.min():.3f}-{l2.max():.3f}, l3={l3.min():.3f}-{l3.max():.3f}")
        
        # 4. 构建方向矩阵Q
        Q = np.stack([eigenvec1, eigenvec2, eigenvec3], axis=1)  # (n, 3, 3)
        
        # 验证特征向量是单位向量
        vec_norms = np.linalg.norm(Q, axis=2)
        if not np.allclose(vec_norms, 1.0, atol=1e-6):
            print(f"   ⚠️ 警告: 特征向量不是单位向量，进行归一化")
            Q = Q / vec_norms[:, :, np.newaxis]
        
        # 5. 构建对数-椭球张量L
        log_E_tilde = np.zeros((len(df), 3, 3))
        log_E_tilde[:, 0, 0] = -2 * l1
        log_E_tilde[:, 1, 1] = -2 * l2
        log_E_tilde[:, 2, 2] = -2 * l3
        
        # 计算 L = Q^T · (log Ẽ) · Q
        L = np.zeros((len(df), 3, 3))
        for i in range(len(df)):
            L[i] = Q[i].T @ log_E_tilde[i] @ Q[i]
        
        # 6. 转换为六维向量（保持几何不变性）
        l_vector = np.zeros((len(df), 6))
        l_vector[:, 0] = L[:, 0, 0]  # L₁₁
        l_vector[:, 1] = L[:, 1, 1]  # L₂₂
        l_vector[:, 2] = L[:, 2, 2]  # L₃₃
        l_vector[:, 3] = np.sqrt(2) * L[:, 0, 1]  # √2L₁₂
        l_vector[:, 4] = np.sqrt(2) * L[:, 0, 2]  # √2L₁₃
        l_vector[:, 5] = np.sqrt(2) * L[:, 1, 2]  # √2L₂₃
        
        # 7. 构建最终特征
        features = {}
        features['Volume'] = volume_normalized
        features['L11'] = l_vector[:, 0]
        features['L22'] = l_vector[:, 1]
        features['L33'] = l_vector[:, 2]
        features['sqrt2_L12'] = l_vector[:, 3]
        features['sqrt2_L13'] = l_vector[:, 4]
        features['sqrt2_L23'] = l_vector[:, 5]
        
        result_df = pd.DataFrame(features)
        
        print(f"   ✅ 成功提取7个特征: {list(result_df.columns)}")
        print(f"   - 特征统计（标准化前）:")
        for col in result_df.columns:
            print(f"     {col}: 均值={result_df[col].mean():.3f}, 标准差={result_df[col].std():.3f}, 范围=[{result_df[col].min():.3f}, {result_df[col].max():.3f}]")
        
        # 8. 关键修正：特征标准化
        print(f"   🔧 应用特征标准化...")
        if fit_scaler and not self.is_fitted:
            # 拟合标准化器
            result_df_scaled = pd.DataFrame(
                self.scaler.fit_transform(result_df),
                columns=result_df.columns,
                index=result_df.index
            )
            self.is_fitted = True
            print(f"   ✅ 标准化器已拟合")
        elif self.is_fitted:
            # 使用已拟合的标准化器
            result_df_scaled = pd.DataFrame(
                self.scaler.transform(result_df),
                columns=result_df.columns,
                index=result_df.index
            )
            print(f"   ✅ 使用已拟合的标准化器")
        else:
            result_df_scaled = result_df.copy()
            print(f"   ⚠️ 未应用标准化")
        
        print(f"   - 特征统计（标准化后）:")
        for col in result_df_scaled.columns:
            print(f"     {col}: 均值={result_df_scaled[col].mean():.3f}, 标准差={result_df_scaled[col].std():.3f}, 范围=[{result_df_scaled[col].min():.3f}, {result_df_scaled[col].max():.3f}]")
        
        return result_df_scaled
    
    def get_feature_names(self) -> list:
        """获取特征名称列表
        
        Returns:
            7个特征名称的列表
        """
        return ['Volume', 'L11', 'L22', 'L33', 'sqrt2_L12', 'sqrt2_L13', 'sqrt2_L23']
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """获取特征描述
        
        Returns:
            特征名称到描述的映射
        """
        return {
            'Volume': '归一化颗粒体积（体素数）',
            'L11': '对数-椭球张量对角线元素 L₁₁',
            'L22': '对数-椭球张量对角线元素 L₂₂', 
            'L33': '对数-椭球张量对角线元素 L₃₃',
            'sqrt2_L12': '对数-椭球张量非对角线元素 √2L₁₂',
            'sqrt2_L13': '对数-椭球张量非对角线元素 √2L₁₃',
            'sqrt2_L23': '对数-椭球张量非对角线元素 √2L₂₃'
        }
    
    def validate_features(self, features_df: pd.DataFrame) -> bool:
        """验证特征的有效性
        
        Args:
            features_df: 特征DataFrame
            
        Returns:
            是否有效
        """
        print("🔍 验证特征有效性...")
        
        # 检查特征数量
        if len(features_df.columns) != 7:
            print(f"   ❌ 特征数量错误: 期望7个，实际{len(features_df.columns)}个")
            return False
        
        # 检查特征名称
        expected_features = self.get_feature_names()
        if not all(col in expected_features for col in features_df.columns):
            print(f"   ❌ 特征名称错误: {list(features_df.columns)}")
            return False
        
        # 检查数值有效性
        if features_df.isnull().any().any():
            print(f"   ❌ 存在缺失值")
            return False
        
        if np.isinf(features_df.values).any():
            print(f"   ❌ 存在无穷值")
            return False
        
        # 检查体积特征（应为体素数，整数且>=1）
        if (features_df['Volume'] < 1).any():
            print(f"   ❌ 存在小于1个体素的体积")
            return False
        # 体积应为整数（允许微小数值误差）
        if not np.allclose(features_df['Volume'].values, np.round(features_df['Volume'].values), atol=1e-6):
            print(f"   ❌ 归一化体积非整数体素数")
            return False
        
        # 检查标准化效果
        means = features_df.mean()
        stds = features_df.std()
        if not np.allclose(means, 0, atol=0.1):
            print(f"   ⚠️ 特征均值未接近0: {means}")
        if not np.allclose(stds, 1, atol=0.1):
            print(f"   ⚠️ 特征标准差未接近1: {stds}")
        
        print("   ✅ 特征验证通过")
        return True


def main():
    """主函数 - 示例用法"""
    print("🚀 修正的Joshua特征工程模块")
    print("=" * 50)
    
    # 创建示例数据
    np.random.seed(42)
    n_particles = 1000
    
    # 模拟数据
    data = {
        'Volume3d (mm^3) ': np.random.lognormal(-12, 1, n_particles),
        'EigenVal1': np.random.lognormal(-6, 0.5, n_particles),
        'EigenVal2': np.random.lognormal(-6, 0.5, n_particles),
        'EigenVal3': np.random.lognormal(-6, 0.5, n_particles),
        'EigenVec1X': np.random.normal(0, 1, n_particles),
        'EigenVec1Y': np.random.normal(0, 1, n_particles),
        'EigenVec1Z': np.random.normal(0, 1, n_particles),
        'EigenVec2X': np.random.normal(0, 1, n_particles),
        'EigenVec2Y': np.random.normal(0, 1, n_particles),
        'EigenVec2Z': np.random.normal(0, 1, n_particles),
        'EigenVec3X': np.random.normal(0, 1, n_particles),
        'EigenVec3Y': np.random.normal(0, 1, n_particles),
        'EigenVec3Z': np.random.normal(0, 1, n_particles),
    }
    
    df = pd.DataFrame(data)
    
    # 归一化特征向量
    for i in range(1, 4):
        vec = df[[f'EigenVec{i}X', f'EigenVec{i}Y', f'EigenVec{i}Z']].values
        vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec_normalized = vec / vec_norm
        df[f'EigenVec{i}X'] = vec_normalized[:, 0]
        df[f'EigenVec{i}Y'] = vec_normalized[:, 1]
        df[f'EigenVec{i}Z'] = vec_normalized[:, 2]
    
    # 创建修正的特征工程师
    engineer = JoshuaFeatureEngineerFixed(voxel_size_um=50)  # 50微米分辨率
    
    # 提取特征
    features = engineer.extract_joshua_features(df)
    
    # 验证特征
    engineer.validate_features(features)
    
    print("\n✅ 修正的Joshua特征工程完成!")
    print(f"最终特征形状: {features.shape}")
    print(f"特征列: {list(features.columns)}")


if __name__ == "__main__":
    main()
