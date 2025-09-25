#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joshua论文特征工程模块 - 基于对数-椭球张量的先进特征提取方法
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class JoshuaFeatureEngineer:
    """基于Joshua论文的对数-椭球张量特征工程"""
    
    def __init__(self, voxel_size_mm: Optional[float] = None):
        """初始化特征工程师
        
        Args:
            voxel_size_mm: 体素尺寸，用于归一化（可选）
        """
        self.voxel_size_mm = voxel_size_mm
        
    def extract_joshua_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于Joshua论文提取7个核心特征
        
        Args:
            df: 包含颗粒数据的DataFrame
            
        Returns:
            包含7个特征的DataFrame
        """
        print("🔬 使用Joshua论文方法提取特征...")
        
        # 1. 提取原始数据
        volume = df['Volume3d (mm^3) '].values
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
        
        print(f"   - 处理 {len(df)} 个颗粒")
        print(f"   - 体积范围: {volume.min():.2e} - {volume.max():.2e} mm³")
        
        # 2. 计算半轴长度的对数
        a1, a2, a3 = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        l1, l2, l3 = np.log(a1), np.log(a2), np.log(a3)
        
        print(f"   - 半轴长度对数范围: l1={l1.min():.3f}-{l1.max():.3f}, l2={l2.min():.3f}-{l2.max():.3f}, l3={l3.min():.3f}-{l3.max():.3f}")
        
        # 3. 构建方向矩阵Q
        Q = np.stack([eigenvec1, eigenvec2, eigenvec3], axis=1)  # (n, 3, 3)
        
        # 验证特征向量是单位向量
        vec_norms = np.linalg.norm(Q, axis=2)
        if not np.allclose(vec_norms, 1.0, atol=1e-6):
            print(f"   ⚠️ 警告: 特征向量不是单位向量，进行归一化")
            Q = Q / vec_norms[:, :, np.newaxis]
        
        # 4. 构建对数-椭球张量L
        log_E_tilde = np.zeros((len(df), 3, 3))
        log_E_tilde[:, 0, 0] = -2 * l1
        log_E_tilde[:, 1, 1] = -2 * l2
        log_E_tilde[:, 2, 2] = -2 * l3
        
        # 计算 L = Q^T · (log Ẽ) · Q
        L = np.zeros((len(df), 3, 3))
        for i in range(len(df)):
            L[i] = Q[i].T @ log_E_tilde[i] @ Q[i]
        
        # 5. 转换为六维向量（保持几何不变性）
        l_vector = np.zeros((len(df), 6))
        l_vector[:, 0] = L[:, 0, 0]  # L₁₁
        l_vector[:, 1] = L[:, 1, 1]  # L₂₂
        l_vector[:, 2] = L[:, 2, 2]  # L₃₃
        l_vector[:, 3] = np.sqrt(2) * L[:, 0, 1]  # √2L₁₂
        l_vector[:, 4] = np.sqrt(2) * L[:, 0, 2]  # √2L₁₃
        l_vector[:, 5] = np.sqrt(2) * L[:, 1, 2]  # √2L₂₃
        
        # 6. 构建最终特征
        features = {}
        features['Volume'] = volume
        features['L11'] = l_vector[:, 0]
        features['L22'] = l_vector[:, 1]
        features['L33'] = l_vector[:, 2]
        features['sqrt2_L12'] = l_vector[:, 3]
        features['sqrt2_L13'] = l_vector[:, 4]
        features['sqrt2_L23'] = l_vector[:, 5]
        
        # 7. 体素尺寸归一化（如果提供）
        if self.voxel_size_mm is not None:
            print(f"   - 应用体素尺寸归一化: {self.voxel_size_mm} mm")
            features['Volume'] = features['Volume'] / (self.voxel_size_mm ** 3)
        
        result_df = pd.DataFrame(features)
        
        print(f"   ✅ 成功提取7个特征: {list(result_df.columns)}")
        print(f"   - 特征统计:")
        for col in result_df.columns:
            print(f"     {col}: 均值={result_df[col].mean():.3f}, 标准差={result_df[col].std():.3f}")
        
        return result_df
    
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
            'Volume': '颗粒体积 (mm³)',
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
        
        # 检查体积特征
        if (features_df['Volume'] <= 0).any():
            print(f"   ❌ 存在非正体积值")
            return False
        
        print("   ✅ 特征验证通过")
        return True
    
    def analyze_feature_distribution(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """分析特征分布
        
        Args:
            features_df: 特征DataFrame
            labels: 标签数组 (0=正常, 1=伪影)
            
        Returns:
            分析结果字典
        """
        print("📊 分析特征分布...")
        
        normal_mask = labels == 0
        artifact_mask = labels == 1
        
        analysis = {}
        
        for col in features_df.columns:
            normal_values = features_df[col][normal_mask]
            artifact_values = features_df[col][artifact_mask]
            
            analysis[col] = {
                'normal_mean': normal_values.mean(),
                'normal_std': normal_values.std(),
                'artifact_mean': artifact_values.mean(),
                'artifact_std': artifact_values.std(),
                'effect_size': abs(normal_values.mean() - artifact_values.mean()) / 
                              np.sqrt((normal_values.var() + artifact_values.var()) / 2),
                'normal_range': (normal_values.min(), normal_values.max()),
                'artifact_range': (artifact_values.min(), artifact_values.max())
            }
        
        # 打印分析结果
        print("   📈 特征分布分析:")
        for col, stats in analysis.items():
            print(f"     {col}:")
            print(f"       正常颗粒: 均值={stats['normal_mean']:.3f}, 标准差={stats['normal_std']:.3f}")
            print(f"       伪影颗粒: 均值={stats['artifact_mean']:.3f}, 标准差={stats['artifact_std']:.3f}")
            print(f"       效应量: {stats['effect_size']:.3f}")
        
        return analysis


def main():
    """主函数 - 示例用法"""
    print("🚀 Joshua特征工程模块")
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
    
    # 创建特征工程师
    engineer = JoshuaFeatureEngineer(voxel_size_mm=0.03)
    
    # 提取特征
    features = engineer.extract_joshua_features(df)
    
    # 验证特征
    engineer.validate_features(features)
    
    # 模拟标签
    labels = np.random.choice([0, 1], size=n_particles, p=[0.8, 0.2])
    
    # 分析特征分布
    engineer.analyze_feature_distribution(features, labels)
    
    print("\n✅ Joshua特征工程完成!")
    print(f"最终特征形状: {features.shape}")
    print(f"特征列: {list(features.columns)}")


if __name__ == "__main__":
    main()
