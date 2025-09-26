#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joshua Feature Analysis Tool - Feature difference analysis based on log-ellipsoid tensor
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

from src.features.joshua_feature_engineering import JoshuaFeatureEngineer


class JoshuaFeatureAnalyzer:
    """Feature analyzer based on Joshua paper methodology"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_engineer = JoshuaFeatureEngineer()
        self.feature_importance_scores = {}
        self.correlation_matrix = None
        
    def analyze_feature_differences(self, df, labels, sample_ids=None, voxel_sizes=None):
        """分析特征差异 - 使用Joshua方法"""
        
        print("🔬 使用Joshua论文方法进行特征分析...")
        
        # 存储原始数据
        self.original_data = df.copy()
        
        # 1. 基本统计
        removed_particles = df[labels == 1]  # 移除的颗粒（伪影）
        kept_particles = df[labels == 0]     # 保留的颗粒（有效）
        
        print(f"📊 数据摘要:")
        print(f"   - 总颗粒数: {len(df)}")
        print(f"   - 移除颗粒: {len(removed_particles)} ({len(removed_particles)/len(df)*100:.1f}%)")
        print(f"   - 保留颗粒: {len(kept_particles)} ({len(kept_particles)/len(df)*100:.1f}%)")
        
        # 2. 体素尺寸归一化（如果提供）
        if voxel_sizes is not None:
            print("📏 应用体素尺寸归一化...")
            # 为每个样本设置体素尺寸
            for sample_id, voxel_size in voxel_sizes.items():
                sample_mask = df['SampleID'] == sample_id
                if sample_mask.any():
                    self.feature_engineer.voxel_size_mm = voxel_size
                    print(f"   - 样本 {sample_id}: {voxel_size} mm")
        
        # 3. 使用Joshua方法提取特征
        print("🔬 提取Joshua特征...")
        features_df = self.feature_engineer.extract_joshua_features(df)
        
        # 4. 验证特征
        if not self.feature_engineer.validate_features(features_df):
            raise ValueError("特征验证失败")
        
        # 5. 特征差异分析
        feature_stats = self.calculate_joshua_feature_statistics(features_df, labels)
        
        # 6. 相关性分析
        print("🔗 计算特征相关性矩阵...")
        self.correlation_matrix = features_df.corr()
        print(f"   - 计算了{len(features_df.columns)}个特征的相关性")
        
        # 7. 特征选择（对于Joshua方法，所有7个特征都是核心特征）
        selected_features = {
            'joshua_features': list(features_df.columns),
            'all_features': list(features_df.columns),
            'combined': list(features_df.columns)
        }
        
        return {
            'feature_stats': feature_stats,
            'selected_features': selected_features,
            'correlation_matrix': self.correlation_matrix,
            'joshua_features': features_df,
            'feature_engineer': self.feature_engineer
        }
    
    def calculate_joshua_feature_statistics(self, features_df, labels):
        """计算Joshua特征的统计信息"""
        print("📈 计算Joshua特征统计信息...")
        
        removed_particles = features_df[labels == 1]
        kept_particles = features_df[labels == 0]
        
        feature_stats = {}
        
        for column in features_df.columns:
            try:
                # 基本统计
                removed_mean = removed_particles[column].mean()
                kept_mean = kept_particles[column].mean()
                removed_std = removed_particles[column].std()
                kept_std = kept_particles[column].std()
                
                # 效应量（Cohen's d）
                pooled_std = np.sqrt(((len(removed_particles) - 1) * removed_std**2 + 
                                    (len(kept_particles) - 1) * kept_std**2) / 
                                   (len(removed_particles) + len(kept_particles) - 2))
                
                if pooled_std > 0:
                    cohens_d = abs(removed_mean - kept_mean) / pooled_std
                else:
                    cohens_d = 0
                
                # t检验
                try:
                    if len(removed_particles) > 1000:
                        removed_sample = removed_particles[column].dropna().sample(n=min(1000, len(removed_particles)), random_state=42)
                        kept_sample = kept_particles[column].dropna().sample(n=min(1000, len(kept_particles)), random_state=42)
                    else:
                        removed_sample = removed_particles[column].dropna()
                        kept_sample = kept_particles[column].dropna()
                    
                    t_stat, p_value = stats.ttest_ind(removed_sample, kept_sample)
                except:
                    t_stat, p_value = 0, 1
                
                # 相对差异
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
                
                print(f"   - {column}: d={cohens_d:.3f}, p={p_value:.2e}, 显著={feature_stats[column]['is_significant']}")
                
            except Exception as e:
                print(f"   ⚠️ 处理特征 {column} 时出错: {e}")
                continue
        
        print(f"   ✅ 完成{len(feature_stats)}个特征的统计分析")
        return feature_stats
    
    def visualize_joshua_feature_analysis(self, analysis_results, save_path=None):
        """可视化Joshua特征分析结果"""
        print("📊 生成Joshua特征分析可视化...")
        
        feature_stats = analysis_results['feature_stats']
        features_df = analysis_results['joshua_features']
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Joshua Feature Analysis Results (7 Core Features)', fontsize=16, fontweight='bold')
        
        # 1. 效应量分布
        cohens_d_values = [stats['cohens_d'] for stats in feature_stats.values()]
        feature_names = list(feature_stats.keys())
        
        # 按效应量排序
        sorted_indices = np.argsort(cohens_d_values)[::-1]
        
        bars = axes[0, 0].bar(range(len(feature_names)), [cohens_d_values[i] for i in sorted_indices])
        axes[0, 0].set_xticks(range(len(feature_names)))
        axes[0, 0].set_xticklabels([feature_names[i] for i in sorted_indices], rotation=45, ha='right')
        axes[0, 0].set_ylabel("Cohen's d (Effect Size)")
        axes[0, 0].set_title('Feature Effect Sizes (Joshua Method)')
        axes[0, 0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Small effect')
        axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium effect')
        axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Large effect')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 在柱状图上显示数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 显著特征
        significant_features = [name for name, stats in feature_stats.items() if stats['is_significant']]
        significant_cohens_d = [feature_stats[name]['cohens_d'] for name in significant_features]
        
        if significant_features:
            bars = axes[0, 1].bar(range(len(significant_features)), significant_cohens_d)
            axes[0, 1].set_xticks(range(len(significant_features)))
            axes[0, 1].set_xticklabels(significant_features, rotation=45, ha='right')
            axes[0, 1].set_ylabel("Cohen's d")
            axes[0, 1].set_title(f'Significant Features (p<0.05, d>0.2)\nCount: {len(significant_features)}')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 显示数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[0, 1].text(0.5, 0.5, 'No significant features found', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Significant Features')
        
        # 3. 相关性热图
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 1:
            sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 0], cbar_kws={'shrink': 0.8}, fmt='.2f')
            axes[1, 0].set_title('Joshua Features Correlation Matrix')
        else:
            axes[1, 0].text(0.5, 0.5, 'No correlation data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Correlation Matrix')
        
        # 4. 特征重要性总结
        feature_importance = [(name, stats['cohens_d']) for name, stats in feature_stats.items()]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        names, importance = zip(*feature_importance)
        bars = axes[1, 1].barh(range(len(names)), importance)
        axes[1, 1].set_yticks(range(len(names)))
        axes[1, 1].set_yticklabels(names)
        axes[1, 1].set_xlabel("Cohen's d (Feature Importance)")
        axes[1, 1].set_title('Joshua Features Importance Ranking')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 显示数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1, 1].text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                           f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 图表已保存到: {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_joshua_feature_report(self, analysis_results, output_path=None):
        """生成Joshua特征分析报告"""
        print("📝 生成Joshua特征分析报告...")
        
        feature_stats = analysis_results['feature_stats']
        feature_engineer = analysis_results['feature_engineer']
        
        report = []
        report.append("# Joshua Feature Analysis Report")
        report.append("=" * 60)
        report.append("")
        report.append("## 方法概述")
        report.append("本分析基于Joshua论文的对数-椭球张量方法，使用7个核心特征：")
        report.append("")
        
        # 特征描述
        feature_descriptions = feature_engineer.get_feature_descriptions()
        for feature, description in feature_descriptions.items():
            report.append(f"- **{feature}**: {description}")
        report.append("")
        
        # 显著特征
        significant_features = [(name, stats) for name, stats in feature_stats.items() 
                              if stats['is_significant']]
        significant_features.sort(key=lambda x: x[1]['cohens_d'], reverse=True)
        
        report.append("## 显著特征分析 (p<0.05, Cohen's d>0.2)")
        report.append("")
        if significant_features:
            for name, stats in significant_features:
                report.append(f"### {name}")
                report.append(f"- **效应量 (Cohen's d)**: {stats['cohens_d']:.3f}")
                report.append(f"- **p值**: {stats['p_value']:.2e}")
                report.append(f"- **相对差异**: {stats['relative_difference']:.1f}%")
                report.append(f"- **移除颗粒均值**: {stats['removed_mean']:.3f}")
                report.append(f"- **保留颗粒均值**: {stats['kept_mean']:.3f}")
                report.append("")
        else:
            report.append("未发现统计显著的特征差异。")
            report.append("")
        
        # 所有特征统计
        report.append("## 所有特征统计")
        report.append("")
        report.append("| 特征 | 效应量 | p值 | 相对差异(%) | 显著 |")
        report.append("|------|--------|-----|-------------|------|")
        
        for name, stats in feature_stats.items():
            significance = "是" if stats['is_significant'] else "否"
            report.append(f"| {name} | {stats['cohens_d']:.3f} | {stats['p_value']:.2e} | {stats['relative_difference']:.1f} | {significance} |")
        
        report.append("")
        
        # 方法优势
        report.append("## Joshua方法优势")
        report.append("")
        report.append("1. **数学严谨性**: 基于椭球体的内在几何结构")
        report.append("2. **特征紧凑性**: 仅用7个特征完整描述椭球体")
        report.append("3. **几何不变性**: 保持Frobenius范数不变")
        report.append("4. **物理意义明确**: 每个特征都有明确的几何解释")
        report.append("5. **避免冗余**: 消除了特征间的多重共线性")
        report.append("")
        
        # 建议
        report.append("## 建议")
        report.append("")
        report.append("1. **使用所有7个特征**: 每个特征都包含独特的几何信息")
        report.append("2. **关注高效应量特征**: 优先考虑Cohen's d > 0.5的特征")
        report.append("3. **验证几何不变性**: 确保特征在不同样本间的一致性")
        report.append("4. **监控模型性能**: 与传统方法对比验证效果")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📁 报告已保存到: {output_path}")
        
        return report_text


def main():
    """主函数 - 示例用法"""
    print("🚀 Joshua特征分析工具")
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
    
    # 模拟标签
    labels = np.random.choice([0, 1], size=n_particles, p=[0.8, 0.2])
    
    # 创建分析器
    analyzer = JoshuaFeatureAnalyzer()
    
    # 执行分析
    results = analyzer.analyze_feature_differences(df, labels)
    
    # 生成可视化
    analyzer.visualize_joshua_feature_analysis(results)
    
    # 生成报告
    analyzer.generate_joshua_feature_report(results)
    
    print("\n✅ Joshua特征分析完成!")


if __name__ == "__main__":
    main()
