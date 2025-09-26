#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joshua Feature Analysis Tool - Feature difference analysis based on log-ellipsoid tensor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
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
        self.original_data = None

    def analyze_feature_differences(self, df, labels, sample_ids=None, voxel_sizes=None):
        """Analyze feature differences using Joshua method"""
        
        print("🔬 Performing feature analysis using Joshua paper method...")
        
        # Store original data
        self.original_data = df.copy()
        
        # 1. Basic statistics
        removed_particles = df[labels == 1]  # Removed particles (artifacts)
        kept_particles = df[labels == 0]     # Kept particles (valid)
        
        print(f"📊 Data summary:")
        print(f"   - Total particles: {len(df)}")
        print(f"   - Removed particles: {len(removed_particles)} ({len(removed_particles)/len(df)*100:.1f}%)")
        print(f"   - Kept particles: {len(kept_particles)} ({len(kept_particles)/len(df)*100:.1f}%)")
        
        # 2. Voxel size normalization (if provided)
        if voxel_sizes is not None:
            print("📏 Applying voxel size normalization...")
            # Set voxel size for each sample
            for sample_id, voxel_size in voxel_sizes.items():
                sample_mask = df['SampleID'] == sample_id
                if sample_mask.any():
                    self.feature_engineer.voxel_size_mm = voxel_size
                    print(f"   - Sample {sample_id}: {voxel_size} mm")
        
        # 3. Extract features using Joshua method
        print("🔬 Extracting Joshua features...")
        features_df = self.feature_engineer.extract_joshua_features(df)
        
        # 4. Validate features
        if not self.feature_engineer.validate_features(features_df):
            raise ValueError("Feature validation failed")
        
        # 5. Feature difference analysis
        feature_stats = self.calculate_joshua_feature_statistics(features_df, labels)
        
        # 6. Correlation analysis
        print("🔗 Computing feature correlation matrix...")
        self.correlation_matrix = features_df.corr()
        print(f"   - Computed correlations for {len(features_df.columns)} features")
        
        # 7. Feature selection (for Joshua method, all 7 features are core features)
        selected_features = {
            'joshua_selected': list(features_df.columns),
            'significant_features': [name for name, stats in feature_stats.items() if stats['is_significant']],
            'high_effect_features': [name for name, stats in feature_stats.items() if stats['cohens_d'] > 0.5]
        }
        
        # 8. Store results
        results = {
            'feature_stats': feature_stats,
            'selected_features': selected_features,
            'correlation_matrix': self.correlation_matrix,
            'features_df': features_df,
            'removed_particles': removed_particles,
            'kept_particles': kept_particles
        }
        
        print("✅ Joshua feature analysis completed!")
        return results

    def calculate_joshua_feature_statistics(self, features_df, labels):
        """Calculate Joshua feature statistics"""
        print("📈 Computing Joshua feature statistics...")
        
        removed_particles = features_df[labels == 1]
        kept_particles = features_df[labels == 0]
        
        feature_stats = {}
        
        for column in features_df.columns:
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
                    cohens_d = (removed_mean - kept_mean) / pooled_std
                else:
                    cohens_d = 0
                
                # t-test
                try:
                    t_stat, p_value = stats.ttest_ind(removed_particles[column], 
                                                    kept_particles[column])
                except:
                    p_value = 1.0
                    t_stat = 0
                
                # Relative difference
                if kept_mean != 0:
                    relative_difference = abs(removed_mean - kept_mean) / abs(kept_mean) * 100
                else:
                    relative_difference = 0
                
                # Store statistics
                feature_stats[column] = {
                    'removed_mean': removed_mean,
                    'kept_mean': kept_mean,
                    'removed_std': removed_std,
                    'kept_std': kept_std,
                    'cohens_d': cohens_d,
                    'p_value': p_value,
                    'relative_difference': relative_difference,
                    'is_significant': p_value < 0.05 and abs(cohens_d) > 0.2,
                    't_statistic': t_stat
                }
                
                print(f"   - {column}: d={cohens_d:.3f}, p={p_value:.2e}, significant={feature_stats[column]['is_significant']}")
                
            except Exception as e:
                print(f"   ⚠️ Error processing feature {column}: {e}")
                continue
        
        print(f"   ✅ Completed statistical analysis for {len(feature_stats)} features")
        return feature_stats

    def visualize_joshua_feature_analysis(self, analysis_results, save_path=None):
        """Visualize Joshua feature analysis results"""
        print("📊 Generating Joshua feature analysis visualization...")
        
        feature_stats = analysis_results['feature_stats']
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Joshua Feature Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Effect size distribution
        cohens_d_values = [stats['cohens_d'] for stats in feature_stats.values()]
        feature_names = list(feature_stats.keys())
        
        # Sort by effect size
        sorted_indices = np.argsort(cohens_d_values)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_cohens_d = [cohens_d_values[i] for i in sorted_indices]
        
        bars = axes[0, 0].bar(range(len(sorted_names)), sorted_cohens_d, 
                             color=['red' if d > 0.5 else 'orange' if d > 0.2 else 'lightblue' 
                                   for d in sorted_cohens_d])
        axes[0, 0].set_title('Effect Size (Cohen\'s d) by Feature')
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Cohen\'s d')
        axes[0, 0].set_xticks(range(len(sorted_names)))
        axes[0, 0].set_xticklabels(sorted_names, rotation=45, ha='right')
        
        # Display values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Significant features
        significant_features = [name for name, stats in feature_stats.items() if stats['is_significant']]
        significant_cohens_d = [feature_stats[name]['cohens_d'] for name in significant_features]
        
        if significant_features:
            bars = axes[0, 1].bar(range(len(significant_features)), significant_cohens_d, 
                                 color='red', alpha=0.7)
            axes[0, 1].set_title(f'Significant Features (n={len(significant_features)})')
            axes[0, 1].set_xlabel('Features')
            axes[0, 1].set_ylabel('Cohen\'s d')
            axes[0, 1].set_xticks(range(len(significant_features)))
            axes[0, 1].set_xticklabels(significant_features, rotation=45, ha='right')
            
            # Display values
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[0, 1].text(0.5, 0.5, 'No significant features', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Significant Features')
        
        # 3. Correlation heatmap
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 1:
            im = axes[1, 0].imshow(self.correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title('Feature Correlation Matrix')
            axes[1, 0].set_xticks(range(len(self.correlation_matrix.columns)))
            axes[1, 0].set_yticks(range(len(self.correlation_matrix.index)))
            axes[1, 0].set_xticklabels(self.correlation_matrix.columns, rotation=45, ha='right')
            axes[1, 0].set_yticklabels(self.correlation_matrix.index)
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Feature importance summary
        feature_importance = [(name, stats['cohens_d']) for name, stats in feature_stats.items()]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        names, importance = zip(*feature_importance)
        bars = axes[1, 1].bar(range(len(names)), importance, 
                             color=['red' if abs(d) > 0.5 else 'orange' if abs(d) > 0.2 else 'lightblue' 
                                   for d in importance])
        axes[1, 1].set_title('Feature Importance (|Cohen\'s d|)')
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('|Cohen\'s d|')
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        
        # Display values
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 Chart saved to: {save_path}")
        
        plt.show()
        return fig

    def generate_joshua_feature_report(self, analysis_results, output_path=None):
        """Generate Joshua feature analysis report"""
        print("📝 Generating Joshua feature analysis report...")
        
        feature_stats = analysis_results['feature_stats']
        feature_engineer = self.feature_engineer
        
        # Generate report
        report = []
        report.append("# Joshua Feature Analysis Report")
        report.append("")
        report.append("## Method Overview")
        report.append("This analysis is based on the Joshua paper's log-ellipsoid tensor method, using 7 core features:")
        report.append("")
        
        # Feature descriptions
        feature_descriptions = feature_engineer.get_feature_descriptions()
        for feature, description in feature_descriptions.items():
            report.append(f"- **{feature}**: {description}")
        
        report.append("")
        
        # Significant features
        significant_features = [(name, stats) for name, stats in feature_stats.items() 
                              if stats['is_significant']]
        
        if significant_features:
            report.append("## Significant Feature Analysis (p<0.05, Cohen's d>0.2)")
            report.append("")
            
            for name, stats in significant_features:
                report.append(f"### {name}")
                report.append(f"- **Effect Size (Cohen's d)**: {stats['cohens_d']:.3f}")
                report.append(f"- **p-value**: {stats['p_value']:.2e}")
                report.append(f"- **Relative Difference**: {stats['relative_difference']:.1f}%")
                report.append(f"- **Removed Particles Mean**: {stats['removed_mean']:.3f}")
                report.append(f"- **Kept Particles Mean**: {stats['kept_mean']:.3f}")
                report.append("")
        else:
            report.append("No statistically significant feature differences found.")
            report.append("")
        
        # All feature statistics
        report.append("## All Feature Statistics")
        report.append("")
        report.append("| Feature | Effect Size | p-value | Relative Diff(%) | Significant |")
        report.append("|---------|-------------|---------|------------------|-------------|")
        
        for name, stats in feature_stats.items():
            significance = "Yes" if stats['is_significant'] else "No"
            report.append(f"| {name} | {stats['cohens_d']:.3f} | {stats['p_value']:.2e} | {stats['relative_difference']:.1f} | {significance} |")
        
        report.append("")
        
        # Method advantages
        report.append("## Joshua Method Advantages")
        report.append("")
        report.append("1. **Mathematical Rigor**: Based on intrinsic geometric structure of ellipsoids")
        report.append("2. **Feature Compactness**: Complete ellipsoid description with only 7 features")
        report.append("3. **Geometric Invariance**: Maintains Frobenius norm invariance")
        report.append("4. **Clear Physical Meaning**: Each feature has clear geometric interpretation")
        report.append("5. **Avoids Redundancy**: Eliminates multicollinearity between features")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("1. **Use all 7 features**: Each feature contains unique geometric information")
        report.append("2. **Focus on high effect size features**: Prioritize features with Cohen's d > 0.5")
        report.append("3. **Verify geometric invariance**: Ensure feature consistency across samples")
        report.append("4. **Monitor model performance**: Compare with traditional methods for validation")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📁 Report saved to: {output_path}")
        
        return report_text


def main():
    """Main function - example usage"""
    print("🚀 Joshua Feature Analysis Tool")
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
    
    # Simulate labels
    labels = np.random.choice([0, 1], size=n_particles, p=[0.8, 0.2])
    
    # Create analyzer
    analyzer = JoshuaFeatureAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze_feature_differences(df, labels)
    
    # Generate visualization
    analyzer.visualize_joshua_feature_analysis(results)
    
    # Generate report
    analyzer.generate_joshua_feature_report(results)
    
    print("\n✅ Joshua feature analysis completed!")


if __name__ == "__main__":
    main()
