#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised learning demo script
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_threshold_selection import SupervisedThresholdLearner, FeatureEngineer
import pandas as pd
import numpy as np

def main():
    """Run supervised learning demo"""
    print("ML Threshold Selection - Supervised Learning Demo")
    print("=" * 50)
    
    # Load sample data
    data_path = Path(__file__).parent.parent / "data" / "sample_particles.csv"
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} particles")
    print(f"Artifact ratio: {df['label'].mean():.1%}")
    
    # Extract features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.extract_all_features(df)
    
    print(f"Extracted {len(features.columns)} features")
    
    # Train model
    learner = SupervisedThresholdLearner()
    results = learner.train(features, df['label'].values)
    
    print(f"Training completed:")
    print(f"  - Features: {results['n_features']}")
    print(f"  - Samples: {results['n_samples']}")
    print(f"  - Train AUC: {results['train_auc']:.3f}")
    
    # Analyze sample
    analysis = learner.analyze_sample(df)
    
    print(f"Sample analysis:")
    print(f"  - Optimal threshold: {analysis['threshold']:.2e} mm³")
    print(f"  - Uncertainty: ±{analysis['uncertainty']:.2e}")
    print(f"  - Particles retained: {analysis['n_retained']}/{analysis['n_total']}")
    print(f"  - Removal rate: {analysis['removal_rate']:.1f}%")
    print(f"  - Mean artifact probability: {analysis['mean_probability']:.3f}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
