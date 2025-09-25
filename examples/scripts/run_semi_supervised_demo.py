#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-supervised learning demo script
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_threshold_selection import SemiSupervisedThresholdLearner
import pandas as pd
import numpy as np

def main():
    """Run semi-supervised learning demo"""
    print("ML Threshold Selection - Semi-supervised Learning Demo")
    print("=" * 55)
    
    # Load expert thresholds
    thresholds_path = Path(__file__).parent.parent / "data" / "expert_thresholds.csv"
    thresholds_df = pd.read_csv(thresholds_path)
    
    print(f"Loaded {len(thresholds_df)} expert thresholds")
    
    # Initialize learner
    learner = SemiSupervisedThresholdLearner()
    
    # Add expert thresholds
    for _, row in thresholds_df.iterrows():
        learner.add_expert_threshold(
            sample_id=row['sample_id'],
            threshold=row['threshold'],
            confidence=row['confidence'],
            notes=row['notes']
        )
    
    # Load sample data (simulate multiple samples)
    data_path = Path(__file__).parent.parent / "data" / "sample_particles.csv"
    df = pd.read_csv(data_path)
    
    # Simulate multiple samples
    for sample_id in thresholds_df['sample_id']:
        # Create a copy of the data for each sample
        sample_df = df.copy()
        sample_df['SampleID'] = sample_id
        learner.load_sample_data(sample_id, data_path)
    
    print(f"Loaded data for {len(thresholds_df)} samples")
    
    # Train model
    results = learner.train(method='threshold_based', model_type='lightgbm')
    
    print(f"Training completed:")
    print(f"  - Features: {results['n_features']}")
    print(f"  - Samples: {results['n_samples']}")
    print(f"  - Train Score: {results['train_score']:.3f}")
    
    # Analyze first sample
    sample_id = thresholds_df['sample_id'].iloc[0]
    analysis = learner.analyze_sample(df)
    
    print(f"Sample analysis for {sample_id}:")
    print(f"  - Optimal threshold: {analysis['threshold']:.2e} mm³")
    print(f"  - Uncertainty: ±{analysis['uncertainty']:.2e}")
    print(f"  - Particles retained: {analysis['n_retained']}/{analysis['n_total']}")
    print(f"  - Removal rate: {analysis['removal_rate']:.1f}%")
    print(f"  - Mean artifact probability: {analysis['mean_probability']:.3f}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
