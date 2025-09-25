#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data validation example script
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_threshold_selection import validate_data_file, DataValidator
import pandas as pd

def main():
    """Run data validation example"""
    print("ML Threshold Selection - Data Validation Example")
    print("=" * 50)
    
    # Example 1: Validate sample data
    sample_data_path = Path(__file__).parent.parent / "data" / "sample_particles.csv"
    
    if sample_data_path.exists():
        print(f"\n1. Validating sample data: {sample_data_path}")
        validate_data_file(str(sample_data_path), 'supervised')
    else:
        print(f"\n1. Sample data not found: {sample_data_path}")
    
    # Example 2: Validate with custom data
    print(f"\n2. Interactive validation")
    print("Enter path to your CSV file (or press Enter to skip):")
    user_path = input().strip()
    
    if user_path and Path(user_path).exists():
        validate_data_file(user_path, 'both')
    else:
        print("Skipping custom validation")
    
    # Example 3: Show expected column format
    print(f"\n3. Expected column format:")
    print("Required columns (with trailing spaces for units):")
    required_cols = [
        'Volume3d (mm^3) ',
        'EigenVal1', 'EigenVal2', 'EigenVal3',
        'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',
        'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',
        'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z'
    ]
    for col in required_cols:
        print(f"  - {col}")
    
    print("\nOptional columns:")
    optional_cols = [
        'BaryCenterX (mm) ', 'BaryCenterY (mm) ', 'BaryCenterZ (mm) ',
        'Anisotropy', 'Elongation', 'Flatness',
        'ExtentMin1 (mm) ', 'ExtentMax1 (mm) ',
        'ExtentMin2 (mm) ', 'ExtentMax2 (mm) ',
        'ExtentMin3 (mm) ', 'ExtentMax3 (mm) ',
        'BinMom2x (mm^2) ', 'BinMom2y (mm^2) ', 'BinMom2z (mm^2) ',
        'BinMomxy (mm^2) ', 'BinMomxz (mm^2) ', 'BinMomyz (mm^2) ',
        'VoxelFaceArea ', 'BorderVoxelCount ', 'GreyMass (mm^3) ',
        'indexMaterials', 'SampleID'
    ]
    for col in optional_cols:
        print(f"  - {col}")
    
    print("\nFor supervised learning, also include:")
    print("  - label (0=normal particle, 1=artifact)")
    
    print("\nValidation completed!")

if __name__ == "__main__":
    main()
