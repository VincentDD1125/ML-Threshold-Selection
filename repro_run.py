#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import pandas as pd

from src.ml_threshold_selection.training_pipeline import train_model_pipeline
from src.ml_threshold_selection.labeling import generate_labels_from_thresholds
from src.ml_threshold_selection.prediction_analysis import compute_dual_thresholds
from src.ml_threshold_selection.export_results import export_filtered_results, export_threshold_report
from src.features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', required=True, help='Directory with training CSV/XLSX files')
    ap.add_argument('--config', required=True, help='CSV with columns: SampleID,ExpertThreshold_mm3,VoxelSize_mm')
    ap.add_argument('--test_file', required=True, help='Test CSV/XLSX file')
    ap.add_argument('--voxel_size_mm', type=float, default=None, help='Override test voxel size (mm)')
    args = ap.parse_args()

    os.makedirs('outputs', exist_ok=True)

    # Load training data
    train_files = glob.glob(os.path.join(args.train_dir, '*.xlsx')) + glob.glob(os.path.join(args.train_dir, '*.xls')) + glob.glob(os.path.join(args.train_dir, '*.csv'))
    if not train_files:
        raise FileNotFoundError('No training files found in --train_dir')
    dfs = []
    for fp in train_files:
        ext = os.path.splitext(fp)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(fp)
        else:
            df = pd.read_excel(fp)
        if 'SampleID' not in df.columns:
            df['SampleID'] = os.path.splitext(os.path.basename(fp))[0]
        dfs.append(df)
    training_data = pd.concat(dfs, ignore_index=True)

    # Load config (thresholds & voxels)
    cfg = pd.read_csv(args.config)
    expert_thresholds = {r['SampleID']: float(r['ExpertThreshold_mm3']) for _, r in cfg.iterrows()}
    voxel_sizes = {r['SampleID']: float(r['VoxelSize_mm']) for _, r in cfg.iterrows()}
    sample_list = sorted(cfg['SampleID'].astype(str).tolist())

    # Generate labels
    training_data = generate_labels_from_thresholds(training_data, expert_thresholds, voxel_sizes, sample_list)

    # Train
    engineer = ResolutionAwareFeatureEngineer()
    model, features, training_results = train_model_pipeline(training_data, voxel_sizes, engineer, lightgbm_available=True)

    # Test
    ext = os.path.splitext(args.test_file)[1].lower()
    test_df = pd.read_csv(args.test_file) if ext == '.csv' else pd.read_excel(args.test_file)

    # Determine test voxel size
    test_sample_id = os.path.splitext(os.path.basename(args.test_file))[0]
    if args.voxel_size_mm is not None:
        voxel_mm = float(args.voxel_size_mm)
    else:
        voxel_mm = float(voxel_sizes.get(test_sample_id, list(voxel_sizes.values())[0]))

    test_features = engineer.extract(test_df, voxel_size_mm=voxel_mm, fit_scaler=False)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(test_features.values)[:, 1]
    else:
        probabilities = model.predict(test_features.values)

    volumes = test_df['Volume3d (mm^3) '].values
    voxel_vol = voxel_mm ** 3
    voxels_cont = volumes / voxel_vol

    loose, strict = compute_dual_thresholds(voxels_cont, probabilities)

    # Export
    loose_file, strict_file = export_filtered_results(
        results_df=test_df,
        probabilities=probabilities,
        loose_threshold_vox=int(loose) if loose is not None else 0,
        strict_threshold_vox=int(strict) if strict is not None else 0,
        voxel_size_mm=voxel_mm,
        outputs_dir='outputs',
    )
    loose_mm = (int(loose) if loose is not None else 0) * voxel_vol
    strict_mm = (int(strict) if strict is not None else 0) * voxel_vol
    loose_kept = int((test_df['Volume3d (mm^3) '] >= loose_mm).sum())
    strict_kept = int((test_df['Volume3d (mm^3) '] >= strict_mm).sum())

    report_filename = os.path.join('outputs', f'Threshold_Report_{int(loose) if loose else 0}vox_{int(strict) if strict else 0}vox.txt')
    export_threshold_report(
        out_path=report_filename,
        total_rows=len(test_df),
        voxel_size_mm=voxel_mm,
        loose_threshold_vox=int(loose) if loose else 0,
        strict_threshold_vox=int(strict) if strict else 0,
        loose_threshold_mm=loose_mm,
        strict_threshold_mm=strict_mm,
        loose_kept=loose_kept,
        strict_kept=strict_kept,
    )

    print('✅ Repro run completed.')
    print('Outputs in ./outputs')


if __name__ == '__main__':
    main()
