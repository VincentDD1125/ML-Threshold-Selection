#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI layout builder extracted from main.
"""

import tkinter as tk
from tkinter import ttk


def build_main_ui(app):
    main_frame = ttk.Frame(app.root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    title_label = ttk.Label(main_frame, text="ML Threshold Selection System - Enhanced", font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    status_text = "Status: "
    status_text += "LightGBM ✅ | " if app.LIGHTGBM_AVAILABLE else "LightGBM ❌ | "
    status_text += "Full Modules ✅" if app.FULL_MODULES_AVAILABLE else "Full Modules ❌"
    status_label = ttk.Label(main_frame, text=status_text, font=("Arial", 10))
    status_label.pack(pady=5)

    # First row: Steps 1-3
    button_frame1 = ttk.Frame(main_frame)
    button_frame1.pack(fill=tk.X, pady=5)
    ttk.Button(button_frame1, text="1. Load Training Data", command=app.load_multiple_training_data, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame1, text="2. Input Expert Thresholds", command=app.input_expert_thresholds, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame1, text="3. Input Voxel Sizes", command=app.input_voxel_sizes, width=20).pack(side=tk.LEFT, padx=3)

    # Second row: Steps 4-6
    button_frame2 = ttk.Frame(main_frame)
    button_frame2.pack(fill=tk.X, pady=5)
    ttk.Button(button_frame2, text="4. Feature Analysis", command=app.analyze_ellipsoid_features, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame2, text="5. Train Model", command=app.train_model, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame2, text="6. Load Test Data", command=app.load_test_data, width=20).pack(side=tk.LEFT, padx=3)

    # Third row: Step 7 and Visualizations
    button_frame3 = ttk.Frame(main_frame)
    button_frame3.pack(fill=tk.X, pady=5)
    ttk.Button(button_frame3, text="7. Predict Analysis", command=app.predict_analysis, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame3, text="📊 Training Visualization", command=app.show_training_visualization, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame3, text="📈 Prediction Visualization", command=app.show_prediction_visualization, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame3, text="🔄 Load Last Time Model", command=app.load_last_time_model, width=20).pack(side=tk.LEFT, padx=3)

    # Fourth row: Additional Tools
    button_frame4 = ttk.Frame(main_frame)
    button_frame4.pack(fill=tk.X, pady=5)
    ttk.Button(button_frame4, text="📦 Fabric Boxplots", command=app.generate_fabric_boxplots, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame4, text="📤 Export Results", command=app.export_results, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame4, text="⚙️ Config Threshold", command=app.configure_strict_threshold, width=20).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_frame4, text="❓ User Guide", command=app.open_user_guide, width=20).pack(side=tk.LEFT, padx=3)

    app.status_label = ttk.Label(main_frame, text="Waiting for operation...", font=("Arial", 12))
    app.status_label.pack(pady=10)

    app.results_text = tk.Text(main_frame, height=25, width=120, font=("Consolas", 10))
    app.results_text.pack(fill=tk.BOTH, expand=True, pady=10)

    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=app.results_text.yview)
    app.results_text.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
