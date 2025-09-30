# 用户指南：模型训练与特征分析（GUI 版）

本指南面向最终用户，帮助你用图形界面完成数据导入、模型训练、预测分析、特征分析与导出结果。无需了解内部代码结构。

## 1. 准备数据
- 必要列：`Volume3d (mm^3) `、`EigenVal1/2/3`、`EigenVec1X/Y/Z`、`EigenVec2X/Y/Z`、`EigenVec3X/Y/Z`
- 建议列：`SampleID`（多样本流程更方便）
- 文件格式：CSV / XLSX

## 2. 启动程序
```bash
python main.py
```
进入主界面后，按从左到右的按钮顺序进行。

## 3. 训练流程（Training）
1) `1. Load Training Data`：选择一个或多个训练文件。
2) `2. Input Expert Thresholds`：为每个样本输入专家阈值（单位：mm³）。
3) `3. Input Voxel Sizes`：为每个样本输入体素边长（单位：mm）。
4) （可选）`4. Feature Analysis`：先进行特征差异分析，查看哪类特征区分度更高。
5) `5. Train Model`：开始训练。默认优先使用 LightGBM，若不可用则回退到 RandomForest。
   - 训练完成后会显示 AUC、Accuracy、Precision、Recall、F1 等指标。
   - 点击 `📊 Training Visualization` 可查看训练可视化（ROC/重要特征/分布/指标）。
6) 训练结束会自动保存会话到 `outputs/last_time_model.pkl`，便于下次一键恢复。

## 4. 预测与双阈值分析（Predict + Dual Thresholds）
1) `6. Load Test Data`：加载待评估的样本文件。
2) （弹窗）输入该样本的体素边长（mm）。
3) `7. Predict Analysis`：执行预测与双阈值分析：
   - **Loose（拐点阈值）**：来自伪影率曲线的拐点；
   - **Strict（P>0.05）**：剔除预测伪影概率大于 0.05 的颗粒。
4) 点击 `📈 Prediction Visualization` 查看预测分布与双曲线；界面日志会显示体素与 mm³ 两种单位的阈值。

## 5. 导出结果（Export Results）
- 点击 `📤 Export Results`：
  - 生成两份 XLSX：`Loose_Threshold_Results_*.xlsx`、`Strict_Threshold_Results_*.xlsx`
  - 生成一份 TXT 报告：`Threshold_Report_*.txt`
- 所有文件输出到 `outputs/` 目录。

## 6. 会话恢复（Load Last Time Model）
- 点击 `🔄 Load Last Time Model` 恢复上次训练的全部信息：训练数据、专家阈值、体素、特征工程器（含 scaler）、模型、分析结果等。

## 7. 特征分析（Feature Analysis）
- 目标：解释性地对比“被剔除的颗粒（伪影）”和“保留的颗粒（有效）”的特征差异。
- 方法：7 维日志椭球张量特征（含体素/分辨率处理），展示效应量（Cohen's d）、显著性（p 值）与相关性矩阵。
- 使用：
  1) 训练前，完成“专家阈值 + 体素”录入；
  2) 点击 `4. Feature Analysis`；
  3) 结果在日志中展示，可配合 `📊 Training Visualization` 观察模型视角。

## 8. Fabric 分析（T 与 P'）
- 目标：跨一系列体积阈值，基于保留颗粒做 Bootstrap，计算并可视化 T 与 P' 的箱线图；
- 方法：
  - 每个阈值下，重采样保留颗粒集合，计算 **log‑欧式均值** 的 fabric 张量；
  - 对其特征值取对数并指数化，按 Jelínek（1981）计算 T 与 P'；
  - Loose/Strict 阈值会一并显示在图中；
- 使用：先完成 `Load Test Data` 与 `Predict Analysis`，然后点击 `📦 Fabric Boxplots`。

## 9. 常见问题（FAQ）
- LightGBM 未安装？界面会自动回退到 RandomForest，功能不受影响。
- 输出在哪里？所有图/报告/模型都在 `outputs/`。
- 想批量评估多个测试样本？使用“多样本测试”（在菜单或分析模块中）。
- 数据不满足列要求？请参照第 1 节补齐最低字段。

## 10. 最佳实践
- 在训练/预测前，务必正确录入体素大小（mm），否则阈值的体素/mm³换算会偏差。
- 训练完成后建议查看 `📊 Training Visualization`，确认特征与分布是否合理。
- Fabric 分析中的 Bootstrap 会根据样本数与阈值数量耗时较长，建议在必要时使用。

---
如需命令行/脚本化调用，请联系开发者或参考 `src/ml_threshold_selection/` 下各模块的 Python API。
