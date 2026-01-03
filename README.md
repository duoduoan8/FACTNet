# FACTNet

论文（PDF）：[`FACTnet.pdf`](FACTnet.pdf)

## 简介

FACTNet（Frequency-Aware Cross-Attention Transformer Network）面向毫米波（mmWave）安检成像的计算超分辨率重建问题。
在实际成像链路中，受衍射极限影响，系统等效为对空间细节的低通滤波，高频信息（边缘/纹理）被显著削弱。
FACTNet 的核心思路是将重建过程与物理退化机理对齐：在频域与空域之间进行联合建模，以更有针对性地恢复被抑制的高频细节、降低伪影并提升视觉可辨识度。

本文方法在网络层面主要包含：
- 频域变换与增强的特征处理（FDTM）
- 频谱引导的交叉注意力机制（FACAM）
- 空域/频域双分支融合（DDF）

更完整的理论动机、结构设计、实验设置与结果请见：[`FACTnet.pdf`](FACTnet.pdf)。

## 代码结构

- `factnet/`：Python 包（网络、模型、数据集、loss、训练/测试入口）
- `options/`：BasicSR 风格的 YAML 配置
- `inference/`：推理与可视化脚本
- `scripts/experiments/`：论文实验复现脚本（训练/推理/传统方法基线）

## 安装

先安装 PyTorch，然后：

```bash
pip install -r requirements.txt
python setup.py develop
```

## 数据与路径

本仓库不包含数据。默认示例配置假设数据位于：

```text
datasets/THZ_SR/
  train/
  vail/
```

如需使用 `meta_info_file`（BasicSR 的 PairedImageDataset），可以用脚本自动生成：

```bash
python scripts/experiments/generate_meta_info_paired.py \
  --gt-root datasets/THZ_SR/train/transform_db_method5_wash_data_no_repeat \
  --lq-root datasets/THZ_SR/train/x4_psf \
  --out options/train/THZ/meta_info_THZ_SR_train_x4_psf.txt
```

## 复现脚本（对应论文实验）

- 训练（fast）：`bash scripts/experiments/train_factnet_thz_x4_psf_fast.sh`
- 训练（full）：`bash scripts/experiments/train_factnet_thz_x4_psf.sh`
- 推理（x4）：`bash scripts/experiments/infer_factnet_x4.sh <LR_DIR> <OUT_DIR> <WEIGHT>`
- 频域响应可视化：`bash scripts/experiments/visualize_freq_response.sh <WEIGHT> <INPUT> <OUTDIR>`
- 传统方法基线（TV / Richardson--Lucy）：`bash scripts/experiments/eval_classical_baselines_thz_x4_psf.sh`

说明：传统方法评测脚本需要额外依赖（例如 `scikit-image`）。
