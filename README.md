# FACTNet

FACTNet is a frequency-domain-informed Transformer model for computational super-resolution of mmWave security screening images.

This repository follows a BasicSR-style project layout:
- `factnet/`: the Python package (networks, models, data, losses)
- `options/`: training configuration files
- `inference/`: simple inference scripts

## Architecture

![](assets/FACTNet.png)

## Installation

Install PyTorch first, then:

```bash
pip install -r requirements.txt
python setup.py develop
```

## Dataset

Edit the dataset roots in the YAML config files under `options/`.
The provided example configs assume your data are placed under:

```
datasets/THZ_SR/
  train/
  vail/
```

Some training configs use a BasicSR `meta_info_file` for paired datasets. You can generate it with:

```bash
python scripts/experiments/generate_meta_info_paired.py \
  --gt-root datasets/THZ_SR/train/transform_db_method5_wash_data_no_repeat \
  --lq-root datasets/THZ_SR/train/x4_psf \
  --out options/train/THZ/meta_info_THZ_SR_train_x4_psf.txt
```

## Training

Example (single GPU):

```bash
bash scripts/experiments/train_factnet_thz_x4_psf_fast.sh
```

Training logs and checkpoints are saved under `experiments/`.

## Inference

Single-image super-resolution:

```bash
python inference/inference_factnet.py \
  --input path/to/LR_images \
  --output results/FACTNet \
  --task FACTNet \
  --scale 4 \
  --model_path path/to/FACTNet_SRx4.pth
```

## Reproducibility Scripts

- Train (full): `bash scripts/experiments/train_factnet_thz_x4_psf.sh`
- Train (fast): `bash scripts/experiments/train_factnet_thz_x4_psf_fast.sh`
- Inference (x4): `bash scripts/experiments/infer_factnet_x4.sh <LR_DIR> <OUT_DIR> <WEIGHT>`
- Frequency response visualization: `bash scripts/experiments/visualize_freq_response.sh <WEIGHT> <INPUT> <OUTDIR>`
- Classical baselines (TV / Richardson--Lucy): `bash scripts/experiments/eval_classical_baselines_thz_x4_psf.sh`

Note: the classical-baseline script requires extra packages (e.g. `scikit-image`).

Stereo image super-resolution (optional):

```bash
python inference/inference_factnetssr.py \
  --input path/to/stereo_lr_x4 \
  --output results/FACTNetSSR \
  --task FACTNetSSR_Local \
  --scale 4 \
  --model_path path/to/FACTNetSSR_SSRx4.pth
```

## Acknowledgements

This codebase is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
