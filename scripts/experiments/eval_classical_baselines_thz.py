import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_tv_bregman, richardson_lucy


try:
    import niqe as niqe_pkg  # type: ignore
except Exception:  # pragma: no cover
    niqe_pkg = None


@dataclass(frozen=True)
class Pair:
    lr_path: Path
    hr_path: Path
    split: str


def _psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float('inf')
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def _to_gray(bgr_float01: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr_float01)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        structural_similarity(
            a,
            b,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def _read_bgr_float01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return img.astype(np.float32) / 255.0


def _ensure_niqe_model(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://raw.githubusercontent.com/aizvorski/video-quality/master/modelparameters.mat'
    import urllib.request

    with urllib.request.urlopen(url) as resp, model_path.open('wb') as f:
        f.write(resp.read())


def _niqe_gray(img_gray_float01: np.ndarray, model_path: Path) -> float:
    if niqe_pkg is None:
        return float('nan')
    _ensure_niqe_model(model_path)
    old_cwd = os.getcwd()
    try:
        os.chdir(str(model_path.parent))
        return float(niqe_pkg.niqe(img_gray_float01))
    finally:
        os.chdir(old_cwd)


def gaussian_psf(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([[1.0]], dtype=np.float32)
    radius = int(math.ceil(truncate * sigma))
    k = radius * 2 + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    psf /= float(psf.sum())
    return psf.astype(np.float32)


def collect_pairs(val_root: Path, scale: int, splits: list[int], *, lr_suffix: str = '') -> list[Pair]:
    pairs: list[Pair] = []
    lr_suffix = (lr_suffix or '').replace('\\', '_').replace('/', '_')
    for split_id in splits:
        hr_dir = val_root / f'split_{split_id}'
        lr_dir = val_root / f'split_{split_id}_x{scale}{lr_suffix}'
        if not hr_dir.is_dir():
            raise FileNotFoundError(f'Missing HR dir: {hr_dir}')
        if not lr_dir.is_dir():
            raise FileNotFoundError(f'Missing LR dir: {lr_dir}')

        for lr_path in sorted(lr_dir.glob('*.png')):
            hr_path = hr_dir / lr_path.name
            if hr_path.exists():
                pairs.append(Pair(lr_path=lr_path, hr_path=hr_path, split=f'split_{split_id}'))
    if not pairs:
        raise RuntimeError(f'No LR/HR pairs found under {val_root} for x{scale}.')
    return pairs


def ibp_super_resolve(
    lr: np.ndarray,
    hr_hw: tuple[int, int],
    *,
    sigma: float,
    ibp_iters: int,
    ibp_beta: float,
    down_interp: int = cv2.INTER_AREA,
    up_interp: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    hr_h, hr_w = hr_hw
    x = cv2.resize(lr, (hr_w, hr_h), interpolation=up_interp)
    for _ in range(ibp_iters):
        if sigma > 0:
            blurred = cv2.GaussianBlur(
                x, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT101
            )
        else:
            blurred = x
        lr_hat = cv2.resize(blurred, (lr.shape[1], lr.shape[0]), interpolation=down_interp)
        err = lr - lr_hat
        err_up = cv2.resize(err, (hr_w, hr_h), interpolation=up_interp)
        x = np.clip(x + ibp_beta * err_up, 0.0, 1.0)
    return x


def tv_split_bregman_sr(
    lr: np.ndarray,
    hr_hw: tuple[int, int],
    *,
    sigma: float,
    ibp_iters: int,
    ibp_beta: float,
    tv_weight: float,
    tv_max_iter: int,
) -> np.ndarray:
    x = ibp_super_resolve(lr, hr_hw, sigma=sigma, ibp_iters=ibp_iters, ibp_beta=ibp_beta)
    x = denoise_tv_bregman(
        x, weight=tv_weight, max_num_iter=tv_max_iter, eps=1e-4, isotropic=True, channel_axis=-1
    )
    return np.clip(x, 0.0, 1.0)


def rl_deconvolution_sr(
    lr: np.ndarray,
    hr_hw: tuple[int, int],
    *,
    sigma: float,
    rl_iters: int,
) -> np.ndarray:
    hr_h, hr_w = hr_hw
    up = cv2.resize(lr, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    psf = gaussian_psf(sigma)
    out_ch = []
    for c in range(up.shape[2]):
        out_ch.append(richardson_lucy(up[..., c], psf, num_iter=rl_iters, clip=False))
    out = np.stack(out_ch, axis=-1)
    return np.clip(out, 0.0, 1.0)


def _parse_splits(s: str) -> list[int]:
    s = s.strip().lower()
    if s == 'all':
        return [1, 2, 3, 4, 5, 6]
    out: list[int] = []
    for part in s.split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError('Empty --splits')
    return out


def main() -> int:
    p = argparse.ArgumentParser(description='Classical baselines on THZ_SR vail set (TV/Split-Bregman + RL).')
    p.add_argument('--val-root', type=Path, default=Path('datasets/THZ_SR/vail'))
    p.add_argument('--scale', type=int, default=4, choices=[2, 3, 4])
    p.add_argument('--splits', type=str, default='all', help='Split IDs: all or comma list (e.g. 1,2,3)')
    p.add_argument('--lr-suffix', type=str, default='_psf', help='Extra suffix appended to LR dir name')
    p.add_argument('--out-dir', type=Path, default=Path('outputs'), help='Output directory')

    p.add_argument('--sigma', type=float, default=0.6, help='Gaussian sigma')
    p.add_argument('--ibp-iters', type=int, default=20)
    p.add_argument('--ibp-beta', type=float, default=1.0)
    p.add_argument('--tv-weight', type=float, default=100.0)
    p.add_argument('--tv-max-iter', type=int, default=50)
    p.add_argument('--rl-iters', type=int, default=10)

    p.add_argument('--niqe-model', type=Path, default=Path('modelparameters.mat'))
    args = p.parse_args()

    splits = _parse_splits(args.splits)
    lr_suffix = (args.lr_suffix or '').replace('\\', '_').replace('/', '_')
    pairs = collect_pairs(args.val_root, args.scale, splits, lr_suffix=lr_suffix)

    out_root = args.out_dir / f'val_x{args.scale}{lr_suffix}'
    out_root.mkdir(parents=True, exist_ok=True)

    metrics_path = out_root / 'metrics.csv'
    summary_path = out_root / 'summary.txt'

    if niqe_pkg is None:
        print('[WARN] niqe package is not installed; NIQE will be reported as NaN.')

    t0 = perf_counter()
    psnr_bic: list[float] = []
    psnr_tv: list[float] = []
    psnr_rl: list[float] = []
    ssim_bic: list[float] = []
    ssim_tv: list[float] = []
    ssim_rl: list[float] = []
    niqe_bic: list[float] = []
    niqe_tv: list[float] = []
    niqe_rl: list[float] = []

    with metrics_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(
            [
                'split',
                'name',
                'psnr_bicubic',
                'psnr_tv_split_bregman',
                'psnr_richardson_lucy',
                'ssim_bicubic',
                'ssim_tv_split_bregman',
                'ssim_richardson_lucy',
                'niqe_bicubic',
                'niqe_tv_split_bregman',
                'niqe_richardson_lucy',
            ]
        )
        for idx, pair in enumerate(pairs, 1):
            lr = _read_bgr_float01(pair.lr_path)
            hr = _read_bgr_float01(pair.hr_path)
            hr_h, hr_w = hr.shape[:2]

            bic = cv2.resize(lr, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
            tv = tv_split_bregman_sr(
                lr,
                (hr_h, hr_w),
                sigma=float(args.sigma),
                ibp_iters=args.ibp_iters,
                ibp_beta=float(args.ibp_beta),
                tv_weight=float(args.tv_weight),
                tv_max_iter=args.tv_max_iter,
            )
            rl = rl_deconvolution_sr(lr, (hr_h, hr_w), sigma=float(args.sigma), rl_iters=args.rl_iters)

            hr_y = _to_gray(hr)
            bic_y = _to_gray(bic)
            tv_y = _to_gray(tv)
            rl_y = _to_gray(rl)

            p_b = _psnr(bic_y, hr_y)
            p_t = _psnr(tv_y, hr_y)
            p_r = _psnr(rl_y, hr_y)
            s_b = _ssim_gray(bic_y, hr_y)
            s_t = _ssim_gray(tv_y, hr_y)
            s_r = _ssim_gray(rl_y, hr_y)
            n_b = _niqe_gray(bic_y, args.niqe_model)
            n_t = _niqe_gray(tv_y, args.niqe_model)
            n_r = _niqe_gray(rl_y, args.niqe_model)

            psnr_bic.append(p_b)
            psnr_tv.append(p_t)
            psnr_rl.append(p_r)
            ssim_bic.append(s_b)
            ssim_tv.append(s_t)
            ssim_rl.append(s_r)
            niqe_bic.append(n_b)
            niqe_tv.append(n_t)
            niqe_rl.append(n_r)

            w.writerow(
                [
                    pair.split,
                    pair.lr_path.name,
                    f'{p_b:.6f}',
                    f'{p_t:.6f}',
                    f'{p_r:.6f}',
                    f'{s_b:.6f}',
                    f'{s_t:.6f}',
                    f'{s_r:.6f}',
                    f'{n_b:.6f}',
                    f'{n_t:.6f}',
                    f'{n_r:.6f}',
                ]
            )

            if idx % 50 == 0 or idx == len(pairs):
                elapsed = perf_counter() - t0
                print(
                    f'[{idx:5d}/{len(pairs):5d}]  '
                    f'PSNR={np.mean(psnr_bic):.3f}/{np.mean(psnr_tv):.3f}/{np.mean(psnr_rl):.3f}  '
                    f'SSIM={np.mean(ssim_bic):.4f}/{np.mean(ssim_tv):.4f}/{np.mean(ssim_rl):.4f}  '
                    f'NIQE={np.nanmean(niqe_bic):.3f}/{np.nanmean(niqe_tv):.3f}/{np.nanmean(niqe_rl):.3f}  '
                    f'elapsed={elapsed:.1f}s'
                )

    with summary_path.open('w', encoding='utf-8') as f:
        f.write(f'val_root={args.val_root}\n')
        f.write(f'scale=x{args.scale}\n')
        f.write(f'splits={splits}\n')
        f.write(f'pairs={len(pairs)}\n')
        f.write(f'sigma={float(args.sigma):.6f}\n')
        f.write(f'ibp_iters={args.ibp_iters}\n')
        f.write(f'ibp_beta={float(args.ibp_beta):.6f}\n')
        f.write(f'tv_weight={float(args.tv_weight):.6f}\n')
        f.write(f'tv_max_iter={args.tv_max_iter}\n')
        f.write(f'rl_iters={args.rl_iters}\n')
        f.write(f'niqe_model={args.niqe_model}\n')
        f.write('\n')
        f.write(f'psnr_bicubic_mean={float(np.mean(psnr_bic)):.6f}\n')
        f.write(f'psnr_tv_split_bregman_mean={float(np.mean(psnr_tv)):.6f}\n')
        f.write(f'psnr_richardson_lucy_mean={float(np.mean(psnr_rl)):.6f}\n')
        f.write(f'ssim_bicubic_mean={float(np.mean(ssim_bic)):.6f}\n')
        f.write(f'ssim_tv_split_bregman_mean={float(np.mean(ssim_tv)):.6f}\n')
        f.write(f'ssim_richardson_lucy_mean={float(np.mean(ssim_rl)):.6f}\n')
        f.write(f'niqe_bicubic_mean={float(np.nanmean(niqe_bic)):.6f}\n')
        f.write(f'niqe_tv_split_bregman_mean={float(np.nanmean(niqe_tv)):.6f}\n')
        f.write(f'niqe_richardson_lucy_mean={float(np.nanmean(niqe_rl)):.6f}\n')

    total = perf_counter() - t0
    print(f'Done. Wrote {metrics_path} and {summary_path} (total {total:.1f}s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

