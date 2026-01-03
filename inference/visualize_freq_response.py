import argparse
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def _read_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    img = img.astype(np.float32) / 255.0
    img = img[:, :, [2, 1, 0]]  # BGR -> RGB
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
    return img.to(device)


def _tensor_to_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    img = img.detach().float().cpu().clamp_(0, 1).squeeze(0)
    if img.ndim != 3:
        raise ValueError(f'Expected (C,H,W), got {tuple(img.shape)}')
    img = img.numpy().transpose(1, 2, 0)
    img = (img * 255.0).round().astype(np.uint8)
    return img


def _compute_fft_mag(x2d: torch.Tensor) -> np.ndarray:
    spectrum = torch.fft.fft2(x2d)
    return torch.abs(spectrum).detach().cpu().numpy()


def _fftshift(arr: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(arr)


def _radial_average(arr2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = arr2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial = np.zeros(max_r, dtype=np.float32)
    for radius in range(max_r):
        mask = r == radius
        if mask.any():
            radial[radius] = float(arr2d[mask].mean())
    freq_axis = np.arange(max_r, dtype=np.float32) / max_r
    return radial, freq_axis


def _radial_gain_from_mags(
    input_mag: np.ndarray,
    output_mag: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    input_mag_shifted = _fftshift(input_mag)
    output_mag_shifted = _fftshift(output_mag)

    if method == 'pixel':
        gain_2d = output_mag_shifted / (input_mag_shifted + eps)
        radial_gain, freq_axis = _radial_average(gain_2d)
        return radial_gain, freq_axis

    if method == 'ring':
        radial_in, freq_axis = _radial_average(input_mag_shifted)
        radial_out, _ = _radial_average(output_mag_shifted)
        radial_gain = radial_out / (radial_in + eps)
        return radial_gain, freq_axis

    raise ValueError(f'Unknown radial method: {method}. Use ring|pixel.')


def visualize_frequency_response(
    x: torch.Tensor,
    y: torch.Tensor,
    save_path: str,
    title_prefix: str,
    vmin: float = 0.0,
    vmax: float | None = None,
):
    if x.shape != y.shape:
        raise ValueError(f'Shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}')
    if x.ndim != 4:
        raise ValueError(f'Expected (B,C,H,W), got {tuple(x.shape)}')

    x2d = x[0].mean(dim=0)
    y2d = y[0].mean(dim=0)

    input_mag = _compute_fft_mag(x2d)
    output_mag = _compute_fft_mag(y2d)
    eps = 1e-8
    freq_response = output_mag / (input_mag + eps)

    input_mag_shifted = _fftshift(input_mag)
    output_mag_shifted = _fftshift(output_mag)
    freq_response_shifted = _fftshift(freq_response)

    if vmax is None:
        p99 = float(np.percentile(freq_response_shifted, 99))
        vmax = float(np.clip(p99, 2.0, 10.0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(x2d.detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title(f'{title_prefix} Input (Spatial)', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(y2d.detach().cpu().numpy(), cmap='gray')
    axes[0, 1].set_title(f'{title_prefix} Output (Spatial)', fontsize=12)
    axes[0, 1].axis('off')

    diff = (y2d - x2d).detach().cpu().numpy()
    axes[0, 2].imshow(diff, cmap='RdBu_r')
    axes[0, 2].set_title(f'{title_prefix} Difference (Out - In)', fontsize=12)
    axes[0, 2].axis('off')

    im1 = axes[1, 0].imshow(np.log1p(input_mag_shifted), cmap='viridis')
    axes[1, 0].set_title(f'{title_prefix} Input Spectrum (log)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(np.log1p(output_mag_shifted), cmap='viridis')
    axes[1, 1].set_title(f'{title_prefix} Output Spectrum (log)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    im3 = axes[1, 2].imshow(freq_response_shifted, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title(f'{title_prefix} Frequency Response (Gain)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return freq_response_shifted


def visualize_radial_frequency_response(
    x: torch.Tensor,
    y: torch.Tensor,
    save_path: str,
    title_prefix: str,
    vmin: float = 0.0,
    vmax: float | None = None,
    method: str = 'ring',
):
    if x.shape != y.shape:
        raise ValueError(f'Shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}')
    if x.ndim != 4:
        raise ValueError(f'Expected (B,C,H,W), got {tuple(x.shape)}')

    x2d = x.mean(dim=(0, 1))
    y2d = y.mean(dim=(0, 1))

    input_mag = _compute_fft_mag(x2d)
    output_mag = _compute_fft_mag(y2d)

    radial_gain, freq_axis = _radial_gain_from_mags(input_mag, output_mag, method=method)
    freq_response_shifted = _fftshift(output_mag / (input_mag + 1e-8))

    if vmax is None:
        p99 = float(np.percentile(freq_response_shifted, 99))
        vmax = float(np.clip(p99, 2.0, 10.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(freq_response_shifted, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{title_prefix} 2D Frequency Response', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].plot(freq_axis, radial_gain, 'b-', linewidth=2)
    axes[1].axhline(y=1.0, color='r', linestyle='--', label='Unity Gain')
    axes[1].fill_between(freq_axis, radial_gain, alpha=0.3)
    axes[1].set_xlabel('Normalized Frequency (0=Low, 1=High)', fontsize=11)
    axes[1].set_ylabel('Gain', fontsize=11)
    axes[1].set_title(f'{title_prefix} Radial Average Gain', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, max(2.0, float(radial_gain.max()) * 1.1)])

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return radial_gain, freq_axis


def visualize_radial_stats(
    radials: np.ndarray,
    freq_axis: np.ndarray,
    save_path: str,
    title: str,
):
    mean = radials.mean(axis=0)
    median = np.median(radials, axis=0)
    p10 = np.percentile(radials, 10, axis=0)
    p90 = np.percentile(radials, 90, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.fill_between(freq_axis, p10, p90, alpha=0.25, label='P10–P90')
    ax.plot(freq_axis, mean, 'b-', linewidth=2, label='Mean')
    ax.plot(freq_axis, median, 'k--', linewidth=1.8, label='Median')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='Unity Gain')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Normalized Frequency (0=Low, 1=High)', fontsize=11)
    ax.set_ylabel('Gain', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(2.0, float(p90.max()) * 1.1)])
    ax.legend(loc='upper left')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _iter_image_paths(path: str, pattern: str) -> list[str]:
    if os.path.isdir(path):
        import glob

        paths = sorted(glob.glob(os.path.join(path, pattern)))
        return [p for p in paths if os.path.isfile(p)]
    return [path]


def _sample_paths(paths: list[str], num_samples: int, seed: int) -> list[str]:
    if num_samples <= 0:
        return paths
    if len(paths) <= num_samples:
        return paths
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=num_samples, replace=False)
    return [paths[i] for i in idx]


def _patch_svg_size(svg_path: str, width_px: int, height_px: int) -> None:
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg = f.read()

    # Matplotlib SVG header typically starts like:
    # <svg ... width="xxxpt" height="yyypt" viewBox="0 0 xxx yyy" ...>
    import re

    svg2 = re.sub(r'\bwidth="[^"]+"', f'width="{width_px}px"', svg, count=1)
    svg2 = re.sub(r'\bheight="[^"]+"', f'height="{height_px}px"', svg2, count=1)

    if svg2 == svg:
        raise RuntimeError(f'Failed to patch SVG width/height: {svg_path}')

    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg2)


def _load_factnet(weight_path: str, device: torch.device):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from factnet.archs.factnet_arch import FACTNet
    from factnet.archs.factnet_utils import FourierUnit

    ckpt = torch.load(weight_path, map_location='cpu')
    state_dict = ckpt.get('params_ema') or ckpt.get('params') or ckpt

    model = FACTNet(
        upscale=4,
        in_chans=3,
        img_size=60,
        window_size=4,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='SFB',
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    fu_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, FourierUnit)]
    return model, fu_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight',
        type=str,
        default='experiments/pretrained_models/FACTNet/FACTNet_SRx4.pth',
        help='Path to net_g_*.pth',
    )
    parser.add_argument(
        '--input',
        type=str,
        default='datasets/THZ_SR/vail/split_6_x4_psf',
        help='Path to an LR image file or a directory of images.',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='freq_vis',
        help='Directory to write visualizations.',
    )
    parser.add_argument('--pattern', type=str, default='*.png', help='Glob pattern used when --input is a directory.')
    parser.add_argument('--num-samples', type=int, default=50, help='Random sample size; set 0 to use all images.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling.')
    parser.add_argument(
        '--radial-method',
        type=str,
        default='ring',
        choices=['ring', 'pixel'],
        help='Radial gain definition: ring=ratio of ring means (stable), pixel=mean of per-pixel ratios.',
    )
    parser.add_argument(
        '--fu-index',
        type=int,
        default=0,
        help='Which FourierUnit (by traversal order) to hook for internal response.',
    )
    parser.add_argument(
        '--log-every',
        type=int,
        default=25,
        help='Print progress every N images; set 0 to disable per-image logs.',
    )
    parser.add_argument(
        '--stats-format',
        type=str,
        default='png',
        choices=['png', 'svg'],
        help='Output format for radial gain stats plots.',
    )
    parser.add_argument('--svg-width', type=int, default=1485, help='SVG width (px) when --stats-format=svg.')
    parser.add_argument('--svg-height', type=int, default=582, help='SVG height (px) when --stats-format=svg.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, fu_modules = _load_factnet(args.weight, device)
    if not fu_modules:
        raise RuntimeError('No FourierUnit modules found in FACTNet model.')
    if not (0 <= args.fu_index < len(fu_modules)):
        raise ValueError(f'--fu-index must be in [0, {len(fu_modules) - 1}], got {args.fu_index}')

    all_paths = _iter_image_paths(args.input, args.pattern)
    if not all_paths:
        raise FileNotFoundError(f'No images found under: {args.input} ({args.pattern})')
    paths = _sample_paths(all_paths, args.num_samples, args.seed) if os.path.isdir(args.input) else all_paths
    paths = sorted(paths)

    os.makedirs(args.outdir, exist_ok=True)

    fu_name, fu_mod = fu_modules[args.fu_index]

    saved = {}

    def _hook(_module, inputs, output):
        # Store spatial mean maps only (avoid copying full feature tensors to CPU).
        fu_in = inputs[0]
        fu_out = output
        saved['fu_in_mean2d'] = fu_in[0].mean(dim=0).detach().cpu()
        saved['fu_out_mean2d'] = fu_out[0].mean(dim=0).detach().cpu()

    handle = fu_mod.register_forward_hook(_hook)

    end2end_radials: list[np.ndarray] = []
    fu_radials: list[np.ndarray] = []
    freq_axis_end2end: np.ndarray | None = None
    freq_axis_fu: np.ndarray | None = None

    for idx, p in enumerate(paths, start=1):
        lr = _read_image_as_tensor(p, device)
        with torch.no_grad():
            sr = model(lr)
        sr = sr.detach()
        bicubic = F.interpolate(lr, size=sr.shape[-2:], mode='bicubic', align_corners=False)

        # End-to-end radial gain (SR vs bicubic).
        x2d = bicubic[0].mean(dim=0)
        y2d = sr[0].mean(dim=0)
        input_mag = _compute_fft_mag(x2d)
        output_mag = _compute_fft_mag(y2d)
        radial_gain, freq_axis = _radial_gain_from_mags(input_mag, output_mag, method=args.radial_method)
        end2end_radials.append(radial_gain)
        if freq_axis_end2end is None:
            freq_axis_end2end = freq_axis

        # FourierUnit radial gain (feature-space).
        if 'fu_in_mean2d' not in saved or 'fu_out_mean2d' not in saved:
            raise RuntimeError('FourierUnit hook did not capture tensors; forward hook may not have fired.')
        fu_in_mean2d = saved['fu_in_mean2d']
        fu_out_mean2d = saved['fu_out_mean2d']
        fu_in_mag = _compute_fft_mag(fu_in_mean2d)
        fu_out_mag = _compute_fft_mag(fu_out_mean2d)
        fu_radial_gain, fu_freq_axis = _radial_gain_from_mags(fu_in_mag, fu_out_mag, method=args.radial_method)
        fu_radials.append(fu_radial_gain)
        if freq_axis_fu is None:
            freq_axis_fu = fu_freq_axis

        if args.log_every > 0 and (idx == 1 or idx % args.log_every == 0 or idx == len(paths)):
            print(f'[{idx}/{len(paths)}] {os.path.basename(p)}')

        # Save per-sample visualizations for the first image (useful as a concrete example).
        if idx == 1:
            cv2.imwrite(
                os.path.join(args.outdir, 'lr_input.png'),
                cv2.cvtColor(_tensor_to_uint8_rgb(lr), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(args.outdir, 'sr_output.png'),
                cv2.cvtColor(_tensor_to_uint8_rgb(sr), cv2.COLOR_RGB2BGR),
            )
            visualize_frequency_response(
                bicubic.cpu(),
                sr.cpu(),
                save_path=os.path.join(args.outdir, 'freq_response_end2end.png'),
                title_prefix='End2End (SR vs Bicubic)',
            )
            visualize_radial_frequency_response(
                bicubic.cpu(),
                sr.cpu(),
                save_path=os.path.join(args.outdir, 'radial_freq_response_end2end.png'),
                title_prefix='End2End (SR vs Bicubic)',
                method=args.radial_method,
            )
            visualize_frequency_response(
                fu_in_mean2d.unsqueeze(0).unsqueeze(0),
                fu_out_mean2d.unsqueeze(0).unsqueeze(0),
                save_path=os.path.join(args.outdir, f'freq_response_fourierunit_{args.fu_index:02d}.png'),
                title_prefix=f'FourierUnit[{args.fu_index}] {fu_name}',
            )
            visualize_radial_frequency_response(
                fu_in_mean2d.unsqueeze(0).unsqueeze(0),
                fu_out_mean2d.unsqueeze(0).unsqueeze(0),
                save_path=os.path.join(args.outdir, f'radial_freq_response_fourierunit_{args.fu_index:02d}.png'),
                title_prefix=f'FourierUnit[{args.fu_index}] {fu_name}',
                method=args.radial_method,
            )

    handle.remove()

    end2end_radials_np = np.stack(end2end_radials, axis=0)
    fu_radials_np = np.stack(fu_radials, axis=0)
    if freq_axis_end2end is None or freq_axis_fu is None:
        raise RuntimeError('Failed to build frequency axis.')

    visualize_radial_stats(
        end2end_radials_np,
        freq_axis_end2end,
        save_path=os.path.join(args.outdir, f'radial_gain_stats_end2end.{args.stats_format}'),
        title=f'End2End (SR vs Bicubic) Radial Gain Stats (n={len(paths)}, method={args.radial_method})',
    )
    visualize_radial_stats(
        fu_radials_np,
        freq_axis_fu,
        save_path=os.path.join(args.outdir, f'radial_gain_stats_fourierunit_{args.fu_index:02d}.{args.stats_format}'),
        title=f'FourierUnit[{args.fu_index}] {fu_name} Radial Gain Stats (n={len(paths)}, method={args.radial_method})',
    )

    if args.stats_format == 'svg':
        _patch_svg_size(
            os.path.join(args.outdir, 'radial_gain_stats_end2end.svg'),
            width_px=args.svg_width,
            height_px=args.svg_height,
        )
        _patch_svg_size(
            os.path.join(args.outdir, f'radial_gain_stats_fourierunit_{args.fu_index:02d}.svg'),
            width_px=args.svg_width,
            height_px=args.svg_height,
        )

    print('Saved to:', args.outdir)
    print('Samples:', len(paths), 'from', args.input)
    print('Hooked FourierUnit:', fu_name)
    print('Radial method:', args.radial_method)


if __name__ == '__main__':
    main()
