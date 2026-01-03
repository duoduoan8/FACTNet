import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description='Generate BasicSR PairedImageDataset meta_info_file from paired folders.')
    p.add_argument('--gt-root', type=Path, required=True, help='Path to GT folder')
    p.add_argument('--lq-root', type=Path, required=True, help='Path to LQ folder')
    p.add_argument('--out', type=Path, required=True, help='Output meta_info txt path')
    p.add_argument('--ext', type=str, default='.png', help='File extension (default: .png)')
    args = p.parse_args()

    gt_root = args.gt_root
    lq_root = args.lq_root
    ext = args.ext if args.ext.startswith('.') else f'.{args.ext}'

    if not gt_root.is_dir():
        raise FileNotFoundError(f'Missing GT folder: {gt_root}')
    if not lq_root.is_dir():
        raise FileNotFoundError(f'Missing LQ folder: {lq_root}')

    names = []
    for lq_path in sorted(lq_root.glob(f'*{ext}')):
        gt_path = gt_root / lq_path.name
        if gt_path.is_file():
            names.append(lq_path.name)

    if not names:
        raise RuntimeError(f'No paired files found under {lq_root} with extension {ext}.')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(names) + '\n', encoding='utf-8')
    print(f'Wrote {len(names)} lines to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

