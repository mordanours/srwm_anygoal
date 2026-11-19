"""Extract low-dimensional state/proprio/action arrays from AnyGoal .h5 or .h5.zst files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _maybe_decompress_zst(src: Path) -> tuple[Path, bool]:
    if src.suffix != ".zst":
        return src, False
    import zstandard

    tmp_h5 = src.with_suffix("")  # strip .zst
    print(f"Decompressing {src.name} -> {tmp_h5.name} ...")
    with src.open("rb") as fin, tmp_h5.open("wb") as fout:
        zstandard.ZstdDecompressor().copy_stream(fin, fout)
    return tmp_h5, True


def extract_state_only(src_path: Path, out_path: Path, cleanup_decompressed: bool = False) -> None:
    h5_path, created_tmp = _maybe_decompress_zst(src_path)
    arrays = {}
    try:
        with h5py.File(h5_path, "r") as f:
            for k in f.keys():
                if any(tok in k.lower() for tok in ("pixel", "image", "rgb")):
                    continue
                arrays[k] = f[k][:]
    finally:
        if created_tmp and cleanup_decompressed and h5_path.exists():
            h5_path.unlink()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    src_mb = src_path.stat().st_size / (1024**2)
    dst_mb = out_path.stat().st_size / (1024**2)
    print(f"Extracted {list(arrays.keys())} | {src_mb:.1f} MB -> {dst_mb:.2f} MB | saved to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strip pixels from AnyGoal .h5/.h5.zst into compact .npz")
    parser.add_argument("src", type=Path, help="Input .h5 or .h5.zst file")
    parser.add_argument("dst", type=Path, help="Output .npz file")
    parser.add_argument("--cleanup-tmp", action="store_true", help="Delete intermediate .h5 if input was .zst")
    args = parser.parse_args()
    extract_state_only(args.src, args.dst, cleanup_decompressed=args.cleanup_tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
