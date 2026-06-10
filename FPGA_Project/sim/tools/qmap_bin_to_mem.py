"""Convert a QMAP binary image into a little-endian 32-bit readmemh file.

Run from the repository root:

    conda run -n llm_fpga python FPGA_Project/sim/tools/qmap_bin_to_mem.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "artifacts"
    / "test_vectors"
    / "qwen3_0p6b_qmap_v1"
    / "q_proj_row0_group0_dot64.qmap.bin"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "FPGA_Project"
    / "sim"
    / "vectors"
    / "qmap_dot64_image_words32.hex"
)


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a QMAP binary image into 32-bit little-endian hex words."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Missing {args.input}. Run 19_export_qmap_dot64_image.py first."
        )

    image = args.input.read_bytes()
    if len(image) % 4 != 0:
        image += bytes(4 - (len(image) % 4))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for offset in range(0, len(image), 4):
            word = int.from_bytes(image[offset : offset + 4], byteorder="little")
            file.write(f"{word:08X}\n")

    print("Converted QMAP image to 32-bit readmemh words")
    print(f"Input: {relpath(args.input)}")
    print(f"Output: {relpath(args.output)}")
    print(f"Words: {len(image) // 4}")


if __name__ == "__main__":
    main()
