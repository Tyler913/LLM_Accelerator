from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from yt8521_lwip220_patch import (
    LIBRARY_VERSION,
    MOTORCOMM_YT8521_PHY_ID,
    PATCH_RELATIVE_PATH,
    PINNED_INPUT_SHA256,
    patch_source_text,
    stage_patched_library,
    verify_staged_library,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _installed_lwip220() -> Path | None:
    candidates: list[Path] = []
    explicit = os.environ.get("QWEB_VITIS_LWIP220_SOURCE", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    vitis_root = os.environ.get("XILINX_VITIS", "").strip()
    if vitis_root:
        candidates.append(
            Path(vitis_root)
            / "data/embeddedsw/ThirdParty/sw_services"
            / LIBRARY_VERSION
        )
    candidates.append(
        Path(r"D:\Applications\Vivado_2025.1.1\2025.1.1\Vitis")
        / "data/embeddedsw/ThirdParty/sw_services"
        / LIBRARY_VERSION
    )
    return next((path for path in candidates if path.is_dir()), None)


class YT8521Lwip220PatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _installed_lwip220()

    def test_pinned_source_patches_deterministically(self) -> None:
        if self.source is None:
            self.skipTest("pinned Vitis 2025.1.1 lwip220 is not installed")
        source_file = self.source / PATCH_RELATIVE_PATH
        self.assertEqual(_sha256(source_file), PINNED_INPUT_SHA256)
        original = source_file.read_text(encoding="utf-8")
        patched = patch_source_text(original)
        self.assertNotEqual(patched, original)
        self.assertIn("PHY_MOTORCOMM_YT8521_ID", patched)
        self.assertIn("get_Motorcomm_YT8521_phy_speed", patched)
        self.assertIn("Unsupported PHY initialization", patched)
        self.assertNotIn(
            "else {\n\t\tRetStatus = get_Marvell_phy_speed",
            patched,
        )
        with self.assertRaisesRegex(RuntimeError, "expected one"):
            patch_source_text(patched)

    def test_full_tree_staging_is_reaudited_and_tamper_evident(self) -> None:
        if self.source is None:
            self.skipTest("pinned Vitis 2025.1.1 lwip220 is not installed")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            destination = Path(temporary) / LIBRARY_VERSION
            record = stage_patched_library(self.source, destination)
            self.assertEqual(record["motorcomm_phy_id"], MOTORCOMM_YT8521_PHY_ID)
            self.assertEqual(record["original_sha256"], PINNED_INPUT_SHA256)
            self.assertNotEqual(
                record["patched_sha256"],
                record["original_sha256"],
            )
            verify_staged_library(record)
            patched_file = destination / PATCH_RELATIVE_PATH
            patched_file.write_text(
                patched_file.read_text(encoding="utf-8") + "\n/* tampered */\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_staged_library(record)

    def test_wrong_library_directory_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "not-the-pinned-library"
            source.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be named"):
                stage_patched_library(
                    source,
                    Path(temporary) / "destination",
                )


if __name__ == "__main__":
    unittest.main()
