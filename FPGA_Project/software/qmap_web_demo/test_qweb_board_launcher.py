from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "run_qweb_board.ps1"
LAUNCHER = HERE / "launch_qweb_board.tcl"
FORMAL_WORKSPACE = Path(r"F:\vwc")
ACCEPTED_WORKBENCH = Path(r"F:\qot_boardtest_prompt_text_v13_20260812")


def run_wrapper(*arguments: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WRAPPER),
            *arguments,
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class QwebBoardLauncherTests(unittest.TestCase):
    def test_launcher_loads_runtime_before_qweb(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("QWEB_RUNTIME_LOADER", text)
        self.assertIn("QWEB_NETWORK_WEB_ELF", text)
        source_position = text.index("source $runtime_loader")
        header_position = text.index("require_all_qmap_headers", source_position)
        qweb_position = text.index("dow $web_elf", header_position)
        self.assertLess(source_position, header_position)
        self.assertLess(header_position, qweb_position)
        self.assertIn("set_force_memory_access 0", text)

        wrapper = WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "Detected Motorcomm YT8521",
            "YT8521 link resolved",
            "QWEB READY http://",
            "TOKENIZER tokens=",
        ):
            self.assertIn(marker, wrapper)

    def test_wrapper_rejects_mutated_network_manifest_before_xsdb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = FORMAL_WORKSPACE / "network_workspace_manifest.json"
            if source.is_file():
                payload = source.read_bytes() + b"\n"
            else:
                payload = b"{}\n"
            (root / "network_workspace_manifest.json").write_bytes(payload)
            result = run_wrapper(
                "-Workspace",
                str(root),
                "-RuntimeWorkbench",
                str(ACCEPTED_WORKBENCH),
                "-AuditOnly",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Network workspace manifest SHA-256 mismatch", result.stderr)
        self.assertNotIn("XSDB", result.stdout)

    @unittest.skipUnless(
        (FORMAL_WORKSPACE / "network_workspace_manifest.json").is_file(),
        "formal F:\\vwc workspace is not present",
    )
    def test_wrapper_rejects_mutated_runtime_manifest_before_xsdb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = ACCEPTED_WORKBENCH / "workbench_manifest.json"
            if source.is_file():
                payload = source.read_bytes() + b"\n"
            else:
                payload = b"{}\n"
            (root / "workbench_manifest.json").write_bytes(payload)
            result = run_wrapper(
                "-Workspace",
                str(FORMAL_WORKSPACE),
                "-RuntimeWorkbench",
                str(root),
                "-AuditOnly",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Runtime workbench manifest SHA-256 mismatch", result.stderr)
        self.assertNotIn("audit-only mode", result.stdout)

    @unittest.skipUnless(
        (FORMAL_WORKSPACE / "network_workspace_manifest.json").is_file()
        and (ACCEPTED_WORKBENCH / "workbench_manifest.json").is_file(),
        "formal network workspace or accepted runtime workbench is not present",
    )
    def test_formal_artifacts_pass_audit_only(self) -> None:
        result = run_wrapper(
            "-Workspace",
            str(FORMAL_WORKSPACE),
            "-RuntimeWorkbench",
            str(ACCEPTED_WORKBENCH),
            "-AuditOnly",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS audited board-hosted QWEB launch inputs", result.stdout)
        self.assertIn("runtime: 61 segments / 394547200 bytes", result.stdout)
        self.assertIn("PASS audit-only mode; XSDB was not invoked", result.stdout)


if __name__ == "__main__":
    unittest.main()
