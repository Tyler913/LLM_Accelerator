from __future__ import annotations

import subprocess
import shutil
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "run_qweb_board.ps1"
LAUNCHER = HERE / "launch_qweb_board.tcl"
FORMAL_WORKSPACE = Path(r"F:\vwk")
ACCEPTED_WORKBENCH = Path(r"F:\qot_boardtest_prompt_text_v13_20260812")
CANONICAL_XSDB = Path(
    r"D:\Applications\Vivado_2025.1.1\2025.1.1\Vitis\bin\xsdb.bat"
)
CANONICAL_LOADER = CANONICAL_XSDB.with_name("loader.bat")


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
    def test_physical_launch_requires_active_evidence_directory(self) -> None:
        result = run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Physical QWEB launch requires -EvidenceDirectory", result.stderr)
        self.assertNotIn("attempting to launch hw_server", result.stdout)

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
        self.assertIn('Join-Path $evidencePath "launch.json"', wrapper)
        self.assertIn('run_id = $launchRunId', wrapper)
        claim_position = wrapper.index(
            "Write-JsonExclusive -Path $launchClaimPath -Payload $launchClaim"
        )
        invoke_position = wrapper.index("& $xsdbCommand @xsdbArguments")
        self.assertLess(claim_position, invoke_position)
        self.assertIn("$ExpectedXsdbSha256", wrapper)
        self.assertIn("$ExpectedLoaderSha256", wrapper)
        self.assertIn("$ExpectedSetupEnvSha256", wrapper)
        self.assertIn("$ExpectedRdiArgsSha256", wrapper)
        self.assertIn("$ExpectedXsdbExeSha256", wrapper)
        self.assertIn('"RDI_JAVALAUNCH"', wrapper)
        self.assertIn('"RDI_VBSLAUNCH"', wrapper)
        self.assertIn('"RDI_SETUP_ENV_FUNCTION"', wrapper)
        self.assertIn('"_RDI_BASELINE"', wrapper)
        self.assertIn('"PATH"', wrapper)
        self.assertIn('"PYTHONPATH"', wrapper)
        self.assertIn('"RDI_PREPEND_PATH"', wrapper)
        self.assertIn('"MYXILINX"', wrapper)
        self.assertIn('"QWEB_HW_SERVER_URL"', wrapper)
        self.assertIn('"QWEB_DEVICE_FILTER"', wrapper)
        self.assertIn('"TCLLIBPATH"', wrapper)
        self.assertIn('$xsdbArguments = @("-no-ini", $launcher)', wrapper)
        self.assertIn('$env:USERPROFILE = $xsdbProfilePath', wrapper)
        self.assertIn('Push-Location -LiteralPath $xsdbProfilePath', wrapper)
        self.assertIn("PASS all 281 QMAP packet headers", wrapper)

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
        "formal F:\\vwd workspace is not present",
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
    def test_fake_xsdb_cannot_forge_a_physical_launch_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_root = root / "bad_hash"
            fake_root.mkdir()
            fake_xsdb = fake_root / "xsdb.bat"
            fake_xsdb.write_text("@exit /b 0\r\n", encoding="ascii")
            (fake_root / "loader.bat").write_text(
                "@exit /b 0\r\n", encoding="ascii"
            )

            candidates = [("bad hash", fake_xsdb)]
            if CANONICAL_XSDB.is_file() and CANONICAL_LOADER.is_file():
                copied_root = root / "genuine_bats_fake_helpers"
                copied_root.mkdir()
                shutil.copyfile(CANONICAL_XSDB, copied_root / "xsdb.bat")
                shutil.copyfile(CANONICAL_LOADER, copied_root / "loader.bat")
                (copied_root / "setupEnv.bat").write_text(
                    "@exit /b 0\r\n", encoding="ascii"
                )
                (copied_root / "rdiArgs.bat").write_text(
                    "@exit /b 0\r\n", encoding="ascii"
                )
                candidates.append(("genuine wrappers with fake helpers", copied_root / "xsdb.bat"))

            for index, (label, candidate) in enumerate(candidates):
                with self.subTest(label=label):
                    evidence = root / f"evidence_{index}"
                    evidence.mkdir()
                    (evidence / "uart_raw.bin").write_bytes(b"")
                    result = run_wrapper(
                        "-Workspace",
                        str(FORMAL_WORKSPACE),
                        "-RuntimeWorkbench",
                        str(ACCEPTED_WORKBENCH),
                        "-Xsdb",
                        str(candidate),
                        "-EvidenceDirectory",
                        str(evidence),
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("requires canonical Vitis 2025.1", result.stderr)
                    self.assertFalse((evidence / "launch.claim.json").exists())
                    self.assertFalse((evidence / "launch.json").exists())

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
