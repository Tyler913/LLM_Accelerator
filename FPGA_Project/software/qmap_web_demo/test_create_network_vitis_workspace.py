from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from create_network_vitis_workspace import (
    NetworkWorkspacePlan,
    execute_plan,
    plan_from_environment,
)
from test_audit_network_xsa import _synthetic_hwh, _write_xsa


class _FakeComponent:
    def __init__(self, result: int | None, elf_path: Path | None = None) -> None:
        self.result = result
        self.elf_path = elf_path
        self.build_count = 0

    def build(self) -> int | None:
        self.build_count += 1
        if self.result == 0 and self.elf_path is not None:
            self.elf_path.parent.mkdir(parents=True, exist_ok=True)
            header = bytearray(64)
            header[:4] = b"\x7fELF"
            header[4] = 2
            header[5] = 1
            header[18:20] = (183).to_bytes(2, "little")
            self.elf_path.write_bytes(bytes(header) + b"fake-aarch64-web")
        return self.result


class _FakeClient:
    def __init__(
        self,
        *,
        platform_result: int = 0,
        app_result: int | None = 0,
        web_result: int | None = 0,
        tamper_snapshot: bool = False,
    ) -> None:
        self.workspace: str | None = None
        self.platform_args: dict[str, object] | None = None
        self.app_args: dict[str, object] | None = None
        self.app_calls: list[dict[str, object]] = []
        self.web_result = web_result
        self.tamper_snapshot = tamper_snapshot
        self.components = {
            "p_net": _FakeComponent(platform_result),
            "a_net_echo": _FakeComponent(app_result),
        }

    def set_workspace(self, *, path: str) -> None:
        self.workspace = path

    def create_advanced_options_dict(self, **kwargs: str) -> dict[str, str]:
        return dict(kwargs)

    def create_platform_component(
        self,
        *,
        name: str,
        hw_design: str,
        os: str,
        cpu: str,
        domain_name: str,
        generate_dtb: bool,
        advanced_options: dict[str, str],
        architecture: str,
        compiler: str,
    ) -> str:
        self.platform_args = {
            "name": name,
            "hw_design": hw_design,
            "os": os,
            "cpu": cpu,
            "domain_name": domain_name,
            "generate_dtb": generate_dtb,
            "advanced_options": advanced_options,
            "architecture": architecture,
            "compiler": compiler,
        }
        if self.tamper_snapshot:
            Path(hw_design).write_bytes(b"tampered-after-vitis-open")
        return "created-platform"

    def create_app_component(
        self,
        *,
        name: str,
        platform: str,
        domain: str,
        template: str,
    ) -> str:
        self.app_args = {
            "name": name,
            "platform": platform,
            "domain": domain,
            "template": template,
        }
        self.app_calls.append(self.app_args)
        if name not in self.components:
            if self.workspace is None:
                raise RuntimeError("test workspace was not selected")
            component_dir = Path(self.workspace) / name
            source_dir = component_dir / "src"
            source_dir.mkdir(parents=True)
            (source_dir / "UserConfig.cmake").write_text(
                'set(USER_COMPILE_SOURCES\n"main.c"\n)\n',
                encoding="utf-8",
            )
            (source_dir / "lscript.ld").write_text(
                "_STACK_SIZE = DEFINED(_STACK_SIZE) ? _STACK_SIZE : 0x2000;\n"
                "_HEAP_SIZE = DEFINED(_HEAP_SIZE) ? _HEAP_SIZE : 0x2000;\n",
                encoding="utf-8",
            )
            (source_dir / "echo.c").write_text(
                "/* fake template echo */\n", encoding="utf-8"
            )
            self.components[name] = _FakeComponent(
                self.web_result,
                component_dir / "build" / f"{name}.elf",
            )
        return "created-app"

    def get_component(self, *, name: str) -> _FakeComponent:
        return self.components[name]


class _FakeVitis:
    def __init__(
        self,
        *,
        platform_result: int = 0,
        app_result: int | None = 0,
        web_result: int | None = 0,
        tamper_snapshot: bool = False,
    ) -> None:
        self.client = _FakeClient(
            platform_result=platform_result,
            app_result=app_result,
            web_result=web_result,
            tamper_snapshot=tamper_snapshot,
        )
        self.create_count = 0
        self.dispose_count = 0

    def create_client(self) -> _FakeClient:
        self.create_count += 1
        return self.client

    def dispose(self) -> None:
        self.dispose_count += 1


class NetworkWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.actual_repo = Path(__file__).resolve().parents[3]
        self.xsa = self.root / "network.xsa"
        _write_xsa(self.xsa, _synthetic_hwh())

    def environment(self, workspace: Path) -> dict[str, str]:
        return {
            "QWEB_NETWORK_XSA": str(self.xsa),
            "QWEB_VITIS_WORKSPACE": str(workspace),
        }

    def test_valid_plan_is_read_only(self) -> None:
        workspace = self.root / "workspace"
        plan = plan_from_environment(
            self.environment(workspace), repository_root=self.repo
        )
        self.assertEqual(plan.workspace, workspace.resolve())
        self.assertEqual(plan.platform_name, "p_net")
        self.assertEqual(plan.app_name, "a_net_echo")
        self.assertEqual(len(plan.xsa_sha256), 64)
        self.assertEqual(len(plan.bitstream_sha256), 64)
        self.assertFalse(workspace.exists())

    def test_network_audit_failure_stops_before_workspace(self) -> None:
        bad_xsa = self.root / "non_network.xsa"
        _write_xsa(
            bad_xsa,
            _synthetic_hwh(
                network_overrides={"PSU__ENET3__PERIPHERAL__ENABLE": "0"}
            ),
        )
        workspace = self.root / "workspace"
        env = self.environment(workspace)
        env["QWEB_NETWORK_XSA"] = str(bad_xsa)
        with self.assertRaisesRegex(RuntimeError, "network XSA audit failed"):
            plan_from_environment(env, repository_root=self.repo)
        self.assertFalse(workspace.exists())

    def test_missing_bitstream_is_rejected(self) -> None:
        no_bit_xsa = self.root / "no_bit.xsa"
        _write_xsa(no_bit_xsa, _synthetic_hwh(), include_bit=False)
        env = self.environment(self.root / "workspace")
        env["QWEB_NETWORK_XSA"] = str(no_bit_xsa)
        with self.assertRaisesRegex(RuntimeError, "exactly one embedded"):
            plan_from_environment(env, repository_root=self.repo)

    def test_workspace_overlap_and_nonempty_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "overlap the source"):
            plan_from_environment(
                self.environment(self.repo / "generated"),
                repository_root=self.repo,
            )
        nonempty = self.root / "nonempty"
        nonempty.mkdir()
        (nonempty / "owned.txt").write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "must not already exist"):
            plan_from_environment(
                self.environment(nonempty), repository_root=self.repo
            )
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(RuntimeError, "must not already exist"):
            plan_from_environment(
                self.environment(empty), repository_root=self.repo
            )

    def test_component_names_are_strict(self) -> None:
        env = self.environment(self.root / "workspace")
        env["QWEB_VITIS_PLATFORM"] = "bad-name"
        with self.assertRaisesRegex(RuntimeError, "must match"):
            plan_from_environment(env, repository_root=self.repo)

    def test_execute_uses_isolated_platform_and_echo_template(self) -> None:
        workspace = self.root / "workspace"
        base_plan = plan_from_environment(
            self.environment(workspace), repository_root=self.repo
        )
        # execute_plan protects the real repository root. Keep the test output
        # outside it while retaining the already audited immutable plan.
        plan = NetworkWorkspacePlan(**base_plan.__dict__)
        fake = _FakeVitis()
        manifest = execute_plan(plan, vitis_module=fake)
        self.assertEqual(fake.dispose_count, 1)
        self.assertEqual(fake.client.workspace, workspace.as_posix())
        staged_xsa = Path(str(fake.client.platform_args["hw_design"]))
        self.assertEqual(staged_xsa.parent, workspace.resolve())
        self.assertTrue(staged_xsa.name.startswith("network_input_"))
        self.assertEqual(staged_xsa.read_bytes(), self.xsa.read_bytes())
        self.assertEqual(fake.client.platform_args["cpu"], "psu_cortexa53_0")
        self.assertEqual(fake.client.platform_args["architecture"], "64-bit")
        self.assertEqual(fake.client.app_args["template"], "lwip_echo_server")
        self.assertEqual(fake.client.app_args["domain"], "standalone_psu_cortexa53_0")
        self.assertEqual(manifest["build_results"]["platform"], 0)
        self.assertEqual(manifest["xsa_snapshot"], str(staged_xsa))
        manifest_path = workspace / "network_workspace_manifest.json"
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(
            json.loads(manifest_path.read_text(encoding="utf-8"))["app_name"],
            "a_net_echo",
        )

    def test_execute_snapshots_and_reaudits_before_vitis_import(self) -> None:
        workspace = self.root / "workspace"
        plan = plan_from_environment(
            self.environment(workspace), repository_root=self.repo
        )
        _write_xsa(
            self.xsa,
            _synthetic_hwh(
                network_overrides={"PSU__ENET3__PERIPHERAL__ENABLE": "0"}
            ),
        )
        fake = _FakeVitis()
        with self.assertRaisesRegex(RuntimeError, "provenance re-audit"):
            execute_plan(plan, vitis_module=fake)
        self.assertTrue(workspace.is_dir())
        self.assertEqual(fake.create_count, 0)
        self.assertEqual(fake.dispose_count, 0)

    def test_build_failure_is_not_written_as_success_manifest(self) -> None:
        cases = (
            ("platform", 1, 0, "platform build failed"),
            ("application-failure", 0, 1, "application build failed"),
            ("application-in-progress", 0, 2, "application build failed"),
            ("application-none", 0, None, "application build failed"),
        )
        for label, platform_result, app_result, message in cases:
            with self.subTest(label=label):
                workspace = self.root / f"workspace-{label}"
                plan = plan_from_environment(
                    self.environment(workspace), repository_root=self.repo
                )
                fake = _FakeVitis(
                    platform_result=platform_result,
                    app_result=app_result,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    execute_plan(plan, vitis_module=fake)
                self.assertEqual(fake.dispose_count, 1)
                self.assertFalse(
                    (workspace / "network_workspace_manifest.json").exists()
                )

    def test_post_build_snapshot_tamper_prevents_manifest(self) -> None:
        workspace = self.root / "workspace"
        plan = plan_from_environment(
            self.environment(workspace), repository_root=self.repo
        )
        fake = _FakeVitis(tamper_snapshot=True)
        with self.assertRaisesRegex(RuntimeError, "provenance re-audit"):
            execute_plan(plan, vitis_module=fake)
        self.assertEqual(fake.dispose_count, 1)
        self.assertFalse(
            (workspace / "network_workspace_manifest.json").exists()
        )

    def test_explicit_web_app_stages_exact_sources_and_audits_elf(self) -> None:
        workspace = self.root / "workspace-web"
        plan = plan_from_environment(
            self.environment(workspace),
            repository_root=self.actual_repo,
            build_web=True,
        )
        self.assertEqual(plan.web_app_name, "a_qweb")
        self.assertIsNotNone(plan.tokenizer_asset)
        self.assertGreater(len(plan.web_sources), 20)

        fake = _FakeVitis()
        manifest = execute_plan(plan, vitis_module=fake)
        self.assertEqual(len(fake.client.app_calls), 2)
        self.assertEqual(fake.client.app_calls[0]["name"], "a_net_echo")
        self.assertEqual(fake.client.app_calls[1]["name"], "a_qweb")
        self.assertEqual(
            fake.client.app_calls[1]["template"], "lwip_echo_server"
        )
        source_dir = workspace / "a_qweb" / "src"
        entry_record = next(
            item
            for item in plan.web_sources
            if item.logical_path.endswith("qweb_board_entry.c")
        )
        self.assertEqual(
            (source_dir / "echo.c").read_bytes(),
            entry_record.source.read_bytes(),
        )
        user_config = (source_dir / "UserConfig.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn('"qweb_lwip_adapter.c"', user_config)
        self.assertIn('"qweb_board_app.c"', user_config)
        self.assertIn('"web_assets.c"', user_config)
        self.assertIn('"tokenizer_asset.S"', user_config)
        linker = (source_dir / "lscript.ld").read_text(encoding="utf-8")
        self.assertIn("_STACK_SIZE : 0x10000", linker)
        self.assertIn("_HEAP_SIZE : 0x10000", linker)
        assembly = (source_dir / "tokenizer_asset.S").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            (source_dir / "qwen3_tokenizer.qtk").as_posix(), assembly
        )
        web_record = manifest["web_application"]
        self.assertEqual(web_record["name"], "a_qweb")
        self.assertEqual(web_record["elf"]["machine"], "AArch64")
        self.assertEqual(manifest["build_results"]["web_application"], 0)
        self.assertGreater(len(web_record["staged_inventory"]), 20)
        persisted = json.loads(
            (workspace / "network_workspace_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            persisted["web_application"]["elf"]["sha256"],
            web_record["elf"]["sha256"],
        )

    def test_web_app_build_failure_never_writes_manifest(self) -> None:
        for result in (1, 2, None):
            with self.subTest(result=result):
                workspace = self.root / f"workspace-web-fail-{result}"
                plan = plan_from_environment(
                    self.environment(workspace),
                    repository_root=self.actual_repo,
                    build_web=True,
                )
                fake = _FakeVitis(web_result=result)
                with self.assertRaisesRegex(
                    RuntimeError, "Web application build failed"
                ):
                    execute_plan(plan, vitis_module=fake)
                self.assertFalse(
                    (workspace / "network_workspace_manifest.json").exists()
                )

    def test_web_source_drift_stops_before_web_component(self) -> None:
        workspace = self.root / "workspace-web-drift"
        plan = plan_from_environment(
            self.environment(workspace),
            repository_root=self.actual_repo,
            build_web=True,
        )
        copied_source = self.root / "drifted-source.c"
        copied_source.write_bytes(plan.web_sources[0].source.read_bytes())
        changed_record = replace(plan.web_sources[0], source=copied_source)
        drifted_plan = replace(
            plan,
            web_sources=(changed_record,) + plan.web_sources[1:],
        )
        copied_source.write_bytes(copied_source.read_bytes() + b"\n")
        fake = _FakeVitis()
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            execute_plan(drifted_plan, vitis_module=fake)
        self.assertEqual(len(fake.client.app_calls), 1)
        self.assertEqual(fake.client.app_calls[0]["name"], "a_net_echo")
        self.assertFalse(
            (workspace / "network_workspace_manifest.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
