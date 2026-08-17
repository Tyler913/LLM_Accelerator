from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest

from create_network_vitis_workspace import (
    NetworkWorkspacePlan,
    PINNED_TOKENIZER_BYTES,
    _audit_aarch64_elf,
    _patch_web_cmake,
    execute_plan,
    plan_from_environment,
)
from test_audit_network_xsa import _synthetic_hwh, _write_xsa


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _write_valid_fake_aarch64_web_elf(path: Path) -> None:
    elf_header = struct.Struct("<16sHHIQQQIHHHHHH")
    program_header = struct.Struct("<IIQQQQQQ")
    section_header = struct.Struct("<IIQQQQIIQQ")
    symbol_entry = struct.Struct("<IBBHQQ")
    function_names = (
        "main",
        "start_application",
        "transfer_data",
        "qweb_board_qot_runner",
        "qweb_job_step",
        "qot_session_step",
        "qtk_tokenize_utf8",
    )
    symbol_names = function_names + (
        "qot_tokenizer_asset_start",
        "qot_tokenizer_asset_end",
    )
    shstr = b"\0.text\0.rodata\0.symtab\0.strtab\0.shstrtab\0"
    section_name_offsets = {
        name: shstr.index(name.encode("ascii"))
        for name in (".text", ".rodata", ".symtab", ".strtab", ".shstrtab")
    }
    strtab = bytearray(b"\0")
    symbol_name_offsets: dict[str, int] = {}
    for name in symbol_names:
        symbol_name_offsets[name] = len(strtab)
        strtab.extend(name.encode("ascii") + b"\0")

    program_offset = elf_header.size
    text_offset = 0x100
    text_size = 0x100
    rodata_offset = 0x1000
    rodata_size = PINNED_TOKENIZER_BYTES
    symtab_offset = _align(rodata_offset + rodata_size, 8)
    symbol_count = 1 + len(symbol_names)
    symtab_size = symbol_count * symbol_entry.size
    strtab_offset = symtab_offset + symtab_size
    shstr_offset = strtab_offset + len(strtab)
    section_offset = _align(shstr_offset + len(shstr), 8)
    section_count = 6
    file_size = section_offset + section_count * section_header.size
    image = bytearray(file_size)
    image[text_offset:text_offset + text_size] = b"\x1f\x20\x03\xd5" * (text_size // 4)
    image[strtab_offset:strtab_offset + len(strtab)] = strtab
    image[shstr_offset:shstr_offset + len(shstr)] = shstr

    load_address = 0
    rodata_address = load_address + rodata_offset - text_offset
    ident = b"\x7fELF" + bytes((2, 1, 1, 0)) + bytes(8)
    elf_header.pack_into(
        image,
        0,
        ident,
        2,
        183,
        1,
        load_address,
        program_offset,
        section_offset,
        0,
        elf_header.size,
        program_header.size,
        1,
        section_header.size,
        section_count,
        5,
    )
    program_header.pack_into(
        image,
        program_offset,
        1,
        5,
        text_offset,
        load_address,
        load_address,
        rodata_offset + rodata_size - text_offset,
        rodata_offset + rodata_size - text_offset,
        0x1000,
    )
    sections = (
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (section_name_offsets[".text"], 1, 0x6, load_address,
         text_offset, text_size, 0, 0, 16, 0),
        (section_name_offsets[".rodata"], 1, 0x2, rodata_address,
         rodata_offset, rodata_size, 0, 0, 64, 0),
        (section_name_offsets[".symtab"], 2, 0, 0,
         symtab_offset, symtab_size, 4, 1, 8, symbol_entry.size),
        (section_name_offsets[".strtab"], 3, 0, 0,
         strtab_offset, len(strtab), 0, 0, 1, 0),
        (section_name_offsets[".shstrtab"], 3, 0, 0,
         shstr_offset, len(shstr), 0, 0, 1, 0),
    )
    for index, values in enumerate(sections):
        section_header.pack_into(
            image, section_offset + index * section_header.size, *values
        )

    symbol_offset = symtab_offset + symbol_entry.size
    for index, name in enumerate(function_names):
        symbol_entry.pack_into(
            image,
            symbol_offset,
            symbol_name_offsets[name],
            0x12,
            0,
            1,
            load_address + 0x40 + index * 4,
            4,
        )
        symbol_offset += symbol_entry.size
    symbol_entry.pack_into(
        image,
        symbol_offset,
        symbol_name_offsets["qot_tokenizer_asset_start"],
        0x11,
        0,
        2,
        rodata_address,
        PINNED_TOKENIZER_BYTES,
    )
    symbol_offset += symbol_entry.size
    symbol_entry.pack_into(
        image,
        symbol_offset,
        symbol_name_offsets["qot_tokenizer_asset_end"],
        0x10,
        0,
        2,
        rodata_address + PINNED_TOKENIZER_BYTES,
        0,
    )
    path.write_bytes(image)


class _FakeComponent:
    def __init__(
        self,
        result: int | None,
        elf_path: Path | None = None,
        generated_source: Path | None = None,
    ) -> None:
        self.result = result
        self.elf_path = elf_path
        self.generated_source = generated_source
        self.build_count = 0

    def build(self) -> int | None:
        self.build_count += 1
        if self.generated_source is not None:
            self.generated_source.write_text(
                "generated-after-build\n", encoding="utf-8"
            )
        if self.result == 0 and self.elf_path is not None:
            self.elf_path.parent.mkdir(parents=True, exist_ok=True)
            _write_valid_fake_aarch64_web_elf(self.elf_path)
        return self.result


class _FakeDomain:
    def __init__(self) -> None:
        self.set_lib_calls: list[dict[str, str | None]] = []
        self.update_path_calls: list[dict[str, str]] = []
        self.set_config_calls: list[dict[str, str]] = []

    def set_lib(self, *, lib_name: str, path: str | None = None) -> bool:
        self.set_lib_calls.append({"lib_name": lib_name, "path": path})
        return True

    def update_path(
        self,
        *,
        option: str,
        name: str,
        new_path: str,
    ) -> bool:
        self.update_path_calls.append(
            {"option": option, "name": name, "new_path": new_path}
        )
        return True

    def set_config(
        self,
        *,
        option: str,
        param: str,
        value: str,
        lib_name: str,
    ) -> bool:
        self.set_config_calls.append(
            {
                "option": option,
                "param": param,
                "value": value,
                "lib_name": lib_name,
            }
        )
        return True


class _FakePlatformComponent(_FakeComponent):
    def __init__(self, result: int | None) -> None:
        super().__init__(result)
        self.domain = _FakeDomain()
        self.get_domain_calls: list[str] = []

    def get_domain(self, *, name: str) -> _FakeDomain:
        self.get_domain_calls.append(name)
        return self.domain


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
        self.embedded_sw_repo_calls: list[dict[str, str]] = []
        self.local_embedded_sw_repos: list[str] = []
        self.web_result = web_result
        self.tamper_snapshot = tamper_snapshot
        self.workspace_existed_before_set: bool | None = None
        self.components = {
            "p_net": _FakePlatformComponent(platform_result),
            "a_net_echo": _FakeComponent(app_result),
        }

    def set_workspace(self, *, path: str) -> None:
        workspace = Path(path)
        self.workspace_existed_before_set = workspace.exists()
        workspace.mkdir(exist_ok=False)
        self.workspace = path

    def create_advanced_options_dict(self, **kwargs: str) -> dict[str, str]:
        return dict(kwargs)

    def set_embedded_sw_repo(self, *, level: str, path: str) -> bool:
        self.embedded_sw_repo_calls.append({"level": level, "path": path})
        self.local_embedded_sw_repos = [path]
        return True

    def get_embedded_sw_repo(self, *, level: str) -> list[str]:
        if level != "LOCAL":
            raise RuntimeError("fake client only supports LOCAL SW repositories")
        return list(self.local_embedded_sw_repos)

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
                'set(USER_COMPILE_SOURCES\n'
                '"echo.c"\n'
                '"i2c_access.c"\n'
                '"iic_phyreset.c"\n'
                '"main.c"\n'
                '"platform.c"\n'
                '"platform_mb.c"\n'
                '"platform_zynq.c"\n'
                '"platform_zynqmp.c"\n'
                '"sfp.c"\n'
                '"si5324.c"\n'
                ')\n',
                encoding="utf-8",
            )
            (source_dir / "CMakeLists.txt").write_text(
                "include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)\n"
                f"set(APP_NAME {name})\n"
                "collect (PROJECT_LIB_SOURCES echo.c)\n"
                "list (APPEND _sources ${USER_COMPILE_SOURCES})\n"
                "add_executable(${APP_NAME}.elf ${_sources})\n",
                encoding="utf-8",
            )
            (source_dir / "main.c").write_text(
                "/* lwip_init(); xemac_add(echo_netif netif_set_up(echo_netif); "
                "start_application(); xemacif_input(echo_netif); transfer_data(); */\n",
                encoding="utf-8",
            )
            (source_dir / "platform.c").write_text(
                "int TcpFastTmrFlag; int TcpSlowTmrFlag;\n",
                encoding="utf-8",
            )
            (source_dir / "platform_zynqmp.c").write_text(
                "void platform_setup_timer(void) {}\n",
                encoding="utf-8",
            )
            (source_dir / "lwip_echo_server.cmake").write_text(
                "set(EMACPS_NUM_DRIVER_INSTANCES 1)\n"
                "set(TTCPS_NUM_DRIVER_INSTANCES 1)\n",
                encoding="utf-8",
            )
            (source_dir / "platform_config.h.in").write_text(
                "#cmakedefine PLATFORM_EMAC_BASEADDR @PLATFORM_EMAC_BASEADDR@\n",
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
            generated_source = source_dir / "compile_commands.json"
            generated_source.write_text("generated-before-build\n", encoding="utf-8")
            self.components[name] = _FakeComponent(
                self.web_result,
                component_dir / "build" / f"{name}.elf",
                generated_source,
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
        self.fake_lwip_source = self.root / "fake-lwip220-source"
        self.fake_lwip_source.mkdir()

    @staticmethod
    def _fake_lwip_stager(source: Path, destination: Path) -> dict[str, object]:
        if not source.is_dir():
            raise RuntimeError("fake lwip220 source disappeared")
        patched = (
            destination
            / "src/lwip-2.2.0/contrib/ports/xilinx/netif"
            / "xemacpsif_physpeed.c"
        )
        patched.parent.mkdir(parents=True)
        patched.write_text("fake audited YT8521 source\n", encoding="utf-8")
        return {
            "library": "lwip220",
            "version": "lwip220_v1_2",
            "source_root": str(source.resolve()),
            "source_tree_sha256": "1" * 64,
            "destination": str(destination.resolve()),
            "staged_tree_sha256": "2" * 64,
            "patched_file": (
                "src/lwip-2.2.0/contrib/ports/xilinx/netif/"
                "xemacpsif_physpeed.c"
            ),
            "original_sha256": "3" * 64,
            "patched_sha256": "4" * 64,
            "motorcomm_phy_id": "0x0000011A",
        }

    @staticmethod
    def _fake_lwip_verifier(record: dict[str, object]) -> None:
        destination = Path(str(record["destination"]))
        patched = destination / str(record["patched_file"])
        if patched.read_text(encoding="utf-8") != "fake audited YT8521 source\n":
            raise RuntimeError("fake staged lwip220 source changed")

    @staticmethod
    def _fake_built_lwip_auditor(
        plan: NetworkWorkspacePlan,
        record: dict[str, object],
    ) -> dict[str, object]:
        if record["motorcomm_phy_id"] != "0x0000011A":
            raise RuntimeError("fake BSP override lost its PHY identity")
        return {
            "bsp_copied_source": {
                "path": str(plan.workspace / "fake-bsp-source.c"),
                "sha256": record["patched_sha256"],
                "bytes": 31,
            },
            "export_archive": {
                "path": str(plan.workspace / "fake-liblwip220.a"),
                "sha256": "5" * 64,
                "bytes": 32,
            },
        }

    @staticmethod
    def _fake_echo_elf_auditor(
        plan: NetworkWorkspacePlan,
    ) -> dict[str, object]:
        return {
            "path": str(
                plan.workspace / plan.app_name / "build" / f"{plan.app_name}.elf"
            ),
            "sha256": "6" * 64,
            "bytes": 64,
            "elf_class": 64,
            "machine": "AArch64",
            "required_patch_markers": [
                "Detected Motorcomm YT8521",
                "YT8521 link resolved",
            ],
        }

    def execute_with_fake(
        self,
        plan: NetworkWorkspacePlan,
        fake: _FakeVitis,
    ) -> dict[str, object]:
        return execute_plan(
            plan,
            vitis_module=fake,
            lwip_source_root=self.fake_lwip_source,
            lwip_stager=self._fake_lwip_stager,
            lwip_verifier=self._fake_lwip_verifier,
            built_lwip_auditor=self._fake_built_lwip_auditor,
            echo_elf_auditor=self._fake_echo_elf_auditor,
        )

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
        manifest = self.execute_with_fake(plan, fake)
        self.assertEqual(fake.dispose_count, 1)
        self.assertEqual(fake.client.workspace, workspace.as_posix())
        self.assertFalse(fake.client.workspace_existed_before_set)
        staged_xsa = Path(str(fake.client.platform_args["hw_design"]))
        self.assertEqual(staged_xsa.parent, workspace.resolve())
        self.assertTrue(staged_xsa.name.startswith("network_input_"))
        self.assertEqual(staged_xsa.read_bytes(), self.xsa.read_bytes())
        self.assertEqual(fake.client.platform_args["cpu"], "psu_cortexa53_0")
        self.assertEqual(fake.client.platform_args["architecture"], "64-bit")
        platform = fake.client.components["p_net"]
        self.assertIsInstance(platform, _FakePlatformComponent)
        self.assertEqual(
            platform.get_domain_calls, ["standalone_psu_cortexa53_0"]
        )
        self.assertEqual(
            platform.domain.set_lib_calls,
            [{"lib_name": "lwip220", "path": None}],
        )
        self.assertEqual(platform.domain.update_path_calls, [])
        self.assertEqual(
            fake.client.embedded_sw_repo_calls,
            [
                {
                    "level": "LOCAL",
                    "path": (
                        workspace.resolve() / "software_repo"
                    ).as_posix(),
                }
            ],
        )
        self.assertEqual(
            platform.domain.set_config_calls,
            [
                {
                    "option": "lib",
                    "param": "lwip220_api_mode",
                    "value": "RAW_API",
                    "lib_name": "lwip220",
                },
                {
                    "option": "lib",
                    "param": "lwip220_dhcp",
                    "value": "true",
                    "lib_name": "lwip220",
                },
                {
                    "option": "lib",
                    "param": "lwip220_lwip_dhcp_does_acd_check",
                    "value": "true",
                    "lib_name": "lwip220",
                },
                {
                    "option": "lib",
                    "param": "lwip220_ipv6_enable",
                    "value": "false",
                    "lib_name": "lwip220",
                },
                {
                    "option": "lib",
                    "param": "lwip220_memp_n_tcp_pcb",
                    "value": "256",
                    "lib_name": "lwip220",
                },
                {
                    "option": "lib",
                    "param": "lwip220_pbuf_pool_size",
                    "value": "2048",
                    "lib_name": "lwip220",
                },
                {
                    "option": "lib",
                    "param": "XILTIMER_en_interval_timer",
                    "value": "true",
                    "lib_name": "xiltimer",
                },
            ],
        )
        self.assertEqual(fake.client.app_args["template"], "lwip_echo_server")
        self.assertEqual(fake.client.app_args["domain"], "standalone_psu_cortexa53_0")
        self.assertEqual(manifest["build_results"]["platform"], 0)
        self.assertEqual(manifest["bsp_libraries"], ["lwip220"])
        self.assertEqual(len(manifest["bsp_config"]), 7)
        self.assertEqual(
            manifest["bsp_library_overrides"][0]["motorcomm_phy_id"],
            "0x0000011A",
        )
        self.assertEqual(
            manifest["echo_application"]["elf"]["required_patch_markers"],
            ["Detected Motorcomm YT8521", "YT8521 link resolved"],
        )
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
            self.execute_with_fake(plan, fake)
        self.assertFalse(workspace.exists())
        self.assertEqual(fake.create_count, 0)
        self.assertEqual(fake.dispose_count, 0)

    def test_existing_sibling_claim_blocks_workspace_creation(self) -> None:
        workspace = self.root / "workspace"
        plan = plan_from_environment(
            self.environment(workspace), repository_root=self.repo
        )
        claim = workspace.with_name(f".{workspace.name}.qweb-claim")
        claim.write_text("owned by another process\n", encoding="utf-8")
        fake = _FakeVitis()
        with self.assertRaisesRegex(RuntimeError, "workspace claim already exists"):
            self.execute_with_fake(plan, fake)
        self.assertFalse(workspace.exists())
        self.assertEqual(fake.create_count, 0)
        self.assertTrue(claim.is_file())

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
                    self.execute_with_fake(plan, fake)
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
            self.execute_with_fake(plan, fake)
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
        manifest = self.execute_with_fake(plan, fake)
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
        self.assertIn('"main.c"', user_config)
        self.assertIn('"platform.c"', user_config)
        cmake_text = (source_dir / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        source_appends = tuple(re.finditer(
            r"list\s*\(APPEND\s+_sources\s+\$\{USER_COMPILE_SOURCES\}\s*\)",
            cmake_text,
        ))
        self.assertEqual(len(source_appends), 1)
        self.assertLess(
            source_appends[0].start(),
            cmake_text.index("add_executable(${APP_NAME}.elf ${_sources})"),
        )
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
        self.assertIn(
            "src/CMakeLists.txt",
            {item["path"] for item in web_record["staged_inventory"]},
        )
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
                    self.execute_with_fake(plan, fake)
                self.assertFalse(
                    (workspace / "network_workspace_manifest.json").exists()
                )

    def test_truncated_or_symbol_less_web_elf_is_rejected(self) -> None:
        elf_path = self.root / "invalid-web.elf"
        header = bytearray(64)
        header[:4] = b"\x7fELF"
        header[4] = 2
        header[5] = 1
        header[6] = 1
        header[16:18] = (2).to_bytes(2, "little")
        header[18:20] = (183).to_bytes(2, "little")
        elf_path.write_bytes(header)
        with self.assertRaisesRegex(RuntimeError, "ELF"):
            _audit_aarch64_elf(elf_path)

    def test_template_cmake_must_append_user_sources_to_real_target(self) -> None:
        current = self.root / "current-CMakeLists.txt"
        current.write_text(
            "include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)\n"
            "list (APPEND _sources ${USER_COMPILE_SOURCES})\n"
            "add_executable(${APP_NAME}.elf ${_sources})\n",
            encoding="utf-8",
        )
        _patch_web_cmake(current)
        _patch_web_cmake(current)
        self.assertEqual(
            len(re.findall(
                r"list\s*\(APPEND\s+_sources\s+\$\{USER_COMPILE_SOURCES\}",
                current.read_text(encoding="utf-8"),
            )),
            1,
        )

        legacy = self.root / "legacy-CMakeLists.txt"
        legacy.write_text(
            "include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)\n"
            "collector_list (_sources PROJECT_LIB_SOURCES)\n"
            "add_executable(${APP_NAME}.elf ${_sources})\n",
            encoding="utf-8",
        )
        _patch_web_cmake(legacy)
        self.assertEqual(
            legacy.read_text(encoding="utf-8").count(
                "list(APPEND _sources ${USER_COMPILE_SOURCES})"
            ),
            1,
        )
        invalid = self.root / "invalid-CMakeLists.txt"
        invalid.write_text(
            "include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)\n"
            "add_executable(${APP_NAME}.elf ${_sources})\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "USER_COMPILE_SOURCES append"):
            _patch_web_cmake(invalid)

        duplicate = self.root / "duplicate-CMakeLists.txt"
        duplicate.write_text(
            "include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)\n"
            "list(APPEND _sources ${USER_COMPILE_SOURCES})\n"
            "list (APPEND _sources ${USER_COMPILE_SOURCES})\n"
            "add_executable(${APP_NAME}.elf ${_sources})\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate Web source"):
            _patch_web_cmake(duplicate)

        after_target = self.root / "after-target-CMakeLists.txt"
        after_target.write_text(
            "include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)\n"
            "add_executable(${APP_NAME}.elf ${_sources})\n"
            "list(APPEND _sources ${USER_COMPILE_SOURCES})\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "occurs after"):
            _patch_web_cmake(after_target)

        duplicate_collector = self.root / "duplicate-collector-CMakeLists.txt"
        duplicate_collector.write_text(
            "include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)\n"
            "collector_list(_sources PROJECT_LIB_SOURCES)\n"
            "collector_list (_sources PROJECT_LIB_SOURCES)\n"
            "add_executable(${APP_NAME}.elf ${_sources})\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "unique PROJECT_LIB_SOURCES"):
            _patch_web_cmake(duplicate_collector)

    def test_web_plan_cannot_omit_a_canonical_source(self) -> None:
        workspace = self.root / "workspace-web-missing-source"
        plan = plan_from_environment(
            self.environment(workspace),
            repository_root=self.actual_repo,
            build_web=True,
        )
        incomplete = replace(plan, web_sources=plan.web_sources[:-1])
        fake = _FakeVitis()
        with self.assertRaisesRegex(
            RuntimeError, "source inventory is not canonical"
        ):
            self.execute_with_fake(incomplete, fake)
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
        with self.assertRaisesRegex(RuntimeError, "source inventory is not canonical"):
            self.execute_with_fake(drifted_plan, fake)
        self.assertEqual(len(fake.client.app_calls), 1)
        self.assertEqual(fake.client.app_calls[0]["name"], "a_net_echo")
        self.assertFalse(
            (workspace / "network_workspace_manifest.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
