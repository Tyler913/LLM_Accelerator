#!/usr/bin/env python3
"""Create an isolated Vitis network workspace from an audited network XSA.

The module deliberately imports ``vitis`` only when a real build is requested,
so its path, provenance, and fail-closed XSA gates remain host-testable with the
repository's normal conda environment. The default is the independent lwIP echo
gate; explicit Web mode additionally stages and builds the audited ``a_qweb``
application.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
from typing import Any, Callable, Iterable, Mapping

from audit_network_xsa import audit_network_xsa
from web_assets_generator import output_drift
from yt8521_lwip220_patch import (
    LIBRARY_NAME as PATCHED_BSP_LIBRARY,
    LIBRARY_VERSION as PATCHED_BSP_LIBRARY_VERSION,
    PATCH_RELATIVE_PATH as PATCHED_BSP_RELATIVE_PATH,
    stage_patched_library,
    verify_staged_library,
)


DEFAULT_WORKSPACE = Path(r"F:\vwn")
DEFAULT_PLATFORM_NAME = "p_net"
DEFAULT_APP_NAME = "a_net_echo"
DEFAULT_WEB_APP_NAME = "a_qweb"
DOMAIN_NAME = "standalone_psu_cortexa53_0"
REQUIRED_BSP_LIBRARIES = ("lwip220",)
VITIS_LWIP220_RELATIVE_PATH = Path(
    "data/embeddedsw/ThirdParty/sw_services/lwip220_v1_2"
)
# Keep this contract synchronized with AMD Vitis 2025.1.1's
# ``lwip_echo_server.yaml``.  Adding lwip220 alone leaves several defaults
# incompatible with the template, so configure every declared dependency
# explicitly before the platform is generated.
REQUIRED_BSP_CONFIG = (
    ("lwip220", "lwip220_api_mode", "RAW_API"),
    ("lwip220", "lwip220_dhcp", "true"),
    ("lwip220", "lwip220_lwip_dhcp_does_acd_check", "true"),
    ("lwip220", "lwip220_ipv6_enable", "false"),
    # A full28 job uses roughly 50 short-lived HTTP connections while the
    # browser polls.  The lwIP default of 32 active PCBs can therefore be
    # exhausted by FIN/close-pending PCBs before their timers retire them.
    # 256 covers the 120-second worst-case closing window with ample margin.
    ("lwip220", "lwip220_memp_n_tcp_pcb", "256"),
    ("lwip220", "lwip220_pbuf_pool_size", "2048"),
    ("xiltimer", "XILTIMER_en_interval_timer", "true"),
)
COMPONENT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
PROTECTED_WORKSPACES = (Path(r"F:\vwi"), Path(r"F:\vws"))
PINNED_TOKENIZER_SHA256 = (
    "c20242603ef4144e3f3f2ec4ba97c0e9c315aadd41f1bd2c5740e2a7ffa03a7d"
)
PINNED_TOKENIZER_BYTES = 3_629_566
WEB_STACK_BYTES = 0x10000
WEB_HEAP_BYTES = 0x10000


@dataclass(frozen=True)
class WebSourceRecord:
    logical_path: str
    source: Path
    destination: str
    sha256: str
    byte_count: int
    compile_source: bool

    def json_record(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "destination": self.destination,
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "compile_source": self.compile_source,
        }


WEB_SOURCE_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("FPGA_Project/software/qmap_one_token_runtime/qmap_one_token_runtime.h",
     "qmap_one_token_runtime.h", False),
    ("FPGA_Project/software/qmap_one_token_runtime/qmap_one_token_regs.h",
     "qmap_one_token_regs.h", False),
    ("FPGA_Project/software/qmap_one_token_runtime/qmap_model_config_generated.h",
     "qmap_model_config_generated.h", False),
    ("FPGA_Project/software/qmap_prompt_demo/qot_session.c",
     "qot_session.c", True),
    ("FPGA_Project/software/qmap_prompt_demo/qot_session.h",
     "qot_session.h", False),
    ("FPGA_Project/software/qmap_prompt_demo/tokenizer_runtime/qtk_tokenizer_runtime.c",
     "qtk_tokenizer_runtime.c", True),
    ("FPGA_Project/software/qmap_prompt_demo/tokenizer_runtime/qtk_tokenizer_runtime.h",
     "qtk_tokenizer_runtime.h", False),
    ("FPGA_Project/software/qmap_prompt_demo/tokenizer_runtime/qtk_text_tokenizer.c",
     "qtk_text_tokenizer.c", True),
    ("FPGA_Project/software/qmap_prompt_demo/tokenizer_runtime/qtk_text_tokenizer.h",
     "qtk_text_tokenizer.h", False),
    ("FPGA_Project/software/qmap_web_demo/qweb_http.c", "qweb_http.c", True),
    ("FPGA_Project/software/qmap_web_demo/qweb_http.h", "qweb_http.h", False),
    ("FPGA_Project/software/qmap_web_demo/qweb_api.c", "qweb_api.c", True),
    ("FPGA_Project/software/qmap_web_demo/qweb_api.h", "qweb_api.h", False),
    ("FPGA_Project/software/qmap_web_demo/qweb_job.c", "qweb_job.c", True),
    ("FPGA_Project/software/qmap_web_demo/qweb_job.h", "qweb_job.h", False),
    ("FPGA_Project/software/qmap_web_demo/qweb_router.c", "qweb_router.c", True),
    ("FPGA_Project/software/qmap_web_demo/qweb_router.h", "qweb_router.h", False),
    ("FPGA_Project/software/qmap_web_demo/qweb_lwip_adapter.c",
     "qweb_lwip_adapter.c", True),
    ("FPGA_Project/software/qmap_web_demo/qweb_lwip_adapter.h",
     "qweb_lwip_adapter.h", False),
    ("FPGA_Project/software/qmap_web_demo/qweb_board_app.c",
     "qweb_board_app.c", True),
    ("FPGA_Project/software/qmap_web_demo/qweb_board_app.h",
     "qweb_board_app.h", False),
    # AMD's template CMake already compiles echo.c. Replace only its body.
    ("FPGA_Project/software/qmap_web_demo/qweb_board_entry.c",
     "echo.c", False),
    ("FPGA_Project/software/qmap_web_demo/qweb_board_entry.h",
     "qweb_board_entry.h", False),
    ("FPGA_Project/software/qmap_web_demo/web_assets.c",
     "web_assets.c", True),
    ("FPGA_Project/software/qmap_web_demo/web_assets.h",
     "web_assets.h", False),
    ("FPGA_Project/software/qmap_web_demo/web_assets_manifest.json",
     "web_assets_manifest.json", False),
    ("FPGA_Project/software/qmap_web_demo/web/index.html",
     "provenance_index.html", False),
    ("FPGA_Project/software/qmap_web_demo/web/styles.css",
     "provenance_styles.css", False),
    ("FPGA_Project/software/qmap_web_demo/web/app.js",
     "provenance_app.js", False),
    ("FPGA_Project/software/qmap_web_demo/web_assets_generator.py",
     "provenance_web_assets_generator.py", False),
)

VITIS_TEMPLATE_COMPILE_SOURCES = (
    "echo.c",
    "i2c_access.c",
    "iic_phyreset.c",
    "main.c",
    "platform.c",
    "platform_mb.c",
    "platform_zynq.c",
    "platform_zynqmp.c",
    "sfp.c",
    "si5324.c",
)

WEB_ADDED_COMPILE_SOURCES = tuple(
    destination
    for _, destination, compile_source in WEB_SOURCE_SPECS
    if compile_source
) + ("tokenizer_asset.S",)

WEB_COMPILE_SOURCES = (
    VITIS_TEMPLATE_COMPILE_SOURCES + WEB_ADDED_COMPILE_SOURCES
)

# Exact AMD Vitis 2025.1.1 lwip_echo_server template inputs used by the real
# builder. Injected fake-Vitis unit tests exercise the semantic contract below,
# while the CLI path also requires these content hashes before modifying CMake.
PINNED_VITIS_TEMPLATE_SHA256 = {
    "i2c_access.c": "f3bce468a8f4a1b6f6a5885254a217d60deef8a0bad90e6f843905c79f95b6d3",
    "iic_phyreset.c": "cbfa0470bad9465ce781ca32530e839879d9d9626a22f0b40c0d843cb0755b59",
    "lwip_echo_server.cmake": "8c6aa1cc9f76894ec7d481021cd6409f8f591cda584cca74bbbde1901f4d87ec",
    "main.c": "1d6f0c8b05995ae3d934f72fceabb90fb9790f963ecd2bc8c06014883da2e182",
    "platform.c": "1d1ef3d70d50b16dcbac38ee0e6d0ea36aed5d5dcb6f227f3701359a4f690650",
    "platform.h": "c245a181566d148e2cf2dc60be18a2291d71ec3797127a0ff3532d67715e9705",
    "platform_config.h.in": "384dab5e84698c7915b1f9e6a3ae81f2a71efe904a4e2c1bf1222e8d057686d5",
    "platform_zynqmp.c": "76830b48abe799d5804526bef485635fd1c8a1ba668bf2940df0b4fedd91033c",
    "sfp.c": "d23826a3a4ed13770c437aeffa5f3556b45c46a867842a8214070643ea6d5f33",
    "si5324.c": "62e55eed0a135d793a2bcea88b65fbe08d1dfe71f9f0ff8057aea268897c4bef",
}

VITIS_GENERATED_SOURCE_PATHS = frozenset(("compile_commands.json",))
VITIS_GENERATED_SOURCE_PREFIXES = (".compile_commands/",)


def _is_vitis_generated_source(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    return (normalized in VITIS_GENERATED_SOURCE_PATHS or
            any(normalized.startswith(prefix)
                for prefix in VITIS_GENERATED_SOURCE_PREFIXES))


@dataclass(frozen=True)
class NetworkWorkspacePlan:
    xsa: Path
    xsa_sha256: str
    bitstream_sha256: str
    workspace: Path
    platform_name: str
    app_name: str
    web_app_name: str | None = None
    tokenizer_asset: Path | None = None
    tokenizer_sha256: str | None = None
    web_sources: tuple[WebSourceRecord, ...] = ()

    def json_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["xsa"] = str(self.xsa)
        record["workspace"] = str(self.workspace)
        record["domain_name"] = DOMAIN_NAME
        record["bsp_libraries"] = list(REQUIRED_BSP_LIBRARIES)
        record["bsp_config"] = [
            {"library": library, "parameter": parameter, "value": value}
            for library, parameter, value in REQUIRED_BSP_CONFIG
        ]
        record["template"] = "lwip_echo_server"
        record.pop("web_app_name", None)
        record.pop("tokenizer_asset", None)
        record.pop("tokenizer_sha256", None)
        if self.web_app_name is not None:
            record["web_application"] = {
                "name": self.web_app_name,
                "template": "lwip_echo_server",
                "tokenizer_asset": str(self.tokenizer_asset),
                "tokenizer_sha256": self.tokenizer_sha256,
                "stack_bytes": WEB_STACK_BYTES,
                "heap_bytes": WEB_HEAP_BYTES,
                "compile_sources": list(WEB_COMPILE_SOURCES),
                "sources": [item.json_record() for item in self.web_sources],
                "vitis_template_contract": {
                    "name": "lwip_echo_server",
                    "version": "AMD Vitis 2025.1.1",
                    "sha256": dict(PINNED_VITIS_TEMPLATE_SHA256),
                },
            }
        record.pop("web_sources", None)
        return record


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_lwip220_source_root(
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    explicit = values.get("QWEB_VITIS_LWIP220_SOURCE", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
    else:
        vitis_root = values.get("XILINX_VITIS", "").strip()
        if not vitis_root:
            raise RuntimeError(
                "XILINX_VITIS or QWEB_VITIS_LWIP220_SOURCE must identify "
                "the pinned Vitis 2025.1.1 lwip220 source"
            )
        candidate = Path(vitis_root).expanduser() / VITIS_LWIP220_RELATIVE_PATH
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(
            f"Vitis lwip220 source must be a regular directory: {candidate}"
        )
    return candidate.resolve(strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_boolean(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("", "0", "false", "no"):
        return False
    if normalized in ("1", "true", "yes"):
        return True
    raise RuntimeError(f"{label} must be one of 0/1/false/true/no/yes")


def _require_regular_source(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")


def _audit_web_sources(repository_root: Path) -> tuple[WebSourceRecord, ...]:
    web_directory = repository_root / "FPGA_Project/software/qmap_web_demo"
    drift = output_drift(web_directory)
    if drift:
        raise RuntimeError(
            "generated Web assets are stale: " + ", ".join(drift)
        )

    records: list[WebSourceRecord] = []
    destinations: set[str] = set()
    for logical_path, destination, compile_source in WEB_SOURCE_SPECS:
        source = repository_root / Path(logical_path)
        _require_regular_source(source, "Web application source")
        if destination in destinations:
            raise RuntimeError(f"duplicate Web staging destination: {destination}")
        destinations.add(destination)
        records.append(
            WebSourceRecord(
                logical_path=logical_path,
                source=source.resolve(strict=True),
                destination=destination,
                sha256=_sha256_file(source),
                byte_count=source.stat().st_size,
                compile_source=compile_source,
            )
        )
    actual_compile_sources = tuple(
        item.destination for item in records if item.compile_source
    ) + ("tokenizer_asset.S",)
    if actual_compile_sources != WEB_ADDED_COMPILE_SOURCES:
        raise RuntimeError("internal Web compile-source inventory mismatch")
    return tuple(records)


def _audit_tokenizer_asset(path: Path) -> str:
    _require_regular_source(path, "tokenizer asset")
    if path.stat().st_size != PINNED_TOKENIZER_BYTES:
        raise RuntimeError(
            "tokenizer asset size mismatch: "
            f"expected {PINNED_TOKENIZER_BYTES}, got {path.stat().st_size}"
        )
    digest = _sha256_file(path)
    if digest != PINNED_TOKENIZER_SHA256:
        raise RuntimeError(
            "tokenizer asset SHA-256 mismatch: "
            f"expected {PINNED_TOKENIZER_SHA256}, got {digest}"
        )
    return digest


def _reaudit_web_inputs(plan: NetworkWorkspacePlan) -> None:
    if plan.web_app_name is None:
        return
    if plan.tokenizer_asset is None or plan.tokenizer_sha256 is None:
        raise RuntimeError("Web application plan lacks tokenizer provenance")
    if _audit_tokenizer_asset(plan.tokenizer_asset) != plan.tokenizer_sha256:
        raise RuntimeError("tokenizer asset changed after planning")
    canonical = _audit_web_sources(Path(__file__).resolve().parents[3])
    if plan.web_sources != canonical:
        raise RuntimeError("Web application plan source inventory is not canonical")
    for record in plan.web_sources:
        _require_regular_source(record.source, "Web application source")
        if (record.source.stat().st_size != record.byte_count or
                _sha256_file(record.source) != record.sha256):
            raise RuntimeError(
                f"Web application source changed after planning: "
                f"{record.logical_path}"
            )


def _audit_vitis_template_sources(
    source_dir: Path,
    *,
    expected_app_name: str,
    require_pinned_hashes: bool,
) -> tuple[Path, ...]:
    required_fragments: dict[str, tuple[str, ...]] = {
        "CMakeLists.txt": (
            "include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)",
            "add_executable(${APP_NAME}.elf ${_sources})",
        ),
        "main.c": (
            "lwip_init();",
            "xemac_add(echo_netif",
            "netif_set_up(echo_netif);",
            "start_application();",
            "xemacif_input(echo_netif);",
            "transfer_data();",
        ),
        "platform.c": ("TcpFastTmrFlag", "TcpSlowTmrFlag"),
        "platform_zynqmp.c": ("platform_setup_timer",),
        "lwip_echo_server.cmake": (
            "EMACPS_NUM_DRIVER_INSTANCES",
            "TTCPS_NUM_DRIVER_INSTANCES",
        ),
        "platform_config.h.in": ("PLATFORM_EMAC_BASEADDR",),
    }
    regular_files: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Vitis template source must not be a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(source_dir).as_posix()
            if _is_vitis_generated_source(relative):
                continue
            regular_files.append(path)
    for relative, fragments in required_fragments.items():
        path = source_dir / relative
        _require_regular_source(path, "required Vitis lwIP template input")
        text = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                raise RuntimeError(
                    f"{path}: missing required template fragment {fragment!r}"
                )
    cmake_text = (source_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    app_name_assignment = f"set(APP_NAME {expected_app_name})"
    if cmake_text.count(app_name_assignment) != 1:
        raise RuntimeError(
            f"{source_dir / 'CMakeLists.txt'}: expected one generated "
            f"component assignment {app_name_assignment!r}"
        )
    if require_pinned_hashes:
        for relative, expected_sha256 in PINNED_VITIS_TEMPLATE_SHA256.items():
            path = source_dir / relative
            _require_regular_source(path, "pinned Vitis 2025.1.1 template input")
            actual_sha256 = _sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"{path}: Vitis template SHA-256 mismatch; "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
    return tuple(regular_files)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _require_component_name(name: str, label: str) -> str:
    if COMPONENT_NAME.fullmatch(name) is None:
        raise RuntimeError(
            f"{label} must match {COMPONENT_NAME.pattern}, got {name!r}"
        )
    return name


def _require_safe_workspace(
    workspace: Path,
    *,
    repository_root: Path,
    xsa: Path,
) -> None:
    if workspace.parent == workspace:
        raise RuntimeError(f"workspace cannot be a drive root: {workspace}")
    if _overlaps(workspace, repository_root):
        raise RuntimeError(
            f"workspace must not overlap the source repository: {workspace}"
        )
    if _is_within(xsa, workspace) or workspace == xsa:
        raise RuntimeError(f"workspace must not contain the input XSA: {workspace}")
    for protected in PROTECTED_WORKSPACES:
        protected_resolved = _resolved(protected)
        if _overlaps(workspace, protected_resolved):
            raise RuntimeError(
                f"workspace overlaps protected proven workspace {protected_resolved}"
            )
    if workspace.exists():
        raise RuntimeError(f"workspace must not already exist: {workspace}")
    if not workspace.parent.is_dir():
        raise RuntimeError(
            f"workspace parent must already exist: {workspace.parent}"
        )


def _audit_snapshot(
    snapshot: Path,
    *,
    xsa_sha256: str,
    bitstream_sha256: str,
) -> dict[str, Any]:
    report = audit_network_xsa(snapshot, expected_sha256=xsa_sha256)
    bit_entries = report.get("bitstream", {}).get("entries", [])
    if not report.get("pass", False) or len(bit_entries) != 1:
        errors = report.get("errors", [])
        raise RuntimeError(
            "staged network XSA failed provenance re-audit: "
            + "; ".join(str(item) for item in errors)
        )
    if str(bit_entries[0].get("sha256", "")) != bitstream_sha256:
        raise RuntimeError("staged embedded bitstream SHA-256 mismatch")
    return report


def _require_initialized_workspace(workspace: Path, snapshot: Path) -> None:
    if not workspace.is_dir() or not snapshot.is_file():
        raise RuntimeError("initialized Vitis workspace or XSA snapshot disappeared")


def plan_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    repository_root: str | Path | None = None,
    build_web: bool | None = None,
) -> NetworkWorkspacePlan:
    values = os.environ if environ is None else environ
    xsa_text = values.get("QWEB_NETWORK_XSA", "").strip()
    if not xsa_text:
        raise RuntimeError("QWEB_NETWORK_XSA must name the new network XSA")
    xsa = _resolved(xsa_text)
    if not xsa.is_file():
        raise FileNotFoundError(xsa)

    workspace_text = values.get("QWEB_VITIS_WORKSPACE", "").strip()
    workspace = _resolved(workspace_text or DEFAULT_WORKSPACE)
    root = _resolved(
        repository_root
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    _require_safe_workspace(workspace, repository_root=root, xsa=xsa)

    platform_name = _require_component_name(
        values.get("QWEB_VITIS_PLATFORM", DEFAULT_PLATFORM_NAME).strip(),
        "QWEB_VITIS_PLATFORM",
    )
    app_name = _require_component_name(
        values.get("QWEB_VITIS_ECHO_APP", DEFAULT_APP_NAME).strip(),
        "QWEB_VITIS_ECHO_APP",
    )
    if platform_name == app_name:
        raise RuntimeError("platform and application component names must differ")

    if build_web is None:
        build_web = _parse_boolean(
            values.get("QWEB_VITIS_BUILD_WEB", ""),
            "QWEB_VITIS_BUILD_WEB",
        )
    web_app_name: str | None = None
    tokenizer_asset: Path | None = None
    tokenizer_sha256: str | None = None
    web_sources: tuple[WebSourceRecord, ...] = ()
    if build_web:
        web_app_name = _require_component_name(
            values.get("QWEB_VITIS_WEB_APP", DEFAULT_WEB_APP_NAME).strip(),
            "QWEB_VITIS_WEB_APP",
        )
        if web_app_name in (platform_name, app_name):
            raise RuntimeError(
                "platform, echo application, and Web application names "
                "must be distinct"
            )
        tokenizer_text = values.get("QWEB_TOKENIZER_ASSET", "").strip()
        tokenizer_asset = _resolved(
            tokenizer_text
            or root / "Temp/qmap_prompt_demo_tokenizer/qwen3_tokenizer.qtk"
        )
        tokenizer_sha256 = _audit_tokenizer_asset(tokenizer_asset)
        web_sources = _audit_web_sources(root)

    report = audit_network_xsa(xsa)
    if not report.get("pass", False):
        errors = report.get("errors", [])
        detail = "; ".join(str(item) for item in errors)
        raise RuntimeError(f"network XSA audit failed: {detail}")
    bit_entries = report.get("bitstream", {}).get("entries", [])
    if len(bit_entries) != 1:
        raise RuntimeError(
            "network XSA must contain exactly one embedded full bitstream"
        )
    return NetworkWorkspacePlan(
        xsa=xsa,
        xsa_sha256=str(report["archive"]["sha256"]),
        bitstream_sha256=str(bit_entries[0]["sha256"]),
        workspace=workspace,
        platform_name=platform_name,
        app_name=app_name,
        web_app_name=web_app_name,
        tokenizer_asset=tokenizer_asset,
        tokenizer_sha256=tokenizer_sha256,
        web_sources=web_sources,
    )


def _tokenizer_assembly_text(copied_asset: Path) -> str:
    return (
        '.section .rodata.qot_tokenizer_asset,"a",%progbits\n'
        ".balign 64\n"
        ".global qot_tokenizer_asset_start\n"
        ".type qot_tokenizer_asset_start, %object\n"
        "qot_tokenizer_asset_start:\n"
        f'.incbin "{copied_asset.as_posix()}"\n'
        ".global qot_tokenizer_asset_end\n"
        "qot_tokenizer_asset_end:\n"
        ".size qot_tokenizer_asset_start, "
        "qot_tokenizer_asset_end - qot_tokenizer_asset_start\n"
        '.section .note.GNU-stack,"",%progbits\n'
    )


def _replace_cmake_sources(user_config: Path) -> None:
    text = user_config.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^set\(USER_COMPILE_SOURCES[ \t]*\r?\n(.*?)^\)",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(
            f"{user_config}: could not locate USER_COMPILE_SOURCES"
        )
    template_sources = tuple(re.findall(r'^\s*"([^"\r\n]+)"\s*$',
                                        match.group(1), re.MULTILINE))
    if template_sources != VITIS_TEMPLATE_COMPILE_SOURCES:
        raise RuntimeError(
            f"{user_config}: unexpected Vitis template compile sources; "
            f"expected {VITIS_TEMPLATE_COMPILE_SOURCES!r}, "
            f"got {template_sources!r}"
        )
    replacement = "set(USER_COMPILE_SOURCES\n" + "".join(
        f'"{name}"\n' for name in WEB_COMPILE_SOURCES
    ) + ")"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(
            f"{user_config}: could not locate USER_COMPILE_SOURCES"
        )
    user_config.write_text(updated, encoding="utf-8", newline="\n")


def _patch_web_cmake(cmake_file: Path) -> None:
    text = cmake_file.read_text(encoding="utf-8")
    source_marker = "list(APPEND _sources ${USER_COMPILE_SOURCES})"
    source_pattern = re.compile(
        r"^[ \t]*list\s*\(\s*APPEND\s+_sources\s+"
        r"\$\{USER_COMPILE_SOURCES\}\s*\)[ \t]*$",
        flags=re.MULTILINE,
    )
    target_pattern = re.compile(
        r"^[ \t]*add_executable\s*\(\s*\$\{APP_NAME\}\.elf\s+"
        r"\$\{_sources\}\s*\)[ \t]*$",
        flags=re.MULTILINE,
    )
    if text.count("include(${CMAKE_CURRENT_SOURCE_DIR}/UserConfig.cmake)") != 1:
        raise RuntimeError(
            f"{cmake_file}: expected one UserConfig.cmake include"
        )
    matches = tuple(source_pattern.finditer(text))
    if len(matches) > 1:
        raise RuntimeError(f"{cmake_file}: duplicate Web source append markers")
    if matches:
        updated = text
    else:
        collector_pattern = re.compile(
            r"^([ \t]*collector_list\s*\(\s*_sources\s+"
            r"PROJECT_LIB_SOURCES\s*\)[ \t]*)$",
            flags=re.MULTILINE,
        )
        collectors = tuple(collector_pattern.finditer(text))
        if len(collectors) != 1:
            raise RuntimeError(
                f"{cmake_file}: could not locate a USER_COMPILE_SOURCES append "
                "or unique PROJECT_LIB_SOURCES collector"
            )
        updated, count = collector_pattern.subn(
            lambda match: match.group(1) + "\n" + source_marker,
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f"{cmake_file}: could not locate a USER_COMPILE_SOURCES append "
                "or unique PROJECT_LIB_SOURCES collector"
            )
        matches = tuple(source_pattern.finditer(updated))
    targets = tuple(target_pattern.finditer(updated))
    if len(targets) != 1:
        raise RuntimeError(
            f"{cmake_file}: expected one _sources application target"
        )
    if matches[0].start() > targets[0].start():
        raise RuntimeError(
            f"{cmake_file}: USER_COMPILE_SOURCES append occurs after the target"
        )
    if updated != text:
        cmake_file.write_text(updated, encoding="utf-8", newline="\n")


def _patch_web_linker_script(linker_script: Path) -> None:
    text = linker_script.read_text(encoding="utf-8")
    stack_pattern = re.compile(
        r"^_STACK_SIZE = DEFINED\(_STACK_SIZE\) \? _STACK_SIZE : 0x[0-9A-Fa-f]+;$",
        flags=re.MULTILINE,
    )
    heap_pattern = re.compile(
        r"^_HEAP_SIZE = DEFINED\(_HEAP_SIZE\) \? _HEAP_SIZE : 0x[0-9A-Fa-f]+;$",
        flags=re.MULTILINE,
    )
    text, stack_count = stack_pattern.subn(
        f"_STACK_SIZE = DEFINED(_STACK_SIZE) ? _STACK_SIZE : 0x{WEB_STACK_BYTES:X};",
        text,
        count=1,
    )
    text, heap_count = heap_pattern.subn(
        f"_HEAP_SIZE = DEFINED(_HEAP_SIZE) ? _HEAP_SIZE : 0x{WEB_HEAP_BYTES:X};",
        text,
        count=1,
    )
    if stack_count != 1 or heap_count != 1:
        raise RuntimeError(
            f"{linker_script}: could not patch unique stack/heap defaults"
        )
    linker_script.write_text(text, encoding="utf-8", newline="\n")


def _populate_web_application(
    component_dir: Path,
    plan: NetworkWorkspacePlan,
    *,
    require_pinned_template: bool,
) -> list[dict[str, Any]]:
    if plan.web_app_name is None or plan.tokenizer_asset is None:
        raise RuntimeError("cannot populate Web application from an echo-only plan")
    _reaudit_web_inputs(plan)
    source_dir = component_dir / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    template_files = _audit_vitis_template_sources(
        source_dir,
        expected_app_name=plan.web_app_name,
        require_pinned_hashes=require_pinned_template,
    )
    user_config = source_dir / "UserConfig.cmake"
    cmake_file = source_dir / "CMakeLists.txt"
    linker_script = source_dir / "lscript.ld"
    _require_regular_source(user_config, "Vitis UserConfig.cmake")
    _require_regular_source(cmake_file, "Vitis application CMakeLists.txt")
    _require_regular_source(linker_script, "Vitis linker script")

    replaced_template_paths = {
        source_dir / "echo.c",
        user_config,
        cmake_file,
        linker_script,
    }
    inventory: list[dict[str, Any]] = []
    for template_file in template_files:
        if template_file in replaced_template_paths:
            continue
        relative = template_file.relative_to(source_dir).as_posix()
        inventory.append(
            {
                "path": f"src/{relative}",
                "sha256": _sha256_file(template_file),
                "bytes": template_file.stat().st_size,
                "origin": "Vitis lwip_echo_server template input",
            }
        )
    for record in plan.web_sources:
        destination = source_dir / record.destination
        if destination.parent != source_dir:
            raise RuntimeError(
                f"invalid Web staging destination: {record.destination}"
            )
        if destination.exists() and destination != source_dir / "echo.c":
            raise RuntimeError(
                f"Web staging destination collides with Vitis template: "
                f"{record.destination}"
            )
        shutil.copy2(record.source, destination)
        if (destination.stat().st_size != record.byte_count or
                _sha256_file(destination) != record.sha256):
            raise RuntimeError(
                f"staged Web source mismatch: {record.destination}"
            )
        inventory.append(
            {
                "path": f"src/{record.destination}",
                "sha256": record.sha256,
                "bytes": record.byte_count,
                "origin": record.logical_path,
            }
        )

    copied_tokenizer = source_dir / "qwen3_tokenizer.qtk"
    if copied_tokenizer.exists():
        raise RuntimeError("tokenizer staging destination already exists")
    shutil.copy2(plan.tokenizer_asset, copied_tokenizer)
    copied_digest = _audit_tokenizer_asset(copied_tokenizer)
    if copied_digest != plan.tokenizer_sha256:
        raise RuntimeError("staged tokenizer provenance mismatch")
    inventory.append(
        {
            "path": "src/qwen3_tokenizer.qtk",
            "sha256": copied_digest,
            "bytes": copied_tokenizer.stat().st_size,
            "origin": "pinned QTKBPE1 tokenizer asset",
        }
    )

    assembly = source_dir / "tokenizer_asset.S"
    if assembly.exists():
        raise RuntimeError("tokenizer assembly destination already exists")
    assembly.write_text(
        _tokenizer_assembly_text(copied_tokenizer),
        encoding="utf-8",
        newline="\n",
    )
    inventory.append(
        {
            "path": "src/tokenizer_asset.S",
            "sha256": _sha256_file(assembly),
            "bytes": assembly.stat().st_size,
            "origin": "deterministic generated tokenizer incbin wrapper",
        }
    )
    _replace_cmake_sources(user_config)
    _patch_web_cmake(cmake_file)
    _patch_web_linker_script(linker_script)
    inventory.extend(
        (
            {
                "path": "src/UserConfig.cmake",
                "sha256": _sha256_file(user_config),
                "bytes": user_config.stat().st_size,
                "origin": "Vitis template with exact Web source list",
            },
            {
                "path": "src/CMakeLists.txt",
                "sha256": _sha256_file(cmake_file),
                "bytes": cmake_file.stat().st_size,
                "origin": "Vitis template with explicit Web source append",
            },
            {
                "path": "src/lscript.ld",
                "sha256": _sha256_file(linker_script),
                "bytes": linker_script.stat().st_size,
                "origin": "Vitis template with explicit Web stack/heap",
            },
        )
    )
    _reaudit_web_inputs(plan)
    return inventory


def _reaudit_staged_web_application(
    component_dir: Path,
    plan: NetworkWorkspacePlan,
    expected_inventory: Iterable[Mapping[str, Any]],
) -> None:
    _reaudit_web_inputs(plan)
    expected = list(expected_inventory)
    seen: set[str] = set()
    for item in expected:
        relative = str(item["path"])
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"invalid staged Web inventory path: {relative}")
        seen.add(relative)
        staged = component_dir / Path(relative)
        _require_regular_source(staged, "staged Web application file")
        if (staged.stat().st_size != int(item["bytes"]) or
                _sha256_file(staged) != str(item["sha256"])):
            raise RuntimeError(
                f"staged Web application file changed: {relative}"
            )
    source_dir = component_dir / "src"
    actual: set[str] = set()
    for path in source_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"staged Web source must not be a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(source_dir).as_posix()
            if _is_vitis_generated_source(relative):
                continue
            actual.add(f"src/{relative}")
    if actual != seen:
        missing = sorted(seen - actual)
        unexpected = sorted(actual - seen)
        raise RuntimeError(
            "staged Web source inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _audit_aarch64_elf(path: Path) -> dict[str, Any]:
    _require_regular_source(path, "built Web ELF")
    data = path.read_bytes()
    elf_header = struct.Struct("<16sHHIQQQIHHHHHH")
    program_header = struct.Struct("<IIQQQQQQ")
    section_header = struct.Struct("<IIQQQQIIQQ")
    symbol_entry = struct.Struct("<IBBHQQ")

    def require_range(offset: int, size: int, label: str) -> None:
        if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
            raise RuntimeError(f"built Web ELF has invalid {label} range")

    def string_at(table: bytes, offset: int, label: str) -> str:
        if offset < 0 or offset >= len(table):
            raise RuntimeError(f"built Web ELF has invalid {label} string offset")
        terminator = table.find(b"\0", offset)
        if terminator < 0:
            raise RuntimeError(f"built Web ELF has unterminated {label} string")
        try:
            return table[offset:terminator].decode("ascii")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"built Web ELF has non-ASCII {label} string") from exc

    if len(data) < elf_header.size:
        raise RuntimeError(f"built Web ELF is truncated: {path}")
    (
        ident,
        elf_type,
        machine,
        version,
        entry,
        program_offset,
        section_offset,
        _flags,
        header_size,
        program_entry_size,
        program_count,
        section_entry_size,
        section_count,
        section_name_index,
    ) = elf_header.unpack_from(data)
    if (ident[:4] != b"\x7fELF" or ident[4] != 2 or ident[5] != 1 or
            ident[6] != 1 or elf_type != 2 or machine != 183 or version != 1 or
            header_size != elf_header.size):
        raise RuntimeError(
            f"built Web ELF is not an executable ELF64 little-endian AArch64 image: {path}"
        )
    if (program_count == 0 or program_entry_size != program_header.size or
            section_count == 0 or section_entry_size != section_header.size or
            section_name_index == 0 or section_name_index >= section_count):
        raise RuntimeError("built Web ELF has invalid header-table metadata")
    require_range(program_offset,
                  program_count * program_entry_size,
                  "program-header table")
    require_range(section_offset,
                  section_count * section_entry_size,
                  "section-header table")

    executable_entry = False
    for index in range(program_count):
        values = program_header.unpack_from(
            data, program_offset + index * program_entry_size
        )
        segment_type, segment_flags, file_offset, virtual_address, _, file_size, memory_size, _ = values
        if segment_type != 1:
            continue
        if file_size > memory_size:
            raise RuntimeError("built Web ELF has a PT_LOAD file size larger than memory size")
        require_range(file_offset, file_size, "PT_LOAD segment")
        if ((segment_flags & 1) != 0 and file_size != 0 and
                virtual_address <= entry < virtual_address + file_size):
            executable_entry = True
    if not executable_entry:
        raise RuntimeError("built Web ELF entry point is not in an executable PT_LOAD segment")

    sections: list[dict[str, int | str]] = []
    raw_sections: list[tuple[int, ...]] = []
    for index in range(section_count):
        raw_sections.append(
            section_header.unpack_from(
                data, section_offset + index * section_entry_size
            )
        )
    shstr = raw_sections[section_name_index]
    if shstr[1] != 3:
        raise RuntimeError("built Web ELF section-name table is not STRTAB")
    require_range(shstr[4], shstr[5], "section-name string table")
    section_names = data[shstr[4]:shstr[4] + shstr[5]]
    for raw in raw_sections:
        name = string_at(section_names, raw[0], "section") if raw[0] != 0 else ""
        sections.append(
            {
                "name": name,
                "type": raw[1],
                "flags": raw[2],
                "address": raw[3],
                "offset": raw[4],
                "size": raw[5],
                "link": raw[6],
                "entry_size": raw[9],
            }
        )
    text_sections = [
        item for item in sections
        if item["name"] == ".text" and item["type"] == 1 and
        (int(item["flags"]) & 0x6) == 0x6 and int(item["size"]) > 0
    ]
    if len(text_sections) != 1:
        raise RuntimeError("built Web ELF lacks one non-empty executable .text section")
    require_range(int(text_sections[0]["offset"]),
                  int(text_sections[0]["size"]),
                  ".text section")
    executable_sections = [
        item for item in sections
        if (int(item["flags"]) & 0x6) == 0x6 and int(item["size"]) > 0
    ]
    if not any(
        int(item["address"]) <= entry <
        int(item["address"]) + int(item["size"])
        for item in executable_sections
    ):
        raise RuntimeError("built Web ELF entry point is not in an executable section")

    symtabs = [
        (index, item) for index, item in enumerate(sections)
        if item["name"] == ".symtab" and item["type"] == 2
    ]
    if len(symtabs) != 1:
        raise RuntimeError("built Web ELF must contain one .symtab")
    _, symtab = symtabs[0]
    symtab_offset = int(symtab["offset"])
    symtab_size = int(symtab["size"])
    symtab_entry_size = int(symtab["entry_size"])
    string_index = int(symtab["link"])
    if (symtab_entry_size != symbol_entry.size or
            symtab_size == 0 or symtab_size % symbol_entry.size != 0 or
            string_index <= 0 or string_index >= section_count or
            sections[string_index]["type"] != 3):
        raise RuntimeError("built Web ELF has invalid .symtab metadata")
    require_range(symtab_offset, symtab_size, ".symtab")
    string_section = sections[string_index]
    require_range(int(string_section["offset"]),
                  int(string_section["size"]),
                  "symbol string table")
    symbol_strings = data[
        int(string_section["offset"]):
        int(string_section["offset"]) + int(string_section["size"])
    ]
    symbols: dict[str, tuple[int, int, int]] = {}
    for offset in range(symtab_offset, symtab_offset + symtab_size, symbol_entry.size):
        name_offset, info, _, section_index, value, size = symbol_entry.unpack_from(data, offset)
        if name_offset == 0:
            continue
        name = string_at(symbol_strings, name_offset, "symbol")
        symbols[name] = (section_index, value, size)

    required_functions = (
        "main",
        "start_application",
        "transfer_data",
        "qweb_board_qot_runner",
        "qweb_job_step",
        "qot_session_step",
        "qtk_tokenize_utf8",
    )
    for name in required_functions:
        symbol = symbols.get(name)
        if symbol is None:
            raise RuntimeError(f"built Web ELF is missing required symbol {name}")
        section_index, value, _ = symbol
        if (section_index <= 0 or section_index >= section_count or value == 0 or
                (int(sections[section_index]["flags"]) & 0x6) != 0x6):
            raise RuntimeError(f"built Web ELF symbol {name} is not executable code")

    tokenizer_start = symbols.get("qot_tokenizer_asset_start")
    tokenizer_end = symbols.get("qot_tokenizer_asset_end")
    if tokenizer_start is None or tokenizer_end is None:
        raise RuntimeError("built Web ELF is missing tokenizer boundary symbols")
    start_section, start_value, _ = tokenizer_start
    end_section, end_value, _ = tokenizer_end
    if (start_section <= 0 or start_section >= section_count or
            end_section != start_section or
            end_value - start_value != PINNED_TOKENIZER_BYTES):
        raise RuntimeError("built Web ELF tokenizer symbol span is invalid")
    tokenizer_section = sections[start_section]
    section_start = int(tokenizer_section["address"])
    section_end = section_start + int(tokenizer_section["size"])
    if (tokenizer_section["type"] != 1 or
            (int(tokenizer_section["flags"]) & 0x2) == 0 or
            start_value < section_start or end_value > section_end):
        raise RuntimeError("built Web ELF tokenizer is not file-backed allocated data")
    require_range(int(tokenizer_section["offset"]),
                  int(tokenizer_section["size"]),
                  "tokenizer section")

    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": len(data),
        "elf_class": 64,
        "machine": "AArch64",
        "elf_type": "ET_EXEC",
        "entry": entry,
        "section_count": section_count,
        "required_symbols": list(required_functions),
        "tokenizer_bytes": end_value - start_value,
    }


def _audit_built_lwip_override(
    plan: NetworkWorkspacePlan,
    override: dict[str, Any],
) -> dict[str, Any]:
    expected_source_hash = str(override.get("patched_sha256", ""))
    if len(expected_source_hash) != 64:
        raise RuntimeError("BSP override lacks a patched-source SHA-256")
    bsp_root = (
        plan.workspace
        / plan.platform_name
        / "psu_cortexa53_0"
        / DOMAIN_NAME
        / "bsp"
    )
    copied_source = (
        bsp_root
        / "libsrc"
        / PATCHED_BSP_LIBRARY
        / PATCHED_BSP_RELATIVE_PATH
    )
    _require_regular_source(copied_source, "Vitis BSP copied lwip220 source")
    copied_hash = _sha256_file(copied_source)
    if copied_hash != expected_source_hash:
        raise RuntimeError(
            "Vitis BSP did not adopt the audited YT8521 lwip220 source: "
            f"expected {expected_source_hash}, got {copied_hash}"
        )
    archive = (
        plan.workspace
        / plan.platform_name
        / "export"
        / plan.platform_name
        / "sw"
        / DOMAIN_NAME
        / "lib"
        / f"lib{PATCHED_BSP_LIBRARY}.a"
    )
    _require_regular_source(archive, "exported patched lwip220 archive")
    with archive.open("rb") as stream:
        archive_magic = stream.read(8)
    if archive_magic != b"!<arch>\n":
        raise RuntimeError(f"exported lwip220 output is not an archive: {archive}")
    return {
        "bsp_copied_source": {
            "path": str(copied_source),
            "sha256": copied_hash,
            "bytes": copied_source.stat().st_size,
        },
        "export_archive": {
            "path": str(archive),
            "sha256": _sha256_file(archive),
            "bytes": archive.stat().st_size,
        },
    }


ECHO_PATCH_MARKERS = (
    "Detected Motorcomm YT8521",
    "YT8521 link resolved",
)


def _audit_echo_elf(plan: NetworkWorkspacePlan) -> dict[str, Any]:
    path = plan.workspace / plan.app_name / "build" / f"{plan.app_name}.elf"
    _require_regular_source(path, "built network echo ELF")
    data = path.read_bytes()
    if (len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or
            data[5] != 1 or int.from_bytes(data[18:20], "little") != 183):
        raise RuntimeError(
            f"built network echo image is not ELF64 little-endian AArch64: {path}"
        )
    missing = [marker for marker in ECHO_PATCH_MARKERS
               if marker.encode("ascii") not in data]
    if missing:
        raise RuntimeError(
            "built network echo ELF does not contain the YT8521 BSP patch: "
            + ", ".join(missing)
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": len(data),
        "elf_class": 64,
        "machine": "AArch64",
        "required_patch_markers": list(ECHO_PATCH_MARKERS),
    }


def execute_plan(
    plan: NetworkWorkspacePlan,
    *,
    vitis_module: Any | None = None,
    lwip_source_root: str | Path | None = None,
    lwip_stager: Callable[[Path, Path], dict[str, Any]] = stage_patched_library,
    lwip_verifier: Callable[[dict[str, Any]], None] = verify_staged_library,
    built_lwip_auditor: Callable[
        [NetworkWorkspacePlan, dict[str, Any]], dict[str, Any]
    ] = _audit_built_lwip_override,
    echo_elf_auditor: Callable[
        [NetworkWorkspacePlan], dict[str, Any]
    ] = _audit_echo_elf,
) -> dict[str, Any]:
    """Execute a previously audited plan in one newly claimed workspace."""

    repository_root = Path(__file__).resolve().parents[3]
    _require_safe_workspace(
        plan.workspace,
        repository_root=repository_root,
        xsa=plan.xsa,
    )
    # Re-audit the original before claiming anything.  This catches a changed
    # XSA without leaving a half-created workspace behind.
    _audit_snapshot(
        plan.xsa,
        xsa_sha256=plan.xsa_sha256,
        bitstream_sha256=plan.bitstream_sha256,
    )
    if lwip_source_root is None:
        resolved_lwip_source = _resolve_lwip220_source_root()
    else:
        resolved_lwip_source = Path(lwip_source_root).expanduser().resolve(
            strict=True
        )

    # Vitis 2025.1 requires a new workspace path to be absent when
    # set_workspace() is called.  Claim the name with an exclusive sibling
    # lock instead of pre-creating an empty directory that Vitis rejects as an
    # unrecognized legacy workspace.
    claim = plan.workspace.with_name(f".{plan.workspace.name}.qweb-claim")
    try:
        with claim.open("x", encoding="utf-8") as stream:
            stream.write(
                f"workspace={plan.workspace}\n"
                f"xsa_sha256={plan.xsa_sha256}\n"
                f"pid={os.getpid()}\n"
            )
    except FileExistsError as exc:
        raise RuntimeError(f"workspace claim already exists: {claim}") from exc

    build_results: dict[str, Any] = {}
    bsp_override_record: dict[str, Any] | None = None
    built_lwip_record: dict[str, Any] | None = None
    echo_elf_record: dict[str, Any] | None = None
    web_inventory: list[dict[str, Any]] = []
    web_elf_record: dict[str, Any] | None = None
    try:
        _require_safe_workspace(
            plan.workspace,
            repository_root=repository_root,
            xsa=plan.xsa,
        )
        vitis_api = (
            importlib.import_module("vitis")
            if vitis_module is None
            else vitis_module
        )
        client = vitis_api.create_client()
        try:
            client.set_workspace(path=plan.workspace.as_posix())
            if not plan.workspace.is_dir():
                raise RuntimeError("Vitis did not initialize the requested workspace")
            snapshot = plan.workspace / f"network_input_{plan.xsa_sha256}.xsa"
            with plan.xsa.open("rb") as source, snapshot.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            _audit_snapshot(
                snapshot,
                xsa_sha256=plan.xsa_sha256,
                bitstream_sha256=plan.bitstream_sha256,
            )
            _require_initialized_workspace(plan.workspace, snapshot)
            embedded_repo_root = plan.workspace / "software_repo"
            bsp_override_record = lwip_stager(
                resolved_lwip_source,
                embedded_repo_root
                / "ThirdParty"
                / "sw_services"
                / PATCHED_BSP_LIBRARY_VERSION,
            )
            bsp_override_record = dict(bsp_override_record)
            bsp_override_record["repository_root"] = str(embedded_repo_root)
            lwip_verifier(bsp_override_record)
            embedded_repo_text = embedded_repo_root.as_posix()
            if client.set_embedded_sw_repo(
                level="LOCAL",
                path=embedded_repo_text,
            ) is not True:
                raise RuntimeError(
                    "Vitis failed to register the isolated embedded SW repository"
                )
            configured_repositories = client.get_embedded_sw_repo(level="LOCAL")
            if not any(
                Path(str(item)).resolve() == embedded_repo_root.resolve()
                for item in configured_repositories
            ):
                raise RuntimeError(
                    "Vitis did not retain the isolated embedded SW repository"
                )
            advanced = client.create_advanced_options_dict(dt_overlay="0")
            client.create_platform_component(
                name=plan.platform_name,
                hw_design=str(snapshot),
                os="standalone",
                cpu="psu_cortexa53_0",
                domain_name=DOMAIN_NAME,
                generate_dtb=False,
                advanced_options=advanced,
                architecture="64-bit",
                compiler="gcc",
            )
            platform = client.get_component(name=plan.platform_name)
            domain = platform.get_domain(name=DOMAIN_NAME)
            for library in REQUIRED_BSP_LIBRARIES:
                set_library = domain.set_lib(lib_name=library)
                if set_library is not True:
                    raise RuntimeError(
                        f"Vitis failed to enable required BSP library {library!r}"
                    )
            for library, parameter, value in REQUIRED_BSP_CONFIG:
                if domain.set_config(
                    option="lib",
                    param=parameter,
                    value=value,
                    lib_name=library,
                ) is not True:
                    raise RuntimeError(
                        "Vitis failed to set required BSP parameter "
                        f"{library}.{parameter}={value}"
                    )
            platform_build = platform.build()
            build_results["platform"] = platform_build
            if platform_build != 0:
                raise RuntimeError(
                    f"Vitis platform build failed with code {platform_build!r}"
                )
            built_lwip_record = built_lwip_auditor(
                plan,
                bsp_override_record,
            )
            xpfm = (
                f"$COMPONENT_LOCATION/../{plan.platform_name}/export/"
                f"{plan.platform_name}/{plan.platform_name}.xpfm"
            )
            client.create_app_component(
                name=plan.app_name,
                platform=xpfm,
                domain=DOMAIN_NAME,
                template="lwip_echo_server",
            )
            application = client.get_component(name=plan.app_name)
            application_build = application.build()
            build_results["application"] = application_build
            if application_build != 0:
                raise RuntimeError(
                    f"Vitis echo application build failed with code {application_build!r}"
                )
            echo_elf_record = echo_elf_auditor(plan)
            if plan.web_app_name is not None:
                _reaudit_web_inputs(plan)
                client.create_app_component(
                    name=plan.web_app_name,
                    platform=xpfm,
                    domain=DOMAIN_NAME,
                    template="lwip_echo_server",
                )
                web_application = client.get_component(name=plan.web_app_name)
                web_component_dir = plan.workspace / plan.web_app_name
                web_inventory = _populate_web_application(
                    web_component_dir,
                    plan,
                    require_pinned_template=vitis_module is None,
                )
                web_build = web_application.build()
                build_results["web_application"] = web_build
                if web_build != 0:
                    raise RuntimeError(
                        f"Vitis Web application build failed with code {web_build!r}"
                    )
                _reaudit_staged_web_application(
                    web_component_dir,
                    plan,
                    web_inventory,
                )
                web_elf_record = _audit_aarch64_elf(
                    web_component_dir / "build" / f"{plan.web_app_name}.elf"
                )
        except (OSError, RuntimeError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Vitis API operation failed: {exc}") from exc
        finally:
            vitis_api.dispose()
    finally:
        claim.unlink(missing_ok=True)

    _audit_snapshot(
        snapshot,
        xsa_sha256=plan.xsa_sha256,
        bitstream_sha256=plan.bitstream_sha256,
    )
    if bsp_override_record is None:
        raise RuntimeError("Vitis build produced no audited BSP library override")
    lwip_verifier(bsp_override_record)
    if built_lwip_record is None:
        raise RuntimeError("Vitis build produced no audited lwip220 output")
    if built_lwip_auditor(plan, bsp_override_record) != built_lwip_record:
        raise RuntimeError("built lwip220 output changed before manifest creation")
    if echo_elf_record is None:
        raise RuntimeError("Vitis build produced no audited network echo ELF")
    if echo_elf_auditor(plan) != echo_elf_record:
        raise RuntimeError("network echo ELF changed before manifest creation")
    if plan.web_app_name is not None:
        _reaudit_staged_web_application(
            plan.workspace / plan.web_app_name,
            plan,
            web_inventory,
        )
        if web_elf_record is None:
            raise RuntimeError("Web application build produced no audited ELF")
        current_elf = _audit_aarch64_elf(Path(str(web_elf_record["path"])))
        if current_elf != web_elf_record:
            raise RuntimeError("built Web ELF changed before manifest creation")
    manifest = plan.json_record()
    manifest["xsa_snapshot"] = str(snapshot)
    manifest["build_results"] = build_results
    bsp_override_record = dict(bsp_override_record)
    bsp_override_record["build_outputs"] = built_lwip_record
    manifest["bsp_library_overrides"] = [bsp_override_record]
    manifest["echo_application"] = {
        "name": plan.app_name,
        "elf": echo_elf_record,
    }
    if plan.web_app_name is not None:
        manifest["web_application"]["staged_inventory"] = web_inventory
        manifest["web_application"]["elf"] = web_elf_record
    manifest_path = plan.workspace / "network_workspace_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="audit paths and XSA without importing Vitis or creating files",
    )
    parser.add_argument(
        "--with-web-app",
        action="store_true",
        help=(
            "also audit, stage, and build the app-owned a_qweb component; "
            "the default remains the isolated AMD echo gate"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan = plan_from_environment(
            build_web=True if args.with_web_app else None
        )
        result = plan.json_record() if args.check_only else execute_plan(plan)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        action = "checked" if args.check_only else "built"
        print(f"PASS {action} isolated network Vitis workspace")
        print(f"  XSA: {result['xsa']}")
        print(f"  workspace: {result['workspace']}")
        print(f"  components: {result['platform_name']} / {result['app_name']}")
        if "web_application" in result:
            print(f"  Web component: {result['web_application']['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
