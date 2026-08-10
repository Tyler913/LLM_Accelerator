#!/usr/bin/env python3
"""Create an isolated Vitis lwIP-echo workspace from an audited network XSA.

The module deliberately imports ``vitis`` only when a real build is requested,
so its path, provenance, and fail-closed XSA gates remain host-testable with the
repository's normal conda environment.
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
import sys
from typing import Any, Iterable, Mapping

from audit_network_xsa import audit_network_xsa
from web_assets_generator import output_drift


DEFAULT_WORKSPACE = Path(r"F:\vwn")
DEFAULT_PLATFORM_NAME = "p_net"
DEFAULT_APP_NAME = "a_net_echo"
DEFAULT_WEB_APP_NAME = "a_qweb"
DOMAIN_NAME = "standalone_psu_cortexa53_0"
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

WEB_COMPILE_SOURCES = tuple(
    destination
    for _, destination, compile_source in WEB_SOURCE_SPECS
    if compile_source
) + ("tokenizer_asset.S",)


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
            }
        record.pop("web_sources", None)
        return record


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


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
    if actual_compile_sources != WEB_COMPILE_SOURCES:
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
    for record in plan.web_sources:
        _require_regular_source(record.source, "Web application source")
        if (record.source.stat().st_size != record.byte_count or
                _sha256_file(record.source) != record.sha256):
            raise RuntimeError(
                f"Web application source changed after planning: "
                f"{record.logical_path}"
            )


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


def _require_claimed_workspace(workspace: Path, snapshot: Path) -> None:
    if not workspace.is_dir() or not snapshot.is_file():
        raise RuntimeError("claimed workspace or XSA snapshot disappeared")
    entries = list(workspace.iterdir())
    if entries != [snapshot]:
        raise RuntimeError(
            "claimed workspace changed before Vitis took ownership: "
            + ", ".join(str(entry) for entry in entries)
        )


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
    replacement = "set(USER_COMPILE_SOURCES\n" + "".join(
        f'"{name}"\n' for name in WEB_COMPILE_SOURCES
    ) + ")"
    pattern = re.compile(
        r"^set\(USER_COMPILE_SOURCES[ \t]*\r?\n.*?^\)",
        flags=re.MULTILINE | re.DOTALL,
    )
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(
            f"{user_config}: could not locate USER_COMPILE_SOURCES"
        )
    user_config.write_text(updated, encoding="utf-8", newline="\n")


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
) -> list[dict[str, Any]]:
    if plan.web_app_name is None or plan.tokenizer_asset is None:
        raise RuntimeError("cannot populate Web application from an echo-only plan")
    _reaudit_web_inputs(plan)
    source_dir = component_dir / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    user_config = source_dir / "UserConfig.cmake"
    linker_script = source_dir / "lscript.ld"
    _require_regular_source(user_config, "Vitis UserConfig.cmake")
    _require_regular_source(linker_script, "Vitis linker script")

    inventory: list[dict[str, Any]] = []
    for record in plan.web_sources:
        destination = source_dir / record.destination
        if destination.parent != source_dir:
            raise RuntimeError(
                f"invalid Web staging destination: {record.destination}"
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


def _audit_aarch64_elf(path: Path) -> dict[str, Any]:
    _require_regular_source(path, "built Web ELF")
    header = path.read_bytes()[:64]
    if (len(header) < 20 or header[:4] != b"\x7fELF" or
            header[4] != 2 or header[5] != 1 or
            int.from_bytes(header[18:20], "little") != 183):
        raise RuntimeError(f"built Web ELF is not ELF64 little-endian AArch64: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "elf_class": 64,
        "machine": "AArch64",
    }


def execute_plan(
    plan: NetworkWorkspacePlan,
    *,
    vitis_module: Any | None = None,
) -> dict[str, Any]:
    """Execute a previously audited plan in one newly claimed workspace."""

    repository_root = Path(__file__).resolve().parents[3]
    _require_safe_workspace(
        plan.workspace,
        repository_root=repository_root,
        xsa=plan.xsa,
    )
    plan.workspace.mkdir(exist_ok=False)
    snapshot = plan.workspace / f"network_input_{plan.xsa_sha256}.xsa"
    with plan.xsa.open("rb") as source, snapshot.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    _audit_snapshot(
        snapshot,
        xsa_sha256=plan.xsa_sha256,
        bitstream_sha256=plan.bitstream_sha256,
    )
    _require_claimed_workspace(plan.workspace, snapshot)
    vitis_api = (
        importlib.import_module("vitis")
        if vitis_module is None
        else vitis_module
    )
    _require_claimed_workspace(plan.workspace, snapshot)
    client = vitis_api.create_client()
    build_results: dict[str, Any] = {}
    web_inventory: list[dict[str, Any]] = []
    web_elf_record: dict[str, Any] | None = None
    try:
        client.set_workspace(path=plan.workspace.as_posix())
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
        platform_build = platform.build()
        build_results["platform"] = platform_build
        if platform_build != 0:
            raise RuntimeError(
                f"Vitis platform build failed with code {platform_build!r}"
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
    finally:
        vitis_api.dispose()

    _audit_snapshot(
        snapshot,
        xsa_sha256=plan.xsa_sha256,
        bitstream_sha256=plan.bitstream_sha256,
    )
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
