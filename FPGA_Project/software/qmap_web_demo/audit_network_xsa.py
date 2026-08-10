#!/usr/bin/env python3
"""Read-only acceptance gate for the network-enabled QMAP hardware XSA.

The audit never extracts or modifies the archive.  It checks the PS network
configuration and also guards the PL address-map contracts used by the current
one-token board application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable
import xml.etree.ElementTree as ET
import zipfile


SCHEMA_VERSION = 1
MAX_HWH_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024

NETWORK_PARAMETERS = {
    "PSU__ENET3__PERIPHERAL__ENABLE": "1",
    "PSU__ENET3__PERIPHERAL__IO": "MIO 64 .. 75",
    "PSU__ENET3__GRP_MDIO__ENABLE": "1",
    "PSU__ENET3__GRP_MDIO__IO": "MIO 76 .. 77",
    "PSU__TTC0__PERIPHERAL__ENABLE": "1",
    "PSU__TTC0__PERIPHERAL__IO": "NA",
    "PSU__CRL_APB__GEM3_REF_CTRL__FREQMHZ": "125",
    "PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ": "125",
    "PSU__CRL_APB__GEM3_REF_CTRL__SRCSEL": "IOPLL",
}

QMAP_INSTANCE = "qmap_one_token_axi_bd_0"
QMAP_VLNV = "xilinx.com:module_ref:qmap_one_token_axi_bd:1.0"
QMAP_BASE = 0xA0040000
QMAP_HIGH = 0xA004FFFF

STATUS_INSTANCE = "axi_gpio_0"
STATUS_VLNV = "xilinx.com:ip:axi_gpio:2.0"
STATUS_BASE = 0xA0010000
STATUS_HIGH = 0xA001FFFF

DDR_INSTANCE = "ddr4_0"
DDR_VLNV = "xilinx.com:ip:ddr4:2.2"
DDR_BASE = 0x400000000
DDR_HIGH = 0x41FFFFFFF
DDR_RANGE = 0x20000000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_zip_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value, 0)
    except (TypeError, ValueError):
        return None


def _modules(root: ET.Element) -> list[ET.Element]:
    return list(root.findall("./MODULES/MODULE"))


def _find_modules(root: ET.Element, instance: str) -> list[ET.Element]:
    return [module for module in _modules(root) if module.get("INSTANCE") == instance]


def _parameter_map(module: ET.Element, errors: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for parameter in module.findall("./PARAMETERS/PARAMETER"):
        name = parameter.get("NAME")
        value = parameter.get("VALUE")
        if not name:
            errors.append(f"{label}: parameter entry is missing NAME")
            continue
        # Vivado HWH legitimately contains placeholder parameters without a
        # VALUE attribute.  Required values still fail below as absent, while
        # unrelated placeholders must not create false audit failures.
        if value is None:
            continue
        if name in result:
            errors.append(f"{label}: duplicate parameter {name}")
        result[name] = value
    return result


def _unique_module(
    root: ET.Element,
    instance: str,
    errors: list[str],
) -> ET.Element | None:
    matches = _find_modules(root, instance)
    if len(matches) != 1:
        errors.append(f"expected exactly one module {instance}, found {len(matches)}")
        return None
    return matches[0]


def _has_memrange(
    module: ET.Element,
    *,
    instance: str,
    base: int,
    high: int,
    master: str,
    memtype: str,
) -> bool:
    return any(
        entry.get("INSTANCE") == instance
        and _number(entry.get("BASEVALUE")) == base
        and _number(entry.get("HIGHVALUE")) == high
        and entry.get("MASTERBUSINTERFACE") == master
        and entry.get("MEMTYPE") == memtype
        for entry in module.findall("./MEMORYMAP/MEMRANGE")
    )


def _module_identity_ok(module: ET.Element, vlnv: str) -> bool:
    return module.get("VLNV") == vlnv and module.get("IS_ENABLE") == "1"


def _check_datapath(root: ET.Element, errors: list[str]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    ps_modules = [module for module in _modules(root) if module.get("MODTYPE") == "zynq_ultra_ps_e"]
    if len(ps_modules) != 1:
        errors.append(f"expected exactly one enabled Zynq UltraScale+ PS module, found {len(ps_modules)}")
        return facts
    ps = ps_modules[0]
    if ps.get("IS_ENABLE") != "1":
        errors.append("Zynq UltraScale+ PS module is not enabled")

    qmap = _unique_module(root, QMAP_INSTANCE, errors)
    if qmap is not None:
        params = _parameter_map(qmap, errors, QMAP_INSTANCE)
        identity_ok = _module_identity_ok(qmap, QMAP_VLNV)
        address_ok = (
            _number(params.get("C_BASEADDR")) == QMAP_BASE
            and _number(params.get("C_HIGHADDR")) == QMAP_HIGH
        )
        ps_map_ok = _has_memrange(
            ps,
            instance=QMAP_INSTANCE,
            base=QMAP_BASE,
            high=QMAP_HIGH,
            master="M_AXI_HPM0_FPD",
            memtype="REGISTER",
        )
        ddr_map_ok = _has_memrange(
            qmap,
            instance=DDR_INSTANCE,
            base=DDR_BASE,
            high=DDR_HIGH,
            master="M_AXI",
            memtype="MEMORY",
        )
        facts["one_token"] = {
            "instance": QMAP_INSTANCE,
            "identity_ok": identity_ok,
            "base": params.get("C_BASEADDR"),
            "high": params.get("C_HIGHADDR"),
            "address_ok": address_ok,
            "ps_register_map_ok": ps_map_ok,
            "pl_ddr_master_map_ok": ddr_map_ok,
        }
        if not identity_ok:
            errors.append(f"{QMAP_INSTANCE}: expected enabled VLNV {QMAP_VLNV}")
        if not address_ok:
            errors.append(f"{QMAP_INSTANCE}: control aperture is not 0x{QMAP_BASE:X}..0x{QMAP_HIGH:X}")
        if not ps_map_ok:
            errors.append(f"PS memory map is missing the {QMAP_INSTANCE} control aperture")
        if not ddr_map_ok:
            errors.append(f"{QMAP_INSTANCE} memory map is missing the PL DDR aperture")

    status = _unique_module(root, STATUS_INSTANCE, errors)
    if status is not None:
        params = _parameter_map(status, errors, STATUS_INSTANCE)
        identity_ok = _module_identity_ok(status, STATUS_VLNV)
        config_ok = (
            _number(params.get("C_ALL_INPUTS")) == 1
            and _number(params.get("C_GPIO_WIDTH")) == 3
            and _number(params.get("C_IS_DUAL")) == 0
        )
        address_ok = (
            _number(params.get("C_BASEADDR")) == STATUS_BASE
            and _number(params.get("C_HIGHADDR")) == STATUS_HIGH
        )
        ps_map_ok = _has_memrange(
            ps,
            instance=STATUS_INSTANCE,
            base=STATUS_BASE,
            high=STATUS_HIGH,
            master="M_AXI_HPM0_FPD",
            memtype="REGISTER",
        )
        facts["pl_ddr_status"] = {
            "instance": STATUS_INSTANCE,
            "identity_ok": identity_ok,
            "input_width": params.get("C_GPIO_WIDTH"),
            "config_ok": config_ok,
            "base": params.get("C_BASEADDR"),
            "high": params.get("C_HIGHADDR"),
            "address_ok": address_ok,
            "ps_register_map_ok": ps_map_ok,
        }
        if not identity_ok:
            errors.append(f"{STATUS_INSTANCE}: expected enabled VLNV {STATUS_VLNV}")
        if not config_ok:
            errors.append(f"{STATUS_INSTANCE}: expected a single 3-bit all-input GPIO")
        if not address_ok:
            errors.append(f"{STATUS_INSTANCE}: status aperture is not 0x{STATUS_BASE:X}..0x{STATUS_HIGH:X}")
        if not ps_map_ok:
            errors.append(f"PS memory map is missing the {STATUS_INSTANCE} status aperture")

    ddr = _unique_module(root, DDR_INSTANCE, errors)
    if ddr is not None:
        params = _parameter_map(ddr, errors, DDR_INSTANCE)
        identity_ok = _module_identity_ok(ddr, DDR_VLNV)
        address_ok = (
            _number(params.get("C0_DDR4_MEMORY_MAP_BASEADDR")) == DDR_BASE
            and _number(params.get("C0_DDR4_MEMORY_MAP_HIGHADDR")) == DDR_HIGH
        )
        range_ok = any(
            _number(block.get("RANGE")) == DDR_RANGE
            for block in ddr.findall("./ADDRESSBLOCKS/ADDRESSBLOCK")
        )
        ps_map_ok = _has_memrange(
            ps,
            instance=DDR_INSTANCE,
            base=DDR_BASE,
            high=DDR_HIGH,
            master="M_AXI_HPM0_FPD",
            memtype="MEMORY",
        )
        facts["pl_ddr"] = {
            "instance": DDR_INSTANCE,
            "identity_ok": identity_ok,
            "base": params.get("C0_DDR4_MEMORY_MAP_BASEADDR"),
            "high": params.get("C0_DDR4_MEMORY_MAP_HIGHADDR"),
            "address_ok": address_ok,
            "address_block_range_ok": range_ok,
            "ps_memory_map_ok": ps_map_ok,
        }
        if not identity_ok:
            errors.append(f"{DDR_INSTANCE}: expected enabled VLNV {DDR_VLNV}")
        if not address_ok or not range_ok:
            errors.append(f"{DDR_INSTANCE}: aperture is not 0x{DDR_BASE:X}..0x{DDR_HIGH:X} ({DDR_RANGE} bytes)")
        if not ps_map_ok:
            errors.append(f"PS memory map is missing the {DDR_INSTANCE} aperture")

    return facts


def _check_network(root: ET.Element, errors: list[str]) -> dict[str, Any]:
    ps_modules = [module for module in _modules(root) if module.get("MODTYPE") == "zynq_ultra_ps_e"]
    if len(ps_modules) != 1:
        return {"parameters": {}, "instances": {}}
    ps = ps_modules[0]
    params = _parameter_map(ps, errors, "zynq_ultra_ps_e")
    param_facts: dict[str, Any] = {}
    for name, expected in NETWORK_PARAMETERS.items():
        actual = params.get(name)
        match = actual == expected
        param_facts[name] = {"actual": actual, "expected": expected, "match": match}
        if not match:
            errors.append(f"network parameter {name}: expected {expected!r}, found {actual!r}")

    instance_facts: dict[str, Any] = {}
    for instance in ("psu_ethernet_3", "psu_ttc_0"):
        count = len(_find_modules(root, instance))
        instance_facts[instance] = {"count": count, "present_once": count == 1}
        if count != 1:
            errors.append(f"expected exactly one generated network module {instance}, found {count}")

    return {"parameters": param_facts, "instances": instance_facts}


def _read_json_entry(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    errors: list[str],
) -> dict[str, Any] | None:
    if info.file_size > MAX_JSON_BYTES:
        errors.append(f"{info.filename} exceeds the {MAX_JSON_BYTES}-byte metadata limit")
        return None
    try:
        payload = json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        errors.append(f"cannot parse {info.filename}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{info.filename} must contain a JSON object")
        return None
    return payload


def _provenance_from_json(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    devices = payload.get("devices")
    first_device = devices[0] if isinstance(devices, list) and devices and isinstance(devices[0], dict) else {}
    part = first_device.get("part") if isinstance(first_device.get("part"), dict) else {}
    files = payload.get("files")
    declared_bits = []
    if isinstance(files, list):
        declared_bits = [
            item.get("name")
            for item in files
            if isinstance(item, dict) and item.get("type") == "FULL_BIT" and isinstance(item.get("name"), str)
        ]
    return {
        "generated_version": payload.get("generatedVersion"),
        "generated_timestamp": payload.get("generatedTimestamp"),
        "generated_change_list": payload.get("generatedChangeList"),
        "generated_ip_change_list": payload.get("generatedIpChangeList"),
        "top_module_name": payload.get("topModuleName"),
        "fpga_part": first_device.get("fpgaPart"),
        "part_name": part.get("name"),
        "declared_full_bit_entries": declared_bits,
    }


def audit_network_xsa(
    xsa_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit *xsa_path* and return a JSON-serializable, fail-closed report."""

    path = Path(xsa_path).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pass": False,
        "errors": [],
        "warnings": [],
        "archive": {"path": str(path)},
        "hwh": {},
        "network": {},
        "datapath": {},
        "bitstream": {"embedded": False, "entries": []},
        "provenance": {},
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    if not path.is_file():
        errors.append("XSA path is not a regular file")
        return report

    try:
        archive_sha = _sha256_file(path)
        report["archive"].update({"size": path.stat().st_size, "sha256": archive_sha})
    except OSError as exc:
        errors.append(f"cannot read XSA: {exc}")
        return report

    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        report["archive"]["expected_sha256"] = expected
        report["archive"]["sha256_match"] = archive_sha == expected
        if archive_sha != expected:
            errors.append(f"XSA SHA-256 mismatch: expected {expected}, found {archive_sha}")

    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            if duplicate_names:
                errors.append(f"duplicate ZIP entry names: {', '.join(duplicate_names)}")

            hwh_infos = [info for info in infos if info.filename.lower().endswith(".hwh")]
            report["hwh"]["archive_entries"] = [info.filename for info in hwh_infos]
            candidates: list[tuple[zipfile.ZipInfo, bytes, ET.Element]] = []
            for info in hwh_infos:
                if info.file_size > MAX_HWH_BYTES:
                    errors.append(f"{info.filename} exceeds the {MAX_HWH_BYTES}-byte HWH limit")
                    continue
                try:
                    data = archive.read(info)
                    root = ET.fromstring(data)
                except (OSError, ET.ParseError) as exc:
                    errors.append(f"cannot parse HWH {info.filename}: {exc}")
                    continue
                ps_count = sum(module.get("MODTYPE") == "zynq_ultra_ps_e" for module in _modules(root))
                if ps_count == 1:
                    candidates.append((info, data, root))

            if len(candidates) != 1:
                errors.append(f"expected exactly one top-level HWH containing the PS, found {len(candidates)}")
            else:
                hwh_info, hwh_data, root = candidates[0]
                system_info = root.find("./SYSTEMINFO")
                report["hwh"].update(
                    {
                        "entry": hwh_info.filename,
                        "size": hwh_info.file_size,
                        "sha256": hashlib.sha256(hwh_data).hexdigest(),
                        "edw_version": root.get("EDWVERSION"),
                        "vivado_version": root.get("VIVADOVERSION"),
                        "timestamp": root.get("TIMESTAMP"),
                        "system": dict(system_info.attrib) if system_info is not None else {},
                    }
                )
                report["network"] = _check_network(root, errors)
                report["datapath"] = _check_datapath(root, errors)

            bit_infos = [info for info in infos if info.filename.lower().endswith(".bit")]
            bit_entries = [
                {
                    "name": info.filename,
                    "size": info.file_size,
                    "sha256": _sha256_zip_entry(archive, info),
                }
                for info in bit_infos
            ]
            report["bitstream"] = {"embedded": bool(bit_entries), "entries": bit_entries}
            if not bit_entries:
                warnings.append("XSA contains no embedded bitstream")

            json_infos = [info for info in infos if info.filename.lower() == "xsa.json"]
            if len(json_infos) > 1:
                errors.append(f"expected at most one xsa.json, found {len(json_infos)}")
            if json_infos:
                metadata = _read_json_entry(archive, json_infos[0], errors)
                report["provenance"] = _provenance_from_json(metadata)
                report["provenance"]["xsa_json_entry"] = json_infos[0].filename
                declared = set(report["provenance"].get("declared_full_bit_entries", []))
                embedded = {entry["name"] for entry in bit_entries}
                if declared != embedded:
                    warnings.append(
                        "xsa.json FULL_BIT declarations do not exactly match embedded .bit entries"
                    )
            else:
                warnings.append("XSA contains no xsa.json provenance metadata")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        errors.append(f"cannot inspect XSA ZIP archive: {exc}")

    report["pass"] = not errors
    return report


def _format_human(report: dict[str, Any]) -> str:
    verdict = "PASS" if report["pass"] else "FAIL"
    archive = report.get("archive", {})
    hwh = report.get("hwh", {})
    bitstream = report.get("bitstream", {})
    lines = [
        f"{verdict} network XSA audit",
        f"XSA: {archive.get('path')}",
        f"XSA SHA-256: {archive.get('sha256', '<unavailable>')}",
        f"Top HWH: {hwh.get('entry', '<not identified>')}",
        f"HWH provenance: Vivado {hwh.get('vivado_version')} / {hwh.get('timestamp')}",
        f"Embedded bitstream: {bitstream.get('embedded', False)} ({len(bitstream.get('entries', []))} entries)",
    ]
    for entry in bitstream.get("entries", []):
        lines.append(f"  BIT {entry['name']} size={entry['size']} sha256={entry['sha256']}")
    for warning in report.get("warnings", []):
        lines.append(f"WARNING: {warning}")
    for error in report.get("errors", []):
        lines.append(f"ERROR: {error}")
    return "\n".join(lines)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xsa", type=Path, help="XSA archive to inspect (never modified)")
    parser.add_argument("--expect-sha256", help="optional expected archive SHA-256 provenance gate")
    parser.add_argument("--json", action="store_true", help="emit the complete machine-readable report")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_network_xsa(args.xsa, expected_sha256=args.expect_sha256)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
