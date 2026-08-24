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


SCHEMA_VERSION = 2
MAX_HWH_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_BIT_HEADER_BYTES = 64 * 1024

EXPECTED_HWH_SYSTEM = {
    "ARCH": "zynquplus",
    "DEVICE": "xczu2eg",
    "PACKAGE": "sfvc784",
    "SPEEDGRADE": "-2",
}
EXPECTED_PART_NAME = "xczu2eg-sfvc784-2-i"
EXPECTED_FPGA_PART = "zynquplus:xczu2eg:sfvc784:-2:i"
EXPECTED_TOP_MODULE = "llm_system_wrapper"

_BIT_MAGIC = bytes.fromhex("0f f0 0f f0 0f f0 0f f0 00")

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

NETWORK_FLOAT_TOLERANCES = {
    "PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ": 0.01,
}

NETWORK_PS_PERIPHERALS = {
    "psu_ethernet_3": {
        "domain": "LPD",
        "name": "GEM3",
        "base": "FF0E0000",
        "high": "FF0EFFFF",
    },
    "psu_ttc_0": {
        "domain": "LPD",
        "name": "TTC0",
        "base": "FF110000",
        "high": "FF11FFFF",
    },
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


def _parse_bitstream_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    errors: list[str],
) -> dict[str, Any]:
    """Parse and bind the standard Xilinx ``.bit`` header.

    Only the bounded header prefix is read.  The declared configuration payload
    length is checked against the ZIP member's uncompressed size, while the
    complete member is independently streamed through SHA-256 by the caller.
    """

    facts: dict[str, Any] = {"parse_ok": False}
    label = f"embedded bitstream {info.filename}"
    if info.file_size <= 0:
        errors.append(f"{label} is empty")
        return facts

    try:
        with archive.open(info, "r") as stream:
            data = stream.read(min(info.file_size, MAX_BIT_HEADER_BYTES))
    except OSError as exc:
        errors.append(f"cannot read {label} header: {exc}")
        return facts

    offset = 0

    def take(count: int, description: str) -> bytes:
        nonlocal offset
        if count < 0 or offset + count > len(data):
            raise ValueError(f"truncated while reading {description}")
        value = data[offset : offset + count]
        offset += count
        return value

    def u16(description: str) -> int:
        return int.from_bytes(take(2, description), "big")

    def text_field(tag: str) -> str:
        length = u16(f"field {tag} length")
        if length <= 0:
            raise ValueError(f"field {tag} is empty")
        raw = take(length, f"field {tag}")
        if raw[-1:] != b"\0":
            raise ValueError(f"field {tag} is not NUL terminated")
        try:
            return raw[:-1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(f"field {tag} is not ASCII: {exc}") from exc

    try:
        magic_length = u16("magic length")
        magic = take(magic_length, "magic")
        if magic != _BIT_MAGIC:
            raise ValueError("unexpected Xilinx bitstream magic")

        first_tag_length = u16("first tag length")
        if first_tag_length != 1 or take(first_tag_length, "first tag") != b"a":
            raise ValueError("first metadata tag is not 'a'")

        fields: dict[str, str] = {"a": text_field("a")}
        for tag in ("b", "c", "d"):
            if take(1, f"tag {tag}") != tag.encode("ascii"):
                raise ValueError(f"expected metadata tag {tag!r}")
            fields[tag] = text_field(tag)

        if take(1, "payload tag") != b"e":
            raise ValueError("expected configuration payload tag 'e'")
        payload_length = int.from_bytes(take(4, "payload length"), "big")
        payload_offset = offset
        payload_size_matches = payload_length == info.file_size - payload_offset
        if payload_length <= 0:
            raise ValueError("configuration payload is empty")
        if not payload_size_matches:
            raise ValueError(
                "configuration payload length does not match the embedded file size"
            )

        design_name = fields["a"]
        design_top = design_name.split(";", 1)[0]
        part_name = fields["b"]
        facts.update(
            {
                "parse_ok": True,
                "design_name": design_name,
                "design_top": design_top,
                "part_name": part_name,
                "date": fields["c"],
                "time": fields["d"],
                "payload_offset": payload_offset,
                "payload_size": payload_length,
                "payload_size_matches": payload_size_matches,
                "target_part_match": part_name == EXPECTED_PART_NAME,
                "target_top_match": design_top == EXPECTED_TOP_MODULE,
            }
        )
        if part_name != EXPECTED_PART_NAME:
            errors.append(
                f"{label} targets part {part_name!r}, expected {EXPECTED_PART_NAME!r}"
            )
        if design_top != EXPECTED_TOP_MODULE:
            errors.append(
                f"{label} design top is {design_top!r}, expected {EXPECTED_TOP_MODULE!r}"
            )
    except ValueError as exc:
        errors.append(f"cannot parse {label} header: {exc}")

    return facts


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


def _check_hwh_system_identity(root: ET.Element, errors: list[str]) -> dict[str, Any]:
    system_infos = root.findall("./SYSTEMINFO")
    if len(system_infos) != 1:
        errors.append(f"expected exactly one HWH SYSTEMINFO entry, found {len(system_infos)}")
        return {
            "actual": {},
            "expected": dict(EXPECTED_HWH_SYSTEM),
            "fields": {},
            "match": False,
        }

    actual = dict(system_infos[0].attrib)
    fields: dict[str, Any] = {}
    for name, expected in EXPECTED_HWH_SYSTEM.items():
        value = actual.get(name)
        match = value == expected
        fields[name] = {"actual": value, "expected": expected, "match": match}
        if not match:
            errors.append(
                f"HWH SYSTEMINFO {name}: expected {expected!r}, found {value!r}"
            )
    return {
        "actual": actual,
        "expected": dict(EXPECTED_HWH_SYSTEM),
        "fields": fields,
        "match": all(field["match"] for field in fields.values()),
    }


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
        return {"parameters": {}, "ps_peripherals": {}}
    ps = ps_modules[0]
    params = _parameter_map(ps, errors, "zynq_ultra_ps_e")
    param_facts: dict[str, Any] = {}
    for name, expected in NETWORK_PARAMETERS.items():
        actual = params.get(name)
        tolerance = NETWORK_FLOAT_TOLERANCES.get(name)
        if tolerance is None:
            match = actual == expected
        else:
            try:
                match = actual is not None and abs(float(actual) - float(expected)) <= tolerance
            except ValueError:
                match = False
        param_facts[name] = {
            "actual": actual,
            "expected": expected,
            "match": match,
            **({"tolerance": tolerance} if tolerance is not None else {}),
        }
        if not match:
            errors.append(f"network parameter {name}: expected {expected!r}, found {actual!r}")

    protection_value = params.get("PSU__PROTECTION__SLAVES")
    protection_entries: list[dict[str, str]] = []
    if protection_value is None:
        errors.append("zynq_ultra_ps_e parameter PSU__PROTECTION__SLAVES is missing")
    else:
        for raw_entry in protection_value.split("|"):
            fields = raw_entry.split(";")
            if len(fields) != 5:
                errors.append(
                    "PSU__PROTECTION__SLAVES contains a malformed entry: "
                    f"{raw_entry!r}"
                )
                continue
            domain, peripheral, base, high, enabled = fields
            protection_entries.append(
                {
                    "domain": domain,
                    "name": peripheral,
                    "base": base.upper(),
                    "high": high.upper(),
                    "enabled": enabled,
                }
            )

    peripheral_facts: dict[str, Any] = {}
    for instance, expected in NETWORK_PS_PERIPHERALS.items():
        matches = [
            entry
            for entry in protection_entries
            if entry["domain"] == expected["domain"]
            and entry["name"] == expected["name"]
            and entry["base"] == expected["base"]
            and entry["high"] == expected["high"]
            and entry["enabled"] == "1"
        ]
        count = len(matches)
        peripheral_facts[instance] = {
            "protection_map_name": expected["name"],
            "domain": expected["domain"],
            "base": f"0x{expected['base']}",
            "high": f"0x{expected['high']}",
            "enabled": True,
            "matching_entries": count,
            "present_once": count == 1,
        }
        if count != 1:
            errors.append(
                "PS protection map must contain exactly one enabled "
                f"{expected['name']} entry at 0x{expected['base']}..0x{expected['high']}; "
                f"found {count}"
            )

    return {"parameters": param_facts, "ps_peripherals": peripheral_facts}


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


def _check_xsa_json_provenance(
    payload: dict[str, Any],
    bit_entries: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    facts = _provenance_from_json(payload)

    devices = payload.get("devices")
    valid_devices = (
        isinstance(devices, list)
        and len(devices) == 1
        and isinstance(devices[0], dict)
    )
    if not valid_devices:
        count = len(devices) if isinstance(devices, list) else "non-list"
        errors.append(f"xsa.json must declare exactly one device, found {count}")

    top = facts.get("top_module_name")
    top_match = top == EXPECTED_TOP_MODULE
    facts["expected_top_module_name"] = EXPECTED_TOP_MODULE
    facts["top_module_match"] = top_match
    if not top_match:
        errors.append(
            f"xsa.json topModuleName: expected {EXPECTED_TOP_MODULE!r}, found {top!r}"
        )

    part_name = facts.get("part_name")
    part_match = part_name == EXPECTED_PART_NAME
    facts["expected_part_name"] = EXPECTED_PART_NAME
    facts["part_name_match"] = part_match
    if not part_match:
        errors.append(
            f"xsa.json part name: expected {EXPECTED_PART_NAME!r}, found {part_name!r}"
        )

    fpga_part = facts.get("fpga_part")
    fpga_part_match = fpga_part == EXPECTED_FPGA_PART
    facts["expected_fpga_part"] = EXPECTED_FPGA_PART
    facts["fpga_part_match"] = fpga_part_match
    if not fpga_part_match:
        errors.append(
            f"xsa.json fpgaPart: expected {EXPECTED_FPGA_PART!r}, found {fpga_part!r}"
        )

    files = payload.get("files")
    full_bit_items = (
        [item for item in files if isinstance(item, dict) and item.get("type") == "FULL_BIT"]
        if isinstance(files, list)
        else []
    )
    if not isinstance(files, list):
        errors.append("xsa.json files must be a list")
    if len(full_bit_items) != 1:
        errors.append(
            f"xsa.json must declare exactly one FULL_BIT entry, found {len(full_bit_items)}"
        )

    declared = facts.get("declared_full_bit_entries", [])
    embedded = [entry["name"] for entry in bit_entries]
    declaration_match = declared == embedded and len(declared) == 1
    facts["embedded_bit_entries"] = embedded
    facts["full_bit_declaration_match"] = declaration_match
    if not declaration_match:
        errors.append(
            "xsa.json FULL_BIT declarations do not exactly match the unique embedded .bit entry"
        )

    return facts


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
                target = _check_hwh_system_identity(root, errors)
                report["hwh"].update(
                    {
                        "entry": hwh_info.filename,
                        "size": hwh_info.file_size,
                        "sha256": hashlib.sha256(hwh_data).hexdigest(),
                        "edw_version": root.get("EDWVERSION"),
                        "vivado_version": root.get("VIVADOVERSION"),
                        "timestamp": root.get("TIMESTAMP"),
                        "system": target["actual"],
                        "target": target,
                    }
                )
                report["network"] = _check_network(root, errors)
                report["datapath"] = _check_datapath(root, errors)

            bit_infos = [info for info in infos if info.filename.lower().endswith(".bit")]
            if len(bit_infos) != 1:
                errors.append(
                    f"expected exactly one embedded .bit entry, found {len(bit_infos)}"
                )
            bit_entries = []
            for info in bit_infos:
                bit_entries.append(
                    {
                        "name": info.filename,
                        "size": info.file_size,
                        "sha256": _sha256_zip_entry(archive, info),
                        "header": _parse_bitstream_header(archive, info, errors),
                    }
                )
            report["bitstream"] = {"embedded": bool(bit_entries), "entries": bit_entries}

            json_infos = [info for info in infos if info.filename == "xsa.json"]
            if len(json_infos) != 1:
                errors.append(f"expected exactly one xsa.json, found {len(json_infos)}")
            if len(json_infos) == 1:
                metadata = _read_json_entry(archive, json_infos[0], errors)
                report["provenance"] = (
                    _check_xsa_json_provenance(metadata, bit_entries, errors)
                    if metadata is not None
                    else {}
                )
                report["provenance"]["xsa_json_entry"] = json_infos[0].filename
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
