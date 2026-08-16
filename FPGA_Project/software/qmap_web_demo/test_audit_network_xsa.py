from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from audit_network_xsa import (
    DDR_BASE,
    DDR_HIGH,
    DDR_RANGE,
    EXPECTED_FPGA_PART,
    EXPECTED_PART_NAME,
    EXPECTED_TOP_MODULE,
    NETWORK_PARAMETERS,
    NETWORK_PS_PERIPHERALS,
    QMAP_BASE,
    QMAP_HIGH,
    STATUS_BASE,
    STATUS_HIGH,
    audit_network_xsa,
)


def _parameters(module: ET.Element, values: dict[str, str]) -> None:
    container = ET.SubElement(module, "PARAMETERS")
    for name, value in values.items():
        ET.SubElement(container, "PARAMETER", NAME=name, VALUE=str(value))


def _memrange(
    module: ET.Element,
    instance: str,
    base: int,
    high: int,
    master: str,
    memtype: str,
) -> None:
    memory_map = module.find("./MEMORYMAP")
    if memory_map is None:
        memory_map = ET.SubElement(module, "MEMORYMAP")
    ET.SubElement(
        memory_map,
        "MEMRANGE",
        INSTANCE=instance,
        BASEVALUE=f"0x{base:X}",
        HIGHVALUE=f"0x{high:X}",
        MASTERBUSINTERFACE=master,
        MEMTYPE=memtype,
    )


def _synthetic_hwh(
    *,
    system_overrides: dict[str, str] | None = None,
    network_overrides: dict[str, str] | None = None,
    omit_network_peripherals: tuple[str, ...] = (),
    include_qmap: bool = True,
    qmap_base: int = QMAP_BASE,
    ps_qmap_base: int = QMAP_BASE,
    qmap_ddr_high: int = DDR_HIGH,
    status_width: int = 3,
    status_base: int = STATUS_BASE,
    ps_status_base: int = STATUS_BASE,
    ddr_high: int = DDR_HIGH,
    ddr_range: int = DDR_RANGE,
    ps_ddr_high: int = DDR_HIGH,
) -> bytes:
    root = ET.Element(
        "EDKSYSTEM",
        EDWVERSION="1.2",
        VIVADOVERSION="2025.1.1",
        TIMESTAMP="synthetic test fixture",
    )
    system = {
        "ARCH": "zynquplus",
        "DEVICE": "xczu2eg",
        "NAME": "llm_system",
        "PACKAGE": "sfvc784",
        "SPEEDGRADE": "-2",
    }
    system.update(system_overrides or {})
    ET.SubElement(root, "SYSTEMINFO", **system)
    modules = ET.SubElement(root, "MODULES")

    ps = ET.SubElement(
        modules,
        "MODULE",
        INSTANCE="zynq_ultra_ps_e_0",
        MODTYPE="zynq_ultra_ps_e",
        VLNV="xilinx.com:ip:zynq_ultra_ps_e:3.5",
        IS_ENABLE="1",
    )
    network = dict(NETWORK_PARAMETERS)
    network.update(network_overrides or {})
    protection_entries = []
    for instance, peripheral in NETWORK_PS_PERIPHERALS.items():
        if instance not in omit_network_peripherals:
            protection_entries.append(
                ";".join(
                    (
                        peripheral["domain"],
                        peripheral["name"],
                        peripheral["base"],
                        peripheral["high"],
                        "1",
                    )
                )
            )
    network["PSU__PROTECTION__SLAVES"] = "|".join(protection_entries)
    _parameters(ps, network)
    if include_qmap:
        _memrange(ps, "qmap_one_token_axi_bd_0", ps_qmap_base, QMAP_HIGH, "M_AXI_HPM0_FPD", "REGISTER")
    _memrange(ps, "axi_gpio_0", ps_status_base, STATUS_HIGH, "M_AXI_HPM0_FPD", "REGISTER")
    _memrange(ps, "ddr4_0", DDR_BASE, ps_ddr_high, "M_AXI_HPM0_FPD", "MEMORY")

    if include_qmap:
        qmap = ET.SubElement(
            modules,
            "MODULE",
            INSTANCE="qmap_one_token_axi_bd_0",
            MODTYPE="qmap_one_token_axi_bd",
            VLNV="xilinx.com:module_ref:qmap_one_token_axi_bd:1.0",
            IS_ENABLE="1",
        )
        _parameters(qmap, {"C_BASEADDR": f"0x{qmap_base:X}", "C_HIGHADDR": f"0x{QMAP_HIGH:X}"})
        _memrange(qmap, "ddr4_0", DDR_BASE, qmap_ddr_high, "M_AXI", "MEMORY")

    status = ET.SubElement(
        modules,
        "MODULE",
        INSTANCE="axi_gpio_0",
        MODTYPE="axi_gpio",
        VLNV="xilinx.com:ip:axi_gpio:2.0",
        IS_ENABLE="1",
    )
    _parameters(
        status,
        {
            "C_ALL_INPUTS": "1",
            "C_GPIO_WIDTH": str(status_width),
            "C_IS_DUAL": "0",
            "C_BASEADDR": f"0x{status_base:X}",
            "C_HIGHADDR": f"0x{STATUS_HIGH:X}",
        },
    )

    ddr = ET.SubElement(
        modules,
        "MODULE",
        INSTANCE="ddr4_0",
        MODTYPE="ddr4",
        VLNV="xilinx.com:ip:ddr4:2.2",
        IS_ENABLE="1",
    )
    _parameters(
        ddr,
        {
            "C0_DDR4_MEMORY_MAP_BASEADDR": f"0x{DDR_BASE:X}",
            "C0_DDR4_MEMORY_MAP_HIGHADDR": f"0x{ddr_high:X}",
        },
    )
    blocks = ET.SubElement(ddr, "ADDRESSBLOCKS")
    ET.SubElement(blocks, "ADDRESSBLOCK", NAME="C0_DDR4_ADDRESS_BLOCK", RANGE=str(ddr_range))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _synthetic_bitstream(
    *,
    part_name: str = EXPECTED_PART_NAME,
    top_module: str = EXPECTED_TOP_MODULE,
    payload: bytes = b"synthetic-configuration-payload",
) -> bytes:
    magic = bytes.fromhex("0f f0 0f f0 0f f0 0f f0 00")

    def sized(value: bytes) -> bytes:
        return len(value).to_bytes(2, "big") + value

    design = f"{top_module};Version=synthetic".encode("ascii") + b"\0"
    part = part_name.encode("ascii") + b"\0"
    date = b"2026/08/11\0"
    time = b"12:00:00\0"
    return b"".join(
        (
            sized(magic),
            sized(b"a"),
            sized(design),
            b"b",
            sized(part),
            b"c",
            sized(date),
            b"d",
            sized(time),
            b"e",
            len(payload).to_bytes(4, "big"),
            payload,
        )
    )


def _write_xsa(
    path: Path,
    hwh: bytes,
    *,
    include_bit: bool = True,
    extra_top_hwh: bytes | None = None,
    malformed_json: bool = False,
    include_json: bool = True,
    bit_name: str = "llm_system_wrapper.bit",
    bit_contents: bytes | None = None,
    bit_part_name: str = EXPECTED_PART_NAME,
    bit_top_module: str = EXPECTED_TOP_MODULE,
    extra_bit: tuple[str, bytes] | None = None,
    json_part_name: str = EXPECTED_PART_NAME,
    json_fpga_part: str = EXPECTED_FPGA_PART,
    json_top_module: str = EXPECTED_TOP_MODULE,
    declared_bit_entries: list[str] | None = None,
) -> bytes:
    bit_payload = (
        _synthetic_bitstream(part_name=bit_part_name, top_module=bit_top_module)
        if bit_contents is None
        else bit_contents
    )
    declarations = (
        ([bit_name] if include_bit else [])
        if declared_bit_entries is None
        else declared_bit_entries
    )
    metadata = {
        "topModuleName": json_top_module,
        "generatedVersion": "2025.1.1",
        "generatedTimestamp": "synthetic test fixture",
        "generatedChangeList": "123",
        "generatedIpChangeList": "456",
        "devices": [
            {
                "fpgaPart": json_fpga_part,
                "part": {"name": json_part_name},
            }
        ],
        "files": [{"name": name, "type": "FULL_BIT"} for name in declarations],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("llm_system.hwh", hwh)
        archive.writestr("llm_system_scoped.hwh", b"<EDKSYSTEM><MODULES/></EDKSYSTEM>")
        if extra_top_hwh is not None:
            archive.writestr("another_top.hwh", extra_top_hwh)
        if include_bit:
            archive.writestr(bit_name, bit_payload)
        if extra_bit is not None:
            archive.writestr(extra_bit[0], extra_bit[1])
        if include_json:
            archive.writestr("xsa.json", b"{" if malformed_json else json.dumps(metadata).encode("utf-8"))
    return bit_payload


class NetworkXsaAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def make_report(self, hwh: bytes | None = None, **zip_options: object) -> dict[str, object]:
        path = self.root / "fixture.xsa"
        _write_xsa(path, hwh or _synthetic_hwh(), **zip_options)
        return audit_network_xsa(path)

    def test_valid_network_xsa_passes_and_reports_provenance(self) -> None:
        path = self.root / "valid.xsa"
        bit_payload = _write_xsa(path, _synthetic_hwh())
        report = audit_network_xsa(path)

        self.assertTrue(report["pass"], report["errors"])
        self.assertEqual(report["hwh"]["entry"], "llm_system.hwh")
        self.assertTrue(report["bitstream"]["embedded"])
        self.assertEqual(report["bitstream"]["entries"][0]["sha256"], hashlib.sha256(bit_payload).hexdigest())
        self.assertEqual(report["provenance"]["generated_version"], "2025.1.1")
        self.assertEqual(report["provenance"]["part_name"], EXPECTED_PART_NAME)
        self.assertTrue(report["hwh"]["target"]["match"])
        self.assertTrue(report["provenance"]["part_name_match"])
        self.assertTrue(report["provenance"]["top_module_match"])
        self.assertTrue(report["provenance"]["full_bit_declaration_match"])
        self.assertTrue(report["bitstream"]["entries"][0]["header"]["parse_ok"])
        self.assertEqual(
            report["bitstream"]["entries"][0]["header"]["part_name"],
            EXPECTED_PART_NAME,
        )
        self.assertEqual(
            report["bitstream"]["entries"][0]["header"]["design_top"],
            EXPECTED_TOP_MODULE,
        )
        self.assertTrue(report["datapath"]["one_token"]["address_ok"])
        self.assertTrue(report["datapath"]["pl_ddr_status"]["config_ok"])
        self.assertTrue(report["datapath"]["pl_ddr"]["address_ok"])

    def test_embedded_bitstream_is_required(self) -> None:
        report = self.make_report(include_bit=False)
        self.assertFalse(report["pass"])
        self.assertFalse(report["bitstream"]["embedded"])
        self.assertTrue(any("exactly one embedded .bit" in error for error in report["errors"]))

    def test_hwh_target_identity_fails_closed_for_every_field(self) -> None:
        wrong_values = {
            "ARCH": "zynq",
            "DEVICE": "xczu3eg",
            "PACKAGE": "sfvc784x",
            "SPEEDGRADE": "-1",
        }
        for index, (name, value) in enumerate(wrong_values.items()):
            with self.subTest(name=name):
                path = self.root / f"wrong_hwh_target_{index}.xsa"
                _write_xsa(path, _synthetic_hwh(system_overrides={name: value}))
                report = audit_network_xsa(path)
                self.assertFalse(report["pass"])
                self.assertFalse(report["hwh"]["target"]["fields"][name]["match"])
                self.assertTrue(any(f"SYSTEMINFO {name}" in error for error in report["errors"]))

    def test_xsa_json_part_and_top_are_required(self) -> None:
        cases = {
            "part": {"json_part_name": "xczu3eg-sfvc784-2-i", "needle": "part name"},
            "fpga_part": {
                "json_fpga_part": "zynquplus:xczu3eg:sfvc784:-2:i",
                "needle": "fpgaPart",
            },
            "top": {"json_top_module": "other_wrapper", "needle": "topModuleName"},
        }
        for index, (name, options) in enumerate(cases.items()):
            with self.subTest(name=name):
                needle = str(options.pop("needle"))
                path = self.root / f"wrong_json_target_{index}.xsa"
                _write_xsa(path, _synthetic_hwh(), **options)
                report = audit_network_xsa(path)
                self.assertFalse(report["pass"])
                self.assertTrue(any(needle in error for error in report["errors"]))

    def test_xsa_json_is_required(self) -> None:
        report = self.make_report(include_json=False)
        self.assertFalse(report["pass"])
        self.assertTrue(any("exactly one xsa.json" in error for error in report["errors"]))

    def test_full_bit_declaration_must_match_unique_embedded_bit(self) -> None:
        report = self.make_report(declared_bit_entries=["different.bit"])
        self.assertFalse(report["pass"])
        self.assertFalse(report["provenance"]["full_bit_declaration_match"])
        self.assertTrue(any("FULL_BIT declarations" in error for error in report["errors"]))

    def test_duplicate_full_bit_declarations_fail_closed(self) -> None:
        report = self.make_report(
            declared_bit_entries=["llm_system_wrapper.bit", "llm_system_wrapper.bit"]
        )
        self.assertFalse(report["pass"])
        self.assertTrue(any("exactly one FULL_BIT" in error for error in report["errors"]))

    def test_empty_bitstream_fails_closed(self) -> None:
        report = self.make_report(bit_contents=b"")
        self.assertFalse(report["pass"])
        self.assertEqual(report["bitstream"]["entries"][0]["size"], 0)
        self.assertTrue(any("bitstream llm_system_wrapper.bit is empty" in error for error in report["errors"]))

    def test_zero_length_configuration_payload_fails_closed(self) -> None:
        report = self.make_report(bit_contents=_synthetic_bitstream(payload=b""))
        self.assertFalse(report["pass"])
        self.assertTrue(
            any("configuration payload is empty" in error for error in report["errors"])
        )

    def test_multiple_embedded_bitstreams_fail_closed(self) -> None:
        report = self.make_report(
            extra_bit=("unexpected.bit", _synthetic_bitstream()),
            declared_bit_entries=["llm_system_wrapper.bit"],
        )
        self.assertFalse(report["pass"])
        self.assertTrue(any("exactly one embedded .bit" in error for error in report["errors"]))

    def test_bitstream_header_part_and_top_are_bound(self) -> None:
        cases = {
            "part": {"bit_part_name": "xczu3eg-sfvc784-2-i", "needle": "targets part"},
            "top": {"bit_top_module": "other_wrapper", "needle": "design top"},
        }
        for index, (name, options) in enumerate(cases.items()):
            with self.subTest(name=name):
                needle = str(options.pop("needle"))
                path = self.root / f"wrong_bit_target_{index}.xsa"
                _write_xsa(path, _synthetic_hwh(), **options)
                report = audit_network_xsa(path)
                self.assertFalse(report["pass"])
                self.assertTrue(any(needle in error for error in report["errors"]))

    def test_unparseable_bitstream_header_fails_closed(self) -> None:
        report = self.make_report(bit_contents=b"not-a-xilinx-bitstream")
        self.assertFalse(report["pass"])
        self.assertFalse(report["bitstream"]["entries"][0]["header"]["parse_ok"])
        self.assertTrue(any("cannot parse embedded bitstream" in error for error in report["errors"]))

    def test_every_required_network_property_fails_closed(self) -> None:
        bad_values = {
            "PSU__ENET3__PERIPHERAL__ENABLE": "0",
            "PSU__ENET3__PERIPHERAL__IO": "MIO 0 .. 11",
            "PSU__ENET3__GRP_MDIO__ENABLE": "0",
            "PSU__ENET3__GRP_MDIO__IO": "MIO 52 .. 53",
            "PSU__TTC0__PERIPHERAL__ENABLE": "0",
            "PSU__TTC0__PERIPHERAL__IO": "MIO 0",
            "PSU__CRL_APB__GEM3_REF_CTRL__FREQMHZ": "100",
            "PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ": "100",
            "PSU__CRL_APB__GEM3_REF_CTRL__SRCSEL": "RPLL",
        }
        for index, (name, value) in enumerate(bad_values.items()):
            with self.subTest(name=name):
                path = self.root / f"bad_network_{index}.xsa"
                _write_xsa(path, _synthetic_hwh(network_overrides={name: value}))
                report = audit_network_xsa(path)
                self.assertFalse(report["pass"])
                self.assertFalse(report["network"]["parameters"][name]["match"])
                self.assertTrue(any(name in error for error in report["errors"]))

    def test_computed_gem3_frequency_accepts_vivado_rounding(self) -> None:
        report = self.make_report(
            _synthetic_hwh(
                network_overrides={
                    "PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ": "124.998749"
                }
            )
        )
        self.assertTrue(report["pass"], report["errors"])
        self.assertTrue(
            report["network"]["parameters"][
                "PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ"
            ]["match"]
        )

    def test_ps_network_peripheral_protection_entries_are_required(self) -> None:
        for instance in NETWORK_PS_PERIPHERALS:
            with self.subTest(instance=instance):
                report = self.make_report(
                    _synthetic_hwh(omit_network_peripherals=(instance,))
                )
                self.assertFalse(report["pass"])
                self.assertFalse(
                    report["network"]["ps_peripherals"][instance]["present_once"]
                )

    def test_one_token_control_and_ddr_map_are_guarded(self) -> None:
        cases = {
            "missing_module": {"include_qmap": False},
            "wrong_control_parameter": {"qmap_base": QMAP_BASE + 0x10000},
            "wrong_ps_control_map": {"ps_qmap_base": QMAP_BASE + 0x10000},
            "wrong_qmap_ddr_map": {"qmap_ddr_high": DDR_HIGH - 4},
        }
        for name, options in cases.items():
            with self.subTest(name=name):
                report = self.make_report(_synthetic_hwh(**options))
                self.assertFalse(report["pass"])
                self.assertTrue(any("qmap_one_token_axi_bd_0" in error for error in report["errors"]))

    def test_pl_ddr_status_contract_is_guarded(self) -> None:
        for name, options in {
            "wrong_width": {"status_width": 4},
            "wrong_parameter_address": {"status_base": STATUS_BASE + 0x10000},
            "wrong_ps_map": {"ps_status_base": STATUS_BASE + 0x10000},
        }.items():
            with self.subTest(name=name):
                report = self.make_report(_synthetic_hwh(**options))
                self.assertFalse(report["pass"])
                self.assertTrue(any("axi_gpio_0" in error for error in report["errors"]))

    def test_pl_ddr_aperture_contract_is_guarded(self) -> None:
        for name, options in {
            "wrong_parameter_high": {"ddr_high": DDR_HIGH - 4},
            "wrong_address_block_range": {"ddr_range": DDR_RANGE // 2},
            "wrong_ps_map_high": {"ps_ddr_high": DDR_HIGH - 4},
        }.items():
            with self.subTest(name=name):
                report = self.make_report(_synthetic_hwh(**options))
                self.assertFalse(report["pass"])
                self.assertTrue(any("ddr4_0" in error for error in report["errors"]))

    def test_multiple_top_level_hwh_files_fail_closed(self) -> None:
        report = self.make_report(extra_top_hwh=_synthetic_hwh())
        self.assertFalse(report["pass"])
        self.assertTrue(any("exactly one top-level HWH" in error for error in report["errors"]))

    def test_malformed_xsa_json_fails_closed(self) -> None:
        report = self.make_report(malformed_json=True)
        self.assertFalse(report["pass"])
        self.assertTrue(any("cannot parse xsa.json" in error for error in report["errors"]))

    def test_expected_archive_hash_is_an_optional_provenance_gate(self) -> None:
        path = self.root / "hash.xsa"
        _write_xsa(path, _synthetic_hwh())
        report = audit_network_xsa(path, expected_sha256="0" * 64)
        self.assertFalse(report["pass"])
        self.assertFalse(report["archive"]["sha256_match"])

    def test_current_non_network_xsa_is_rejected_but_datapath_is_preserved(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        current_xsa = repo / "Temp/final_vivado_export_20260802/llm_system_qwen3_one_token_boardready.xsa"
        if not current_xsa.is_file():
            self.skipTest(f"current checked artifact not present: {current_xsa}")

        report = audit_network_xsa(current_xsa)
        self.assertFalse(report["pass"])
        self.assertEqual(
            report["network"]["parameters"]["PSU__ENET3__PERIPHERAL__ENABLE"]["actual"],
            "0",
        )
        self.assertTrue(report["datapath"]["one_token"]["address_ok"])
        self.assertTrue(report["datapath"]["one_token"]["pl_ddr_master_map_ok"])
        self.assertTrue(report["datapath"]["pl_ddr_status"]["config_ok"])
        self.assertTrue(report["datapath"]["pl_ddr"]["ps_memory_map_ok"])
        self.assertTrue(report["bitstream"]["embedded"])
        self.assertTrue(report["hwh"]["target"]["match"])
        self.assertTrue(report["provenance"]["part_name_match"])
        self.assertTrue(report["provenance"]["top_module_match"])
        self.assertTrue(report["provenance"]["full_bit_declaration_match"])
        self.assertTrue(report["bitstream"]["entries"][0]["header"]["parse_ok"])
        self.assertFalse(any("parameter entry is missing" in error for error in report["errors"]))
        self.assertTrue(
            all(
                error.startswith("network parameter ")
                or error.startswith("PS protection map ")
                for error in report["errors"]
            ),
            report["errors"],
        )


if __name__ == "__main__":
    unittest.main()
