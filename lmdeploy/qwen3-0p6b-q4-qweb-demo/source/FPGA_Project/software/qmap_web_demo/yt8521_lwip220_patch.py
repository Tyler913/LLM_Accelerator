#!/usr/bin/env python3
"""Stage an audited Vitis 2025.1.1 lwip220 tree with YT8521 support.

The AMD installation is never modified.  A complete library copy is made in
the isolated Vitis workspace, the one pinned PHY source is patched by exact
text substitutions, and both source-tree hashes are returned for the build
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable


LIBRARY_NAME = "lwip220"
LIBRARY_VERSION = "lwip220_v1_2"
PATCH_RELATIVE_PATH = Path(
    "src/lwip-2.2.0/contrib/ports/xilinx/netif/xemacpsif_physpeed.c"
)
METADATA_RELATIVE_PATH = Path("data/lwip220.yaml")
PINNED_INPUT_SHA256 = (
    "b45bad2d4c9e2543db7ec8e70b7b450633d748de749b8e67dc4f27894a63430d"
)
PINNED_METADATA_SHA256 = (
    "898d479f1ff6b828ab4666f371ca067e69b21b8b767d4b412422c18e8b09800e"
)
MOTORCOMM_YT8521_PHY_ID = "0x0000011A"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_files(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"lwip220 root must be a regular directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"lwip220 source must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise RuntimeError(f"lwip220 source tree is empty: {root}")
    return tuple(files)


def _tree_sha256(root: Path, files: Iterable[Path] | None = None) -> str:
    digest = hashlib.sha256()
    selected = _regular_files(root) if files is None else tuple(files)
    for path in selected:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"pinned lwip220 source expected one {label}, found {count}"
        )
    return text.replace(old, new, 1)


def patch_source_text(text: str) -> str:
    """Return the deterministic YT8521 variant of the pinned AMD source."""

    text = _replace_once(
        text,
        '#include "xemac_ieee_reg.h"\n',
        '#include "xemac_ieee_reg.h"\n#include "sleep.h"\n',
        "sleep include insertion point",
    )
    text = _replace_once(
        text,
        "#define PHY_XILINX_PCS_PMA_ID2\t\t\t0x0C00\n",
        "#define PHY_XILINX_PCS_PMA_ID2\t\t\t0x0C00\n"
        "#define PHY_MOTORCOMM_YT8521_ID\t\t\t0x0000011AU\n\n"
        "#define YT8521_PAGE_SELECT\t\t\t0x1EU\n"
        "#define YT8521_PAGE_DATA\t\t\t0x1FU\n"
        "#define YT8521_REG_SPACE_SELECT\t\t0xA000U\n"
        "#define YT8521_CHIP_CONFIG\t\t\t0xA001U\n"
        "#define YT8521_RGMII_CONFIG1\t\t\t0xA003U\n"
        "#define YT8521_CLOCK_GATING\t\t\t0x000CU\n"
        "#define YT8521_SLEEP_CONTROL1\t\t\t0x0027U\n"
        "#define YT8521_UTP_SPACE_MASK\t\t\t0x0002U\n"
        "#define YT8521_CHIP_SOFT_RESET\t\t\t0x8000U\n"
        "#define YT8521_RXC_DELAY_ENABLE\t\t\t0x0100U\n"
        "#define YT8521_RX_DELAY_MASK\t\t\t0x3C00U\n"
        "#define YT8521_TX_DELAY_MASK\t\t\t0x000FU\n"
        "#define YT8521_DELAY_1950_PS\t\t\t0x000DU\n"
        "#define YT8521_AUTO_SLEEP\t\t\t0x8000U\n"
        "#define YT8521_RX_CLOCK_GATED\t\t\t0x1000U\n"
        "#define YT8521_SPECIFIC_STATUS\t\t\t0x0011U\n"
        "#define YT8521_STATUS_SPEED_MASK\t\t0xC000U\n"
        "#define YT8521_STATUS_DUPLEX\t\t\t0x2000U\n"
        "#define YT8521_STATUS_RESOLVED\t\t\t0x0800U\n"
        "#define YT8521_STATUS_LINK\t\t\t0x0400U\n"
        "#define YT8521_BMCR_POWER_DOWN\t\t\t0x0800U\n"
        "#define YT8521_BMCR_ISOLATE\t\t\t0x0400U\n"
        "#define YT8521_AUTONEG_POLLS\t\t\t150U\n"
        "#define YT8521_AUTONEG_POLL_MS\t\t\t100U\n",
        "Motorcomm constant insertion point",
    )

    old_identify = """static void phy_identify(XEmacPs *xemacpsp, u32_t phy_addr, u32_t emacnum)
{
\tu16_t phy_reg;
\tu16_t phy_id;

\tXEmacPs_PhyRead(xemacpsp, phy_addr, PHY_DETECT_REG,
\t\t\t&phy_reg);
\tXEmacPs_PhyRead(xemacpsp, phy_addr, PHY_IDENTIFIER_1_REG,
\t\t\t&phy_id);

\tif (((phy_reg != 0xFFFF) &&
\t     ((phy_reg & PHY_DETECT_MASK) == PHY_DETECT_MASK)) ||
\t    (phy_id == PHY_XILINX_PCS_PMA_ID1)) {
\t\t/* Found a valid PHY address */
\t\tLWIP_DEBUGF(NETIF_DEBUG, ("XEmacPs detect_phy: PHY detected at address %d.\\r\\n",
\t\t\t\t\t  phy_addr));
\t\tif (emacnum == 0) {
\t\t\tphymapemac0[phy_addr] = TRUE;
\t\t} else {
\t\t\tphymapemac1[phy_addr] = TRUE;
\t\t}

\t\tXEmacPs_PhyRead(xemacpsp, phy_addr, PHY_IDENTIFIER_1_REG,
\t\t\t\t&phy_reg);
\t\tif ((phy_reg != PHY_MARVELL_IDENTIFIER) &&
\t\t    (phy_reg != PHY_TI_IDENTIFIER) &&
\t\t    (phy_reg != PHY_REALTEK_IDENTIFIER) &&
\t\t    (phy_reg != PHY_ADI_IDENTIFIER)) {
\t\t\txil_printf("WARNING: Not a Marvell or TI or Realtek or Xilinx PCS PMA Ethernet PHY or ADI Ethernet PHY. Please verify the initialization sequence\\r\\n");
\t\t}
\t}
}
"""
    new_identify = """static void phy_identify(XEmacPs *xemacpsp, u32_t phy_addr, u32_t emacnum)
{
\tu16_t phy_reg;
\tu16_t phy_id1;
\tu16_t phy_id2;
\tu32_t phy_id;

\tXEmacPs_PhyRead(xemacpsp, phy_addr, PHY_DETECT_REG,
\t\t\t&phy_reg);
\tXEmacPs_PhyRead(xemacpsp, phy_addr, PHY_IDENTIFIER_1_REG,
\t\t\t&phy_id1);
\tXEmacPs_PhyRead(xemacpsp, phy_addr, PHY_IDENTIFIER_2_REG,
\t\t\t&phy_id2);
\tphy_id = ((u32_t)phy_id1 << 16) | (u32_t)phy_id2;

\tif (((phy_reg != 0xFFFF) &&
\t     ((phy_reg & PHY_DETECT_MASK) == PHY_DETECT_MASK)) ||
\t    (phy_id1 == PHY_XILINX_PCS_PMA_ID1)) {
\t\t/* Found a valid PHY address */
\t\tLWIP_DEBUGF(NETIF_DEBUG, ("XEmacPs detect_phy: PHY detected at address %d.\\r\\n",
\t\t\t\t\t  phy_addr));
\t\tif (emacnum == 0) {
\t\t\tphymapemac0[phy_addr] = TRUE;
\t\t} else {
\t\t\tphymapemac1[phy_addr] = TRUE;
\t\t}

\t\tif (phy_id == PHY_MOTORCOMM_YT8521_ID) {
\t\t\txil_printf("Detected Motorcomm YT8521 at PHY address %d (id1=0x%04x id2=0x%04x)\\r\\n",
\t\t\t\t   phy_addr, phy_id1, phy_id2);
\t\t} else if ((phy_id1 != PHY_MARVELL_IDENTIFIER) &&
\t\t\t   (phy_id1 != PHY_TI_IDENTIFIER) &&
\t\t\t   (phy_id1 != PHY_REALTEK_IDENTIFIER) &&
\t\t\t   (phy_id1 != PHY_ADI_IDENTIFIER) &&
\t\t\t   (phy_id1 != PHY_XILINX_PCS_PMA_ID1)) {
\t\t\txil_printf("WARNING: Unsupported Ethernet PHY at address %d (id1=0x%04x id2=0x%04x)\\r\\n",
\t\t\t\t   phy_addr, phy_id1, phy_id2);
\t\t}
\t}
}
"""
    text = _replace_once(
        text, old_identify, new_identify, "phy_identify implementation"
    )

    old_dispatch = """static u32_t get_IEEE_phy_speed(XEmacPs *xemacpsp, u32_t phy_addr)
{
\tu16_t phy_identity;
\tu32_t RetStatus;
\tchar *PhyType;

#ifdef SDT
\tPhyType = xemacpsp->Config.PhyType;
#endif

\tXEmacPs_PhyRead(xemacpsp, phy_addr, PHY_IDENTIFIER_1_REG,
\t\t\t\t\t&phy_identity);
\tif (phy_identity == PHY_TI_IDENTIFIER) {
\t\tif (!strcmp(PhyType, "sgmii")) {
\t\t\tRetStatus = get_TI_phy_speed_sgmii(xemacpsp, phy_addr);
\t\t} else {
\t\t\tRetStatus = get_TI_phy_speed(xemacpsp, phy_addr);
\t\t}
\t} else if (phy_identity == PHY_REALTEK_IDENTIFIER) {
\t\tRetStatus = get_Realtek_phy_speed(xemacpsp, phy_addr);
\t} else if (phy_identity == PHY_XILINX_PCS_PMA_ID1) {
\t\tRetStatus = get_Xilinx_pcs_pma_phy_speed(xemacpsp, phy_addr);
\t} else if (phy_identity == PHY_ADI_IDENTIFIER) {
\t\tRetStatus = get_Adi_phy_speed(xemacpsp, phy_addr);
\t} else {
\t\tRetStatus = get_Marvell_phy_speed(xemacpsp, phy_addr);
\t}

\treturn RetStatus;
}
"""
    new_dispatch = r"""static u32_t yt8521_ext_read(XEmacPs *xemacpsp, u32_t phy_addr,
				      u16_t reg, u16_t *value)
{
	if (XEmacPs_PhyWrite(xemacpsp, phy_addr, YT8521_PAGE_SELECT, reg) !=
	    XST_SUCCESS) {
		return XST_FAILURE;
	}
	if (XEmacPs_PhyRead(xemacpsp, phy_addr, YT8521_PAGE_DATA, value) !=
	    XST_SUCCESS) {
		return XST_FAILURE;
	}
	return XST_SUCCESS;
}

static u32_t yt8521_ext_write(XEmacPs *xemacpsp, u32_t phy_addr,
				       u16_t reg, u16_t value)
{
	if (XEmacPs_PhyWrite(xemacpsp, phy_addr, YT8521_PAGE_SELECT, reg) !=
	    XST_SUCCESS) {
		return XST_FAILURE;
	}
	if (XEmacPs_PhyWrite(xemacpsp, phy_addr, YT8521_PAGE_DATA, value) !=
	    XST_SUCCESS) {
		return XST_FAILURE;
	}
	return XST_SUCCESS;
}

static u32_t yt8521_ext_modify(XEmacPs *xemacpsp, u32_t phy_addr,
					u16_t reg, u16_t clear_mask,
					u16_t set_mask)
{
	u16_t value;

	if (yt8521_ext_read(xemacpsp, phy_addr, reg, &value) != XST_SUCCESS) {
		return XST_FAILURE;
	}
	value = (u16_t)((value & (u16_t)(~clear_mask)) | set_mask);
	return yt8521_ext_write(xemacpsp, phy_addr, reg, value);
}

static u32_t yt8521_config_rgmii_delay(XEmacPs *xemacpsp, u32_t phy_addr,
					const char *phy_type)
{
	u16_t delay = 0;

	if ((phy_type != NULL) &&
	    ((!strcmp(phy_type, "rgmii-id")) ||
	     (!strcmp(phy_type, "rgmii-rxid")))) {
		delay |= (u16_t)(YT8521_DELAY_1950_PS << 10);
	}
	if ((phy_type != NULL) &&
	    ((!strcmp(phy_type, "rgmii-id")) ||
	     (!strcmp(phy_type, "rgmii-txid")))) {
		delay |= YT8521_DELAY_1950_PS;
	}
	if (yt8521_ext_modify(xemacpsp, phy_addr, YT8521_CHIP_CONFIG,
				YT8521_RXC_DELAY_ENABLE, 0) != XST_SUCCESS) {
		return XST_FAILURE;
	}
	return yt8521_ext_modify(xemacpsp, phy_addr, YT8521_RGMII_CONFIG1,
				  YT8521_RX_DELAY_MASK | YT8521_TX_DELAY_MASK,
				  delay);
}

static u32_t get_Motorcomm_YT8521_phy_speed(XEmacPs *xemacpsp,
					     u32_t phy_addr)
{
	u16_t advertise;
	u16_t bmcr;
	u16_t bmsr_latch = 0;
	u16_t bmsr_now = 0;
	u16_t chip_config;
	u16_t control1000;
	u16_t specific = 0;
	u32_t poll;
	const char *phy_type = NULL;

#ifdef SDT
	phy_type = xemacpsp->Config.PhyType;
#endif
	if (yt8521_ext_read(xemacpsp, phy_addr, YT8521_CHIP_CONFIG,
			     &chip_config) != XST_SUCCESS) {
		return XST_FAILURE;
	}
	xil_printf("YT8521 phy_addr=%d chip_config=0x%04x mode=%d phy_type=%s\r\n",
		   phy_addr, chip_config, chip_config & 0x7U,
		   (phy_type == NULL) ? "unknown" : phy_type);

	/* The Motorcomm SDK defines bit 15 of A001 as active-low software
	 * reset: clear it, then wait for hardware to return it to one. */
	chip_config &= (u16_t)(~YT8521_CHIP_SOFT_RESET);
	if (yt8521_ext_write(xemacpsp, phy_addr, YT8521_CHIP_CONFIG,
			      chip_config) != XST_SUCCESS) {
		return XST_FAILURE;
	}
	for (poll = 0; poll < 12U; poll++) {
		msleep(50U);
		if (yt8521_ext_read(xemacpsp, phy_addr, YT8521_CHIP_CONFIG,
				     &chip_config) != XST_SUCCESS) {
			return XST_FAILURE;
		}
		if ((chip_config & YT8521_CHIP_SOFT_RESET) != 0U) {
			break;
		}
	}
	if ((chip_config & YT8521_CHIP_SOFT_RESET) == 0U) {
		xil_printf("YT8521 software reset timeout\r\n");
		return XST_FAILURE;
	}

	if (yt8521_ext_modify(xemacpsp, phy_addr, YT8521_REG_SPACE_SELECT,
				YT8521_UTP_SPACE_MASK, 0) != XST_SUCCESS ||
	    yt8521_ext_modify(xemacpsp, phy_addr, YT8521_SLEEP_CONTROL1,
				YT8521_AUTO_SLEEP, 0) != XST_SUCCESS ||
	    yt8521_ext_modify(xemacpsp, phy_addr, YT8521_CLOCK_GATING,
				YT8521_RX_CLOCK_GATED, 0) != XST_SUCCESS ||
	    yt8521_config_rgmii_delay(xemacpsp, phy_addr, phy_type) !=
		XST_SUCCESS) {
		xil_printf("YT8521 extended-register initialization failed\r\n");
		return XST_FAILURE;
	}

	if (XEmacPs_PhyRead(xemacpsp, phy_addr,
			     IEEE_AUTONEGO_ADVERTISE_REG, &advertise) != XST_SUCCESS) {
		return XST_FAILURE;
	}
	advertise |= IEEE_ASYMMETRIC_PAUSE_MASK | IEEE_PAUSE_MASK |
		     ADVERTISE_100 | ADVERTISE_10;
	if (XEmacPs_PhyWrite(xemacpsp, phy_addr,
			      IEEE_AUTONEGO_ADVERTISE_REG, advertise) != XST_SUCCESS ||
	    XEmacPs_PhyRead(xemacpsp, phy_addr,
			     IEEE_1000_ADVERTISE_REG_OFFSET, &control1000) !=
		XST_SUCCESS) {
		return XST_FAILURE;
	}
	control1000 |= ADVERTISE_1000;
	if (XEmacPs_PhyWrite(xemacpsp, phy_addr,
			      IEEE_1000_ADVERTISE_REG_OFFSET, control1000) !=
		XST_SUCCESS ||
	    XEmacPs_PhyRead(xemacpsp, phy_addr, IEEE_CONTROL_REG_OFFSET,
			     &bmcr) != XST_SUCCESS) {
		return XST_FAILURE;
	}
	bmcr &= (u16_t)(~(YT8521_BMCR_POWER_DOWN | YT8521_BMCR_ISOLATE));
	bmcr |= IEEE_CTRL_AUTONEGOTIATE_ENABLE | IEEE_STAT_AUTONEGOTIATE_RESTART;
	if (XEmacPs_PhyWrite(xemacpsp, phy_addr, IEEE_CONTROL_REG_OFFSET,
			      bmcr) != XST_SUCCESS) {
		return XST_FAILURE;
	}

	xil_printf("Start YT8521 UTP autonegotiation\r\n");
	for (poll = 0; poll < YT8521_AUTONEG_POLLS; poll++) {
		msleep(YT8521_AUTONEG_POLL_MS);
		if (XEmacPs_PhyRead(xemacpsp, phy_addr, IEEE_STATUS_REG_OFFSET,
				     &bmsr_latch) != XST_SUCCESS ||
		    XEmacPs_PhyRead(xemacpsp, phy_addr, IEEE_STATUS_REG_OFFSET,
				     &bmsr_now) != XST_SUCCESS ||
		    XEmacPs_PhyRead(xemacpsp, phy_addr, YT8521_SPECIFIC_STATUS,
				     &specific) != XST_SUCCESS) {
			return XST_FAILURE;
		}
		if (((bmsr_now & IEEE_STAT_AUTONEGOTIATE_COMPLETE) != 0U) &&
		    ((specific & (YT8521_STATUS_LINK | YT8521_STATUS_RESOLVED)) ==
		     (YT8521_STATUS_LINK | YT8521_STATUS_RESOLVED))) {
			break;
		}
	}
	if (poll == YT8521_AUTONEG_POLLS) {
		xil_printf("YT8521 autonegotiation timeout: bmcr=0x%04x bmsr_latch=0x%04x bmsr_now=0x%04x adv=0x%04x ctrl1000=0x%04x status=0x%04x\r\n",
			   bmcr, bmsr_latch, bmsr_now, advertise,
			   control1000, specific);
		return XST_FAILURE;
	}

	xil_printf("YT8521 link resolved: bmsr=0x%04x status=0x%04x duplex=%s\r\n",
		   bmsr_now, specific,
		   ((specific & YT8521_STATUS_DUPLEX) != 0U) ? "full" : "half");
	switch (specific & YT8521_STATUS_SPEED_MASK) {
	case 0x8000U:
		return SPEED_1000MBPS;
	case 0x4000U:
		return SPEED_100MBPS;
	case 0x0000U:
		return SPEED_10MBPS;
	default:
		xil_printf("YT8521 reported invalid speed status 0x%04x\r\n",
			   specific);
		return XST_FAILURE;
	}
}

static u32_t get_IEEE_phy_speed(XEmacPs *xemacpsp, u32_t phy_addr)
{
	u16_t phy_identity1;
	u16_t phy_identity2;
	u32_t phy_identity;
	u32_t RetStatus;
	char *PhyType = NULL;

#ifdef SDT
	PhyType = xemacpsp->Config.PhyType;
#endif

	if (XEmacPs_PhyRead(xemacpsp, phy_addr, PHY_IDENTIFIER_1_REG,
			     &phy_identity1) != XST_SUCCESS ||
	    XEmacPs_PhyRead(xemacpsp, phy_addr, PHY_IDENTIFIER_2_REG,
			     &phy_identity2) != XST_SUCCESS) {
		return XST_FAILURE;
	}
	phy_identity = ((u32_t)phy_identity1 << 16) | (u32_t)phy_identity2;
	if (phy_identity == PHY_MOTORCOMM_YT8521_ID) {
		RetStatus = get_Motorcomm_YT8521_phy_speed(xemacpsp, phy_addr);
	} else if (phy_identity1 == PHY_TI_IDENTIFIER) {
		if ((PhyType != NULL) && (!strcmp(PhyType, "sgmii"))) {
			RetStatus = get_TI_phy_speed_sgmii(xemacpsp, phy_addr);
		} else {
			RetStatus = get_TI_phy_speed(xemacpsp, phy_addr);
		}
	} else if (phy_identity1 == PHY_REALTEK_IDENTIFIER) {
		RetStatus = get_Realtek_phy_speed(xemacpsp, phy_addr);
	} else if (phy_identity1 == PHY_XILINX_PCS_PMA_ID1) {
		RetStatus = get_Xilinx_pcs_pma_phy_speed(xemacpsp, phy_addr);
	} else if (phy_identity1 == PHY_ADI_IDENTIFIER) {
		RetStatus = get_Adi_phy_speed(xemacpsp, phy_addr);
	} else if (phy_identity1 == PHY_MARVELL_IDENTIFIER) {
		RetStatus = get_Marvell_phy_speed(xemacpsp, phy_addr);
	} else {
		xil_printf("Unsupported PHY initialization: address=%d id1=0x%04x id2=0x%04x\r\n",
			   phy_addr, phy_identity1, phy_identity2);
		RetStatus = XST_FAILURE;
	}

	return RetStatus;
}
"""
    text = _replace_once(
        text, old_dispatch, new_dispatch, "PHY-speed dispatch implementation"
    )
    return text


def stage_patched_library(source_root: Path, destination_root: Path) -> dict[str, Any]:
    source = source_root.resolve(strict=True)
    destination = destination_root.resolve()
    if source.name != LIBRARY_VERSION:
        raise RuntimeError(
            f"lwip220 source directory must be named {LIBRARY_VERSION}: {source}"
        )
    if destination.exists():
        raise RuntimeError(f"patched lwip220 destination already exists: {destination}")
    source_files = _regular_files(source)
    patch_input = source / PATCH_RELATIVE_PATH
    if patch_input not in source_files:
        raise RuntimeError(f"pinned lwip220 PHY source is missing: {patch_input}")
    actual_input_hash = _sha256_file(patch_input)
    if actual_input_hash != PINNED_INPUT_SHA256:
        raise RuntimeError(
            "Vitis lwip220 PHY source SHA-256 mismatch: "
            f"expected {PINNED_INPUT_SHA256}, got {actual_input_hash}"
        )
    metadata_input = source / METADATA_RELATIVE_PATH
    if metadata_input not in source_files:
        raise RuntimeError(f"pinned lwip220 metadata is missing: {metadata_input}")
    actual_metadata_hash = _sha256_file(metadata_input)
    if actual_metadata_hash != PINNED_METADATA_SHA256:
        raise RuntimeError(
            "Vitis lwip220 metadata SHA-256 mismatch: "
            f"expected {PINNED_METADATA_SHA256}, got {actual_metadata_hash}"
        )
    source_tree_hash = _tree_sha256(source, source_files)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    patch_output = destination / PATCH_RELATIVE_PATH
    patch_output.chmod(patch_output.stat().st_mode | stat.S_IWRITE)
    original_text = patch_output.read_text(encoding="utf-8")
    patched_text = patch_source_text(original_text)
    patch_output.write_text(patched_text, encoding="utf-8", newline="\n")
    patched_hash = _sha256_file(patch_output)
    if patched_hash == actual_input_hash:
        raise RuntimeError("YT8521 patch did not change the pinned lwip220 source")
    staged_files = _regular_files(destination)
    if len(staged_files) != len(source_files):
        raise RuntimeError("patched lwip220 file count changed during staging")
    record = {
        "library": LIBRARY_NAME,
        "version": LIBRARY_VERSION,
        "source_root": str(source),
        "source_tree_sha256": source_tree_hash,
        "destination": str(destination),
        "staged_tree_sha256": _tree_sha256(destination, staged_files),
        "patched_file": PATCH_RELATIVE_PATH.as_posix(),
        "original_sha256": actual_input_hash,
        "metadata_sha256": actual_metadata_hash,
        "patched_sha256": patched_hash,
        "motorcomm_phy_id": MOTORCOMM_YT8521_PHY_ID,
    }
    verify_staged_library(record)
    return record


def verify_staged_library(record: dict[str, Any]) -> None:
    """Fail if a staged library no longer matches its audited record."""

    if record.get("library") != LIBRARY_NAME:
        raise RuntimeError("staged BSP override has an unexpected library name")
    if record.get("version") != LIBRARY_VERSION:
        raise RuntimeError("staged BSP override has an unexpected library version")
    if record.get("motorcomm_phy_id") != MOTORCOMM_YT8521_PHY_ID:
        raise RuntimeError("staged BSP override has an unexpected PHY ID")
    if record.get("original_sha256") != PINNED_INPUT_SHA256:
        raise RuntimeError("staged BSP override has an unexpected input hash")
    if record.get("metadata_sha256") != PINNED_METADATA_SHA256:
        raise RuntimeError("staged BSP override has an unexpected metadata hash")
    destination = Path(str(record.get("destination", ""))).resolve(strict=True)
    patch_relative = str(record.get("patched_file", ""))
    if patch_relative != PATCH_RELATIVE_PATH.as_posix():
        raise RuntimeError("staged BSP override has an unexpected patched path")
    patch_output = destination / PATCH_RELATIVE_PATH
    if patch_output.is_symlink() or not patch_output.is_file():
        raise RuntimeError(f"staged BSP override source is missing: {patch_output}")
    actual_patch_hash = _sha256_file(patch_output)
    if actual_patch_hash != record.get("patched_sha256"):
        raise RuntimeError(
            "staged BSP override patched-file SHA-256 mismatch: "
            f"expected {record.get('patched_sha256')}, got {actual_patch_hash}"
        )
    actual_tree_hash = _tree_sha256(destination)
    if actual_tree_hash != record.get("staged_tree_sha256"):
        raise RuntimeError(
            "staged BSP override tree SHA-256 mismatch: "
            f"expected {record.get('staged_tree_sha256')}, got {actual_tree_hash}"
        )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    record = stage_patched_library(args.source_root, args.destination_root)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
