#include "qmap_row1024_image.h"

#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xil_types.h"
#include "xstatus.h"

#define DDR4_STATUS_BASE ((UINTPTR)0x00000000A0010000ULL)
#define PL_CTRL_STATUS_BASE ((UINTPTR)0x00000000A0020000ULL)
#define PL_RESULT_BASE ((UINTPTR)0x00000000A0030000ULL)

#define AXI_GPIO_DATA_OFFSET 0x00U
#define AXI_GPIO_TRI_OFFSET 0x04U
#define AXI_GPIO_DATA2_OFFSET 0x08U
#define AXI_GPIO_TRI2_OFFSET 0x0CU

#define DDR4_STATUS_CALIB_COMPLETE_MASK 0x00000001U
#define DDR4_STATUS_UI_RESET_MASK 0x00000002U
#define DDR4_STATUS_AXI_RESETN_MASK 0x00000004U

#define PL_CTRL_START_MASK 0x00000001U
#define PL_CTRL_CLEAR_MASK 0x00000002U

#define PL_STATUS_BUSY_MASK 0x00000001U
#define PL_STATUS_DONE_MASK 0x00000002U
#define PL_STATUS_ERROR_MASK 0x00000004U
#define PL_STATUS_COMPARE_MATCH_MASK 0x00000008U
#define PL_STATUS_EXPECTED_DONE_MATCH 0x0000000AU

#define QMAP_MAGIC 0x50414D51U
#define QMAP_VERSION 1U
#define QMAP_HEADER_BYTES 256U
#define QMAP_DESCRIPTOR_BYTES 128U
#define QMAP_DESCRIPTOR_COUNT 4U
#define QMAP_DESCRIPTOR_CAPACITY 8U
#define QMAP_DESCRIPTOR_TABLE_OFFSET 0x0100U
#define QMAP_PAYLOAD_BASE_OFFSET 0x0500U

#define EXPECTED_ROW_SUM_Q26_LOW32 0xFFCADDC7U

#define HEARTBEAT_DELAY_CYCLES 100000000U
#define SHORT_DELAY_CYCLES 1000000U
#define DDR4_STATUS_POLL_ATTEMPTS 200U
#define PL_STATUS_POLL_ATTEMPTS 50000U

#define DEBUG_MARKER_BASE ((UINTPTR)0x00000000FFFFF000ULL)
#define DEBUG_MARKER_STAGE_OFFSET 0x00U
#define DEBUG_MARKER_DETAIL_OFFSET 0x04U
#define DEBUG_MARKER_PL_STATUS_OFFSET 0x08U

#define DBG_STAGE_MAIN_START 0x10000001U
#define DBG_STAGE_GPIO_CONFIGURED 0x10000002U
#define DBG_STAGE_DDR4_READY 0x10000003U
#define DBG_STAGE_WRITE_QMAP_BEGIN 0x20000000U
#define DBG_STAGE_WRITE_QMAP_DONE 0x20000001U
#define DBG_STAGE_READBACK_BEGIN 0x30000000U
#define DBG_STAGE_READBACK_DONE 0x30000001U
#define DBG_STAGE_HEADER_BEGIN 0x40000000U
#define DBG_STAGE_HEADER_DONE 0x40000001U
#define DBG_STAGE_PL_BEGIN 0x50000000U
#define DBG_STAGE_PL_DONE 0x50000001U
#define DBG_STAGE_RESULT_BEGIN 0x60000000U
#define DBG_STAGE_PASS 0x70000001U
#define DBG_STAGE_FAIL_DDR4 0xE0000001U
#define DBG_STAGE_FAIL_WRITE_QMAP 0xE0000002U
#define DBG_STAGE_FAIL_READBACK 0xE0000003U
#define DBG_STAGE_FAIL_HEADER 0xE0000004U
#define DBG_STAGE_FAIL_PL 0xE0000005U
#define DBG_STAGE_FAIL_RESULT 0xE0000006U

static void debug_mark(u32 stage)
{
    Xil_Out32(DEBUG_MARKER_BASE + DEBUG_MARKER_STAGE_OFFSET, stage);
    Xil_DCacheFlushRange((INTPTR)(DEBUG_MARKER_BASE + DEBUG_MARKER_STAGE_OFFSET), sizeof(u32));
}

static void debug_detail(u32 detail)
{
    Xil_Out32(DEBUG_MARKER_BASE + DEBUG_MARKER_DETAIL_OFFSET, detail);
    Xil_DCacheFlushRange((INTPTR)(DEBUG_MARKER_BASE + DEBUG_MARKER_DETAIL_OFFSET), sizeof(u32));
}

static void debug_pl_status(u32 status)
{
    Xil_Out32(DEBUG_MARKER_BASE + DEBUG_MARKER_PL_STATUS_OFFSET, status);
    Xil_DCacheFlushRange((INTPTR)(DEBUG_MARKER_BASE + DEBUG_MARKER_PL_STATUS_OFFSET), sizeof(u32));
}

static void print_addr(UINTPTR addr)
{
    xil_printf("0x%x_%08x", (u32)(addr >> 32), (u32)addr);
}

static void print_u64_hex(u64 value)
{
    xil_printf("0x%x_%08x", (u32)(value >> 32), (u32)value);
}

static void delay_cycles(u32 cycles)
{
    for (volatile u32 i = 0U; i < cycles; ++i) {
    }
}

static void delay_spin(void)
{
    delay_cycles(HEARTBEAT_DELAY_CYCLES);
}

static u32 load_le32(const u8 *bytes)
{
    return ((u32)bytes[0]) |
           ((u32)bytes[1] << 8) |
           ((u32)bytes[2] << 16) |
           ((u32)bytes[3] << 24);
}

static u64 read_le64(UINTPTR addr)
{
    u32 low = Xil_In32(addr);
    u32 high = Xil_In32(addr + 4U);

    return ((u64)high << 32) | (u64)low;
}

static void print_ddr4_status(u32 status)
{
    xil_printf("DDR4 status raw 0x%x calib_complete=%d ui_reset=%d axi_resetn=%d\r\n",
               status,
               (status & DDR4_STATUS_CALIB_COMPLETE_MASK) ? 1 : 0,
               (status & DDR4_STATUS_UI_RESET_MASK) ? 1 : 0,
               (status & DDR4_STATUS_AXI_RESETN_MASK) ? 1 : 0);
}

static int wait_ddr4_ready(void)
{
    xil_printf("\r\nDDR4 status GPIO check\r\n");
    xil_printf("Status GPIO base ");
    print_addr(DDR4_STATUS_BASE);
    xil_printf("\r\n");

    for (u32 attempt = 0U; attempt < DDR4_STATUS_POLL_ATTEMPTS; ++attempt) {
        u32 status = Xil_In32(DDR4_STATUS_BASE + AXI_GPIO_DATA_OFFSET);

        if ((status & DDR4_STATUS_CALIB_COMPLETE_MASK) != 0U &&
            (status & DDR4_STATUS_UI_RESET_MASK) == 0U &&
            (status & DDR4_STATUS_AXI_RESETN_MASK) != 0U) {
            xil_printf("DDR4 controller and AXI reset path report ready\r\n");
            print_ddr4_status(status);
            return XST_SUCCESS;
        }

        if (attempt == 0U || ((attempt + 1U) % 20U) == 0U) {
            xil_printf("DDR4 not ready poll %d/%d: ",
                       (int)(attempt + 1U), (int)DDR4_STATUS_POLL_ATTEMPTS);
            print_ddr4_status(status);
        }

        delay_cycles(SHORT_DELAY_CYCLES);
    }

    xil_printf("DDR4 is not ready; stopping before QMAP access\r\n");
    return XST_FAILURE;
}

static int check_u32(const char *name, UINTPTR addr, u32 expected)
{
    u32 actual = Xil_In32(addr);

    if (actual != expected) {
        xil_printf("FAIL %s expected 0x%x got 0x%x at ", name, expected, actual);
        print_addr(addr);
        xil_printf("\r\n");
        return XST_FAILURE;
    }

    xil_printf("PASS %s = 0x%x\r\n", name, actual);
    return XST_SUCCESS;
}

static int check_u64(const char *name, UINTPTR addr, u64 expected)
{
    u64 actual = read_le64(addr);

    if (actual != expected) {
        xil_printf("FAIL %s expected ", name);
        print_u64_hex(expected);
        xil_printf(" got ");
        print_u64_hex(actual);
        xil_printf(" at ");
        print_addr(addr);
        xil_printf("\r\n");
        return XST_FAILURE;
    }

    xil_printf("PASS %s = ", name);
    print_u64_hex(actual);
    xil_printf("\r\n");
    return XST_SUCCESS;
}

static int write_qmap_image(void)
{
    xil_printf("\r\nWriting QMAP row1024 image for PL master\r\n");
    xil_printf("Target base ");
    print_addr((UINTPTR)QMAP_ROW1024_BASE_ADDR);
    xil_printf(" bytes 0x%x\r\n", QMAP_ROW1024_IMAGE_SIZE);

    for (u32 offset = 0U; offset < QMAP_ROW1024_IMAGE_SIZE; offset += 4U) {
        u32 value = load_le32(&qmap_row1024_image[offset]);
        Xil_Out32((UINTPTR)QMAP_ROW1024_BASE_ADDR + offset, value);
    }

    Xil_DCacheFlushRange((INTPTR)QMAP_ROW1024_BASE_ADDR, QMAP_ROW1024_IMAGE_SIZE);
    Xil_DCacheInvalidateRange((INTPTR)QMAP_ROW1024_BASE_ADDR, QMAP_ROW1024_IMAGE_SIZE);

    xil_printf("QMAP row1024 image write completed\r\n");
    return XST_SUCCESS;
}

static int compare_qmap_readback(void)
{
    xil_printf("\r\nQMAP row1024 readback word compare\r\n");

    for (u32 offset = 0U; offset < QMAP_ROW1024_IMAGE_SIZE; offset += 4U) {
        u32 expected = load_le32(&qmap_row1024_image[offset]);
        u32 actual = Xil_In32((UINTPTR)QMAP_ROW1024_BASE_ADDR + offset);

        if (actual != expected) {
            xil_printf("FAIL readback mismatch offset 0x%x addr ", offset);
            print_addr((UINTPTR)QMAP_ROW1024_BASE_ADDR + offset);
            xil_printf(" expected 0x%x got 0x%x\r\n", expected, actual);
            return XST_FAILURE;
        }
    }

    xil_printf("Readback compare PASSED for %d bytes\r\n", (int)QMAP_ROW1024_IMAGE_SIZE);
    return XST_SUCCESS;
}

static int check_qmap_header(void)
{
    UINTPTR base = (UINTPTR)QMAP_ROW1024_BASE_ADDR;

    xil_printf("\r\nQMAP row1024 header check\r\n");

    if (check_u32("magic", base + 0x00U, QMAP_MAGIC) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("version", base + 0x04U, QMAP_VERSION) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("header_bytes", base + 0x08U, QMAP_HEADER_BYTES) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("descriptor_bytes", base + 0x0CU, QMAP_DESCRIPTOR_BYTES) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("descriptor_count", base + 0x10U, QMAP_DESCRIPTOR_COUNT) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("descriptor_capacity", base + 0x14U, QMAP_DESCRIPTOR_CAPACITY) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("descriptor_table_addr",
                  base + 0x18U,
                  (u64)QMAP_ROW1024_BASE_ADDR + QMAP_DESCRIPTOR_TABLE_OFFSET) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("payload_base_addr",
                  base + 0x20U,
                  (u64)QMAP_ROW1024_BASE_ADDR + QMAP_PAYLOAD_BASE_OFFSET) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("image_base_addr", base + 0x28U, (u64)QMAP_ROW1024_BASE_ADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("image_bytes", base + 0x30U, (u64)QMAP_ROW1024_IMAGE_SIZE) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    xil_printf("QMAP row1024 header check PASSED\r\n");
    return XST_SUCCESS;
}

static void configure_gpio_registers(void)
{
    xil_printf("\r\nConfiguring AXI GPIO direction registers\r\n");

    Xil_Out32(PL_CTRL_STATUS_BASE + AXI_GPIO_TRI_OFFSET, 0x00000000U);
    Xil_Out32(PL_CTRL_STATUS_BASE + AXI_GPIO_TRI2_OFFSET, 0xFFFFFFFFU);
    Xil_Out32(PL_RESULT_BASE + AXI_GPIO_TRI_OFFSET, 0xFFFFFFFFU);
    Xil_Out32(PL_RESULT_BASE + AXI_GPIO_TRI2_OFFSET, 0xFFFFFFFFU);

    Xil_Out32(PL_CTRL_STATUS_BASE + AXI_GPIO_DATA_OFFSET, 0x00000000U);

    xil_printf("Control/status GPIO base ");
    print_addr(PL_CTRL_STATUS_BASE);
    xil_printf("\r\n");
    xil_printf("Result GPIO base ");
    print_addr(PL_RESULT_BASE);
    xil_printf("\r\n");
}

static u32 read_pl_status(void)
{
    return Xil_In32(PL_CTRL_STATUS_BASE + AXI_GPIO_DATA2_OFFSET) & 0x0000000FU;
}

static void print_pl_status(u32 status)
{
    xil_printf("PL status raw 0x%x compare_match=%d error=%d done=%d busy=%d\r\n",
               status,
               (status & PL_STATUS_COMPARE_MATCH_MASK) ? 1 : 0,
               (status & PL_STATUS_ERROR_MASK) ? 1 : 0,
               (status & PL_STATUS_DONE_MASK) ? 1 : 0,
               (status & PL_STATUS_BUSY_MASK) ? 1 : 0);
}

static void pulse_control(u32 mask)
{
    Xil_Out32(PL_CTRL_STATUS_BASE + AXI_GPIO_DATA_OFFSET, mask);
    delay_cycles(SHORT_DELAY_CYCLES);
    Xil_Out32(PL_CTRL_STATUS_BASE + AXI_GPIO_DATA_OFFSET, 0x00000000U);
    delay_cycles(SHORT_DELAY_CYCLES);
}

static int run_pl_compute(void)
{
    xil_printf("\r\nStarting PL QMAP row1024 compute path\r\n");
    debug_pl_status(read_pl_status());

    xil_printf("Clearing sticky PL status\r\n");
    pulse_control(PL_CTRL_CLEAR_MASK);
    xil_printf("After clear: ");
    u32 status_after_clear = read_pl_status();
    debug_pl_status(status_after_clear);
    print_pl_status(status_after_clear);

    xil_printf("Pulsing start\r\n");
    pulse_control(PL_CTRL_START_MASK);
    debug_pl_status(read_pl_status());

    for (u32 attempt = 0U; attempt < PL_STATUS_POLL_ATTEMPTS; ++attempt) {
        u32 status = read_pl_status();
        debug_pl_status(status);

        if ((status & PL_STATUS_DONE_MASK) != 0U) {
            xil_printf("PL row1024 compute completed at poll %d/%d\r\n",
                       (int)(attempt + 1U), (int)PL_STATUS_POLL_ATTEMPTS);
            print_pl_status(status);

            if ((status & PL_STATUS_ERROR_MASK) != 0U) {
                xil_printf("FAIL PL reported error\r\n");
                return XST_FAILURE;
            }
            if ((status & PL_STATUS_COMPARE_MATCH_MASK) == 0U) {
                xil_printf("FAIL PL did not report compare_match\r\n");
                return XST_FAILURE;
            }
            if (status != PL_STATUS_EXPECTED_DONE_MATCH) {
                xil_printf("FAIL expected final PL status 0x%x got 0x%x\r\n",
                           PL_STATUS_EXPECTED_DONE_MATCH, status);
                return XST_FAILURE;
            }

            return XST_SUCCESS;
        }

        if (attempt == 0U || ((attempt + 1U) % 1000U) == 0U) {
            xil_printf("PL row1024 compute poll %d/%d: ",
                       (int)(attempt + 1U), (int)PL_STATUS_POLL_ATTEMPTS);
            print_pl_status(status);
        }

        delay_cycles(10000U);
    }

    xil_printf("FAIL PL row1024 compute timed out\r\n");
    print_pl_status(read_pl_status());
    return XST_FAILURE;
}

static int check_pl_results(void)
{
    u32 row_sum_low32 = Xil_In32(PL_RESULT_BASE + AXI_GPIO_DATA_OFFSET);
    u32 expected_low32 = Xil_In32(PL_RESULT_BASE + AXI_GPIO_DATA2_OFFSET);

    xil_printf("\r\nPL row1024 result GPIO check\r\n");
    xil_printf("row_sum_q26_low32 = 0x%x (%d)\r\n", row_sum_low32, (s32)row_sum_low32);
    xil_printf("expected_row_sum_q26_low32 = 0x%x (%d)\r\n", expected_low32, (s32)expected_low32);

    if (row_sum_low32 != EXPECTED_ROW_SUM_Q26_LOW32) {
        xil_printf("FAIL row_sum_q26_low32 expected 0x%x got 0x%x\r\n",
                   EXPECTED_ROW_SUM_Q26_LOW32, row_sum_low32);
        return XST_FAILURE;
    }
    if (expected_low32 != EXPECTED_ROW_SUM_Q26_LOW32) {
        xil_printf("FAIL expected_row_sum_q26_low32 expected 0x%x got 0x%x\r\n",
                   EXPECTED_ROW_SUM_Q26_LOW32, expected_low32);
        return XST_FAILURE;
    }

    xil_printf("PL row1024 result GPIO check PASSED\r\n");
    return XST_SUCCESS;
}

int main(void)
{
    debug_mark(DBG_STAGE_MAIN_START);
    debug_detail(0U);
    debug_pl_status(0U);

    xil_printf("\r\nQMAP row1024 PL compute smoke application\r\n");
    xil_printf("Using 64-bit UINTPTR addresses\r\n");
    xil_printf("Embedded image SHA256 %s\r\n", QMAP_ROW1024_IMAGE_SHA256);

    configure_gpio_registers();
    debug_mark(DBG_STAGE_GPIO_CONFIGURED);

    if (wait_ddr4_ready() != XST_SUCCESS) {
        debug_mark(DBG_STAGE_FAIL_DDR4);
        while (1) {
            xil_printf("heartbeat: DDR4 controller not ready\r\n");
            delay_spin();
        }
    }
    debug_mark(DBG_STAGE_DDR4_READY);

    debug_mark(DBG_STAGE_WRITE_QMAP_BEGIN);
    if (write_qmap_image() != XST_SUCCESS) {
        debug_mark(DBG_STAGE_FAIL_WRITE_QMAP);
        xil_printf("\r\nQMAP row1024 PL compute smoke FAILED\r\n");
        while (1) {
            delay_spin();
        }
    }
    debug_mark(DBG_STAGE_WRITE_QMAP_DONE);

    debug_mark(DBG_STAGE_READBACK_BEGIN);
    if (compare_qmap_readback() != XST_SUCCESS) {
        debug_mark(DBG_STAGE_FAIL_READBACK);
        xil_printf("\r\nQMAP row1024 PL compute smoke FAILED\r\n");
        while (1) {
            delay_spin();
        }
    }
    debug_mark(DBG_STAGE_READBACK_DONE);

    debug_mark(DBG_STAGE_HEADER_BEGIN);
    if (check_qmap_header() != XST_SUCCESS) {
        debug_mark(DBG_STAGE_FAIL_HEADER);
        xil_printf("\r\nQMAP row1024 PL compute smoke FAILED\r\n");
        while (1) {
            delay_spin();
        }
    }
    debug_mark(DBG_STAGE_HEADER_DONE);

    debug_mark(DBG_STAGE_PL_BEGIN);
    if (run_pl_compute() != XST_SUCCESS) {
        debug_mark(DBG_STAGE_FAIL_PL);
        debug_detail(read_pl_status());
        xil_printf("\r\nQMAP row1024 PL compute smoke FAILED\r\n");
        while (1) {
            delay_spin();
        }
    }
    debug_mark(DBG_STAGE_PL_DONE);

    debug_mark(DBG_STAGE_RESULT_BEGIN);
    if (check_pl_results() != XST_SUCCESS) {
        debug_mark(DBG_STAGE_FAIL_RESULT);
        debug_detail(read_pl_status());
        xil_printf("\r\nQMAP row1024 PL compute smoke FAILED\r\n");
        while (1) {
            delay_spin();
        }
    }

    debug_mark(DBG_STAGE_PASS);
    debug_detail(read_pl_status());

    xil_printf("\r\nQMAP row1024 PL compute smoke PASSED\r\n");
    while (1) {
        xil_printf("heartbeat\r\n");
        delay_spin();
    }
}
