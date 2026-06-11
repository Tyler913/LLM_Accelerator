#include "qmap_dot64_image.h"

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

#define EXPECTED_PARTIAL_SUM_LOW32 24751U
#define EXPECTED_SCALED_SUM_Q26_LOW32 3019622U

#define HEARTBEAT_DELAY_CYCLES 100000000U
#define SHORT_DELAY_CYCLES 1000000U
#define DDR4_STATUS_POLL_ATTEMPTS 200U
#define PL_STATUS_POLL_ATTEMPTS 50000U

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
    xil_printf("\r\nWriting QMAP image for PL master\r\n");
    xil_printf("Target base ");
    print_addr((UINTPTR)QMAP_DOT64_BASE_ADDR);
    xil_printf(" bytes 0x%x\r\n", QMAP_DOT64_IMAGE_SIZE);

    for (u32 offset = 0U; offset < QMAP_DOT64_IMAGE_SIZE; offset += 4U) {
        u32 value = load_le32(&qmap_dot64_image[offset]);
        Xil_Out32((UINTPTR)QMAP_DOT64_BASE_ADDR + offset, value);
    }

    Xil_DCacheFlushRange((INTPTR)QMAP_DOT64_BASE_ADDR, QMAP_DOT64_IMAGE_SIZE);
    Xil_DCacheInvalidateRange((INTPTR)QMAP_DOT64_BASE_ADDR, QMAP_DOT64_IMAGE_SIZE);

    xil_printf("QMAP image write completed\r\n");
    return XST_SUCCESS;
}

static int compare_qmap_readback(void)
{
    xil_printf("\r\nQMAP readback word compare\r\n");

    for (u32 offset = 0U; offset < QMAP_DOT64_IMAGE_SIZE; offset += 4U) {
        u32 expected = load_le32(&qmap_dot64_image[offset]);
        u32 actual = Xil_In32((UINTPTR)QMAP_DOT64_BASE_ADDR + offset);

        if (actual != expected) {
            xil_printf("FAIL readback mismatch offset 0x%x addr ", offset);
            print_addr((UINTPTR)QMAP_DOT64_BASE_ADDR + offset);
            xil_printf(" expected 0x%x got 0x%x\r\n", expected, actual);
            return XST_FAILURE;
        }
    }

    xil_printf("Readback compare PASSED for %d bytes\r\n", (int)QMAP_DOT64_IMAGE_SIZE);
    return XST_SUCCESS;
}

static int check_qmap_header(void)
{
    UINTPTR base = (UINTPTR)QMAP_DOT64_BASE_ADDR;

    xil_printf("\r\nQMAP header check\r\n");

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
                  (u64)QMAP_DOT64_BASE_ADDR + QMAP_DESCRIPTOR_TABLE_OFFSET) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("payload_base_addr",
                  base + 0x20U,
                  (u64)QMAP_DOT64_BASE_ADDR + QMAP_PAYLOAD_BASE_OFFSET) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("image_base_addr", base + 0x28U, (u64)QMAP_DOT64_BASE_ADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("image_bytes", base + 0x30U, (u64)QMAP_DOT64_IMAGE_SIZE) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    xil_printf("QMAP header check PASSED\r\n");
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
    xil_printf("\r\nStarting PL QMAP dot64 compute path\r\n");

    xil_printf("Clearing sticky PL status\r\n");
    pulse_control(PL_CTRL_CLEAR_MASK);
    xil_printf("After clear: ");
    print_pl_status(read_pl_status());

    xil_printf("Pulsing start\r\n");
    pulse_control(PL_CTRL_START_MASK);

    for (u32 attempt = 0U; attempt < PL_STATUS_POLL_ATTEMPTS; ++attempt) {
        u32 status = read_pl_status();

        if ((status & PL_STATUS_DONE_MASK) != 0U) {
            xil_printf("PL compute completed at poll %d/%d\r\n",
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
            xil_printf("PL compute poll %d/%d: ",
                       (int)(attempt + 1U), (int)PL_STATUS_POLL_ATTEMPTS);
            print_pl_status(status);
        }

        delay_cycles(10000U);
    }

    xil_printf("FAIL PL compute timed out\r\n");
    print_pl_status(read_pl_status());
    return XST_FAILURE;
}

static int check_pl_results(void)
{
    u32 partial_sum = Xil_In32(PL_RESULT_BASE + AXI_GPIO_DATA_OFFSET);
    u32 scaled_sum_q26 = Xil_In32(PL_RESULT_BASE + AXI_GPIO_DATA2_OFFSET);

    xil_printf("\r\nPL result GPIO check\r\n");
    xil_printf("partial_sum_low32 = 0x%x (%d)\r\n", partial_sum, (int)partial_sum);
    xil_printf("scaled_sum_q26_low32 = 0x%x (%d)\r\n", scaled_sum_q26, (int)scaled_sum_q26);

    if (partial_sum != EXPECTED_PARTIAL_SUM_LOW32) {
        xil_printf("FAIL partial_sum expected 0x%x got 0x%x\r\n",
                   EXPECTED_PARTIAL_SUM_LOW32, partial_sum);
        return XST_FAILURE;
    }
    if (scaled_sum_q26 != EXPECTED_SCALED_SUM_Q26_LOW32) {
        xil_printf("FAIL scaled_sum_q26 expected 0x%x got 0x%x\r\n",
                   EXPECTED_SCALED_SUM_Q26_LOW32, scaled_sum_q26);
        return XST_FAILURE;
    }

    xil_printf("PL result GPIO check PASSED\r\n");
    return XST_SUCCESS;
}

int main(void)
{
    xil_printf("\r\nQMAP PL compute smoke application\r\n");
    xil_printf("Using 64-bit UINTPTR addresses\r\n");
    xil_printf("Embedded image SHA256 %s\r\n", QMAP_DOT64_IMAGE_SHA256);

    configure_gpio_registers();

    if (wait_ddr4_ready() != XST_SUCCESS) {
        while (1) {
            xil_printf("heartbeat: DDR4 controller not ready\r\n");
            delay_spin();
        }
    }

    if (write_qmap_image() != XST_SUCCESS ||
        compare_qmap_readback() != XST_SUCCESS ||
        check_qmap_header() != XST_SUCCESS ||
        run_pl_compute() != XST_SUCCESS ||
        check_pl_results() != XST_SUCCESS) {
        xil_printf("\r\nQMAP PL compute smoke FAILED\r\n");
        while (1) {
            delay_spin();
        }
    }

    xil_printf("\r\nQMAP PL compute smoke PASSED\r\n");
    while (1) {
        xil_printf("heartbeat\r\n");
        delay_spin();
    }
}
