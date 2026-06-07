#include "qmap_dot64_image.h"

#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xil_types.h"
#include "xparameters.h"
#include "xstatus.h"

#if defined(XPAR_XGPIO_0_BASEADDR)
#define DDR4_STATUS_GPIO_BASE XPAR_XGPIO_0_BASEADDR
#elif defined(XPAR_AXI_GPIO_0_BASEADDR)
#define DDR4_STATUS_GPIO_BASE XPAR_AXI_GPIO_0_BASEADDR
#elif defined(XPAR_AXI_GPIO_0_BASEADDRESS)
#define DDR4_STATUS_GPIO_BASE XPAR_AXI_GPIO_0_BASEADDRESS
#else
#define DDR4_STATUS_GPIO_BASE 0xA0010000U
#endif

#define DDR4_STATUS_BASE ((UINTPTR)DDR4_STATUS_GPIO_BASE)

#define DDR4_STATUS_CALIB_COMPLETE_MASK 0x00000001U
#define DDR4_STATUS_UI_RESET_MASK 0x00000002U
#define DDR4_STATUS_AXI_RESETN_MASK 0x00000004U

#define QMAP_MAGIC 0x50414D51U
#define QMAP_VERSION 1U
#define QMAP_HEADER_BYTES 256U
#define QMAP_DESCRIPTOR_BYTES 128U
#define QMAP_DESCRIPTOR_COUNT 4U
#define QMAP_DESCRIPTOR_CAPACITY 8U
#define QMAP_DESCRIPTOR_TABLE_OFFSET 0x0100U
#define QMAP_PAYLOAD_BASE_OFFSET 0x0500U

#define QMAP_DTYPE_I64 8U
#define QMAP_DTYPE_PACKED_Q4_S4 16U
#define QMAP_DTYPE_U16_Q2_14 17U
#define QMAP_DTYPE_I16_Q4_12 18U

#define HEARTBEAT_DELAY_CYCLES 100000000U
#define STATUS_POLL_DELAY_CYCLES 1000000U
#define STATUS_POLL_ATTEMPTS 200U

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

static u32 read_ddr4_status(void)
{
    return Xil_In32(DDR4_STATUS_BASE);
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

    for (u32 attempt = 0U; attempt < STATUS_POLL_ATTEMPTS; ++attempt) {
        u32 status = read_ddr4_status();
        if ((status & DDR4_STATUS_CALIB_COMPLETE_MASK) != 0U &&
            (status & DDR4_STATUS_UI_RESET_MASK) == 0U &&
            (status & DDR4_STATUS_AXI_RESETN_MASK) != 0U) {
            xil_printf("DDR4 controller and AXI reset path report ready\r\n");
            print_ddr4_status(status);
            return XST_SUCCESS;
        }

        if (attempt == 0U || ((attempt + 1U) % 20U) == 0U) {
            xil_printf("DDR4 not ready poll %d/%d: ",
                       (int)(attempt + 1U), (int)STATUS_POLL_ATTEMPTS);
            print_ddr4_status(status);
        }

        delay_cycles(STATUS_POLL_DELAY_CYCLES);
    }

    xil_printf("DDR4 is not ready; skipping QMAP access to avoid AXI hang\r\n");
    return XST_FAILURE;
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
    xil_printf("\r\nWriting QMAP image\r\n");
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
    xil_printf("\r\nQMAP readback byte compare\r\n");

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

static int check_descriptor_slot(u32 slot, u32 tensor_id, u32 role, u32 dtype, u64 base_addr, u64 nbytes)
{
    UINTPTR desc = (UINTPTR)QMAP_DOT64_BASE_ADDR +
                   QMAP_DESCRIPTOR_TABLE_OFFSET +
                   (UINTPTR)slot * QMAP_DESCRIPTOR_BYTES;

    xil_printf("\r\nQMAP descriptor slot %d check\r\n", (int)slot);
    if (check_u32("tensor_id", desc + 0x00U, tensor_id) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("role", desc + 0x04U, role) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("dtype", desc + 0x08U, dtype) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("base_addr", desc + 0x20U, base_addr) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("nbytes", desc + 0x28U, nbytes) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

static int check_qmap_descriptors(void)
{
    u64 base = (u64)QMAP_DOT64_BASE_ADDR;

    if (check_descriptor_slot(0U, 1U, 1U, QMAP_DTYPE_I16_Q4_12, base + 0x0500U, 128U) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_descriptor_slot(1U, 2U, 2U, QMAP_DTYPE_PACKED_Q4_S4, base + 0x0580U, 32U) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_descriptor_slot(2U, 3U, 3U, QMAP_DTYPE_U16_Q2_14, base + 0x05A0U, 2U) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_descriptor_slot(3U, 4U, 5U, QMAP_DTYPE_I64, base + 0x05C0U, 16U) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    xil_printf("\r\nQMAP descriptor check PASSED\r\n");
    return XST_SUCCESS;
}

static int check_qmap_payload_spots(void)
{
    UINTPTR base = (UINTPTR)QMAP_DOT64_BASE_ADDR;
    xil_printf("\r\nQMAP payload spot check\r\n");

    if (check_u32("activation first word", base + 0x0500U, load_le32(&qmap_dot64_image[0x0500U])) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("weight first word", base + 0x0580U, load_le32(&qmap_dot64_image[0x0580U])) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u32("scale word", base + 0x05A0U, load_le32(&qmap_dot64_image[0x05A0U])) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("partial_sum_int64", base + 0x05C0U, 24751ULL) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (check_u64("scaled_sum_q26_int64", base + 0x05C8U, 3019622ULL) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    xil_printf("QMAP payload spot check PASSED\r\n");
    return XST_SUCCESS;
}

int main(void)
{
    xil_printf("\r\nQMAP load/readback smoke application\r\n");
    xil_printf("Using 64-bit UINTPTR addresses\r\n");
    xil_printf("Embedded image SHA256 %s\r\n", QMAP_DOT64_IMAGE_SHA256);

    if (wait_ddr4_ready() != XST_SUCCESS) {
        while (1) {
            xil_printf("heartbeat: DDR4 controller not ready\r\n");
            delay_spin();
        }
    }

    if (write_qmap_image() != XST_SUCCESS ||
        compare_qmap_readback() != XST_SUCCESS ||
        check_qmap_header() != XST_SUCCESS ||
        check_qmap_descriptors() != XST_SUCCESS ||
        check_qmap_payload_spots() != XST_SUCCESS) {
        xil_printf("\r\nQMAP load/readback FAILED\r\n");
        while (1) {
            delay_spin();
        }
    }

    xil_printf("\r\nQMAP load/readback PASSED\r\n");
    while (1) {
        xil_printf("heartbeat\r\n");
        delay_spin();
    }
}
