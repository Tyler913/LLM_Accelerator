#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xil_types.h"
#include "xparameters.h"
#include "xstatus.h"

#ifndef XPAR_XBRAM_0_BASEADDR
#define XPAR_XBRAM_0_BASEADDR 0xA0000000U
#define XPAR_XBRAM_0_HIGHADDR 0xA0001FFFU
#endif

#ifndef XPAR_DDR4_0_BASEADDRESS
#define XPAR_DDR4_0_BASEADDRESS 0x0000000400000000ULL
#define XPAR_DDR4_0_HIGHADDRESS 0x000000041FFFFFFFULL
#endif

#if defined(XPAR_XGPIO_0_BASEADDR)
#define DDR4_STATUS_GPIO_BASE XPAR_XGPIO_0_BASEADDR
#elif defined(XPAR_AXI_GPIO_0_BASEADDR)
#define DDR4_STATUS_GPIO_BASE XPAR_AXI_GPIO_0_BASEADDR
#elif defined(XPAR_AXI_GPIO_0_BASEADDRESS)
#define DDR4_STATUS_GPIO_BASE XPAR_AXI_GPIO_0_BASEADDRESS
#else
#define DDR4_STATUS_GPIO_BASE 0xA0010000U
#endif

#define BRAM_BASE ((UINTPTR)XPAR_XBRAM_0_BASEADDR)
#define BRAM_HIGH ((UINTPTR)XPAR_XBRAM_0_HIGHADDR)
#define PL_DDR4_BASE ((UINTPTR)XPAR_DDR4_0_BASEADDRESS)
#define PL_DDR4_HIGH ((UINTPTR)XPAR_DDR4_0_HIGHADDRESS)
#define DDR4_STATUS_BASE ((UINTPTR)DDR4_STATUS_GPIO_BASE)

#define DDR4_STATUS_CALIB_COMPLETE_MASK 0x00000001U
#define DDR4_STATUS_UI_RESET_MASK 0x00000002U
#define DDR4_STATUS_AXI_RESETN_MASK 0x00000004U

#define HEARTBEAT_DELAY_CYCLES 100000000U
#define STATUS_POLL_DELAY_CYCLES 1000000U
#define STATUS_POLL_ATTEMPTS 200U

typedef struct {
    const char *name;
    UINTPTR addr;
    u32 value;
} TestWord;

static void print_addr(UINTPTR addr)
{
    xil_printf("0x%x_%08x", (u32)(addr >> 32), (u32)addr);
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

    xil_printf("DDR4 is not ready; skipping PL DDR4 memory access to avoid AXI hang\r\n");
    return XST_FAILURE;
}

static int check_word(const char *space, const TestWord *test)
{
    xil_printf("TRY %s %s write addr ", space, test->name);
    print_addr(test->addr);
    xil_printf(" value 0x%x\r\n", test->value);

    Xil_Out32(test->addr, test->value);

    xil_printf("WRITE returned for %s %s\r\n", space, test->name);
    Xil_DCacheFlushRange((INTPTR)test->addr, sizeof(u32));
    Xil_DCacheInvalidateRange((INTPTR)test->addr, sizeof(u32));

    u32 actual = Xil_In32(test->addr);
    if (actual != test->value) {
        xil_printf("FAIL %s %s addr ", space, test->name);
        print_addr(test->addr);
        xil_printf(" expected 0x%x got 0x%x\r\n", test->value, actual);
        return XST_FAILURE;
    }

    xil_printf("PASS %s %s addr ", space, test->name);
    print_addr(test->addr);
    xil_printf(" value 0x%x\r\n", actual);
    return XST_SUCCESS;
}

static int run_bram_test(void)
{
    const TestWord tests[] = {
        {"base", BRAM_BASE + 0x0000U, 0xA5A50000U},
        {"base+4", BRAM_BASE + 0x0004U, 0x5A5A0001U},
        {"base+8", BRAM_BASE + 0x0008U, 0x12345678U},
        {"middle", BRAM_BASE + 0x0400U, 0xDEADBEEFU},
        {"last", BRAM_HIGH - 3U, 0xC001D00DU},
    };

    xil_printf("\r\nAXI BRAM smoke test\r\n");
    xil_printf("BRAM range ");
    print_addr(BRAM_BASE);
    xil_printf(" - ");
    print_addr(BRAM_HIGH);
    xil_printf("\r\n");

    for (u32 i = 0U; i < (u32)(sizeof(tests) / sizeof(tests[0])); ++i) {
        if (check_word("BRAM", &tests[i]) != XST_SUCCESS) {
            return XST_FAILURE;
        }
    }

    xil_printf("AXI BRAM smoke test PASSED\r\n");
    return XST_SUCCESS;
}

static int run_pl_ddr4_test(void)
{
    const TestWord tests[] = {
        {"base", PL_DDR4_BASE + 0x00000000ULL, 0xD4D40000U},
        {"base+4", PL_DDR4_BASE + 0x00000004ULL, 0x4D4D0001U},
        {"near", PL_DDR4_BASE + 0x00001000ULL, 0x13572468U},
        {"middle", PL_DDR4_BASE + 0x10000000ULL, 0x24681357U},
        {"last", PL_DDR4_HIGH - 3ULL, 0xDD44AA55U},
    };

    xil_printf("\r\nPL DDR4 smoke test\r\n");
    xil_printf("PL DDR4 range ");
    print_addr(PL_DDR4_BASE);
    xil_printf(" - ");
    print_addr(PL_DDR4_HIGH);
    xil_printf("\r\n");

    for (u32 i = 0U; i < (u32)(sizeof(tests) / sizeof(tests[0])); ++i) {
        if (check_word("PL_DDR4", &tests[i]) != XST_SUCCESS) {
            return XST_FAILURE;
        }
    }

    xil_printf("PL DDR4 smoke test PASSED\r\n");
    return XST_SUCCESS;
}

int main(void)
{
    xil_printf("\r\nPL DDR4 smoke application\r\n");
    xil_printf("Using 64-bit UINTPTR addresses\r\n");

    if (run_bram_test() != XST_SUCCESS) {
        xil_printf("Stopping before PL DDR4 test because BRAM path failed\r\n");
        while (1) {
            delay_spin();
        }
    }

    xil_printf("\r\nWaiting before PL DDR4 access...\r\n");
    delay_spin();

    if (wait_ddr4_ready() != XST_SUCCESS) {
        while (1) {
            xil_printf("heartbeat: DDR4 controller not ready\r\n");
            delay_spin();
        }
    }

    if (run_pl_ddr4_test() != XST_SUCCESS) {
        xil_printf("PL DDR4 smoke test FAILED\r\n");
        while (1) {
            delay_spin();
        }
    }

    xil_printf("\r\nAll smoke tests PASSED\r\n");
    while (1) {
        xil_printf("heartbeat\r\n");
        delay_spin();
    }
}
