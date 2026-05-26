#include "xil_io.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xstatus.h"

#ifndef XPAR_XBRAM_0_BASEADDR
#define XPAR_XBRAM_0_BASEADDR 0x80000000U
#define XPAR_XBRAM_0_HIGHADDR 0x80001FFFU
#endif

#define BRAM_BASE ((UINTPTR)XPAR_XBRAM_0_BASEADDR)
#define BRAM_HIGH ((UINTPTR)XPAR_XBRAM_0_HIGHADDR)
#define HEARTBEAT_DELAY_CYCLES 100000000U

typedef struct {
    u32 offset;
    u32 value;
} TestWord;

static int check_word(u32 offset, u32 expected)
{
    UINTPTR addr = BRAM_BASE + offset;
    Xil_Out32(addr, expected);

    u32 actual = Xil_In32(addr);
    if (actual != expected) {
        xil_printf("FAIL offset 0x%x addr 0x%x expected 0x%x got 0x%x\r\n",
                   offset, (u32)addr, expected, actual);
        return XST_FAILURE;
    }

    xil_printf("PASS offset 0x%x addr 0x%x value 0x%x\r\n",
               offset, (u32)addr, actual);
    return XST_SUCCESS;
}

static void delay_spin(void)
{
    for (volatile u32 i = 0U; i < HEARTBEAT_DELAY_CYCLES; ++i) {
    }
}

static int run_bram_test(void)
{
    const u32 last_word_offset = (u32)(BRAM_HIGH - BRAM_BASE - 3U);
    const TestWord tests[] = {
        {0x0000U, 0xA5A50000U},
        {0x0004U, 0x5A5A0001U},
        {0x0008U, 0x12345678U},
        {0x0400U, 0xDEADBEEFU},
        {last_word_offset, 0xC001D00DU},
    };

    for (u32 i = 0U; i < (u32)(sizeof(tests) / sizeof(tests[0])); ++i) {
        if (check_word(tests[i].offset, tests[i].value) != XST_SUCCESS) {
            xil_printf("AXI BRAM smoke test FAILED\r\n");
            return XST_FAILURE;
        }
    }

    xil_printf("AXI BRAM smoke test PASSED\r\n");
    return XST_SUCCESS;
}

int main(void)
{
    u32 iteration = 0U;

    xil_printf("\r\nAXI BRAM smoke test\r\n");
    xil_printf("BRAM base 0x%x high 0x%x\r\n", (u32)BRAM_BASE, (u32)BRAM_HIGH);

    while (1) {
        xil_printf("\r\nIteration %d\r\n", (int)iteration++);
        (void)run_bram_test();
        delay_spin();
    }
}
