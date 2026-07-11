#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__has_include)
#  if __has_include("xparameters.h")
#    include "xparameters.h"
#  endif
#endif

#include "qmap_one_token_runtime.h"

#ifndef QOT_BASEADDR
#  if defined(XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_BASEADDR)
#    define QOT_BASEADDR ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_S_AXI_BASEADDR)
#    define QOT_BASEADDR ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_S_AXI_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_S_AXI_CTRL_BASEADDR)
#    define QOT_BASEADDR ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_S_AXI_CTRL_BASEADDR)
#  else
#    define QOT_BASEADDR ((uintptr_t)0u)
#  endif
#endif

#ifndef QOT_POLL_ATTEMPTS
#define QOT_POLL_ATTEMPTS 1000000u
#endif

static void print_result(const qot_result_t *result)
{
    printf("status=0x%08" PRIx32 " busy=%u done=%u err=%u cmd_err=%u state=0x%02" PRIx32 " phase=0x%02" PRIx32 "\n",
           result->status,
           (result->status & QOT_STATUS_BUSY_MASK) ? 1u : 0u,
           (result->status & QOT_STATUS_DONE_STICKY_MASK) ? 1u : 0u,
           (result->status & QOT_STATUS_ERROR_STICKY_MASK) ? 1u : 0u,
           (result->status & QOT_STATUS_COMMAND_ERR_MASK) ? 1u : 0u,
           qot_status_state(result->status),
           qot_status_phase(result->status));
    printf("layers started=%" PRIu32 " completed=%" PRIu32 " done_mask=0x%08" PRIx32 " error_mask=0x%08" PRIx32 "\n",
           result->layers_started,
           result->layers_completed,
           result->layer_done_mask,
           result->layer_error_mask);
    printf("memory rd=%" PRIu32 "/%" PRIu32 " wr=%" PRIu32 "/%" PRIu32 "\n",
           result->mem_read_reqs,
           result->mem_read_words,
           result->mem_write_reqs,
           result->mem_write_words);
    printf("token=%" PRIu32 " score_q26=%" PRId64 " last_output=0x%016" PRIx64 " tail_hidden=0x%016" PRIx64 "\n",
           result->token_id,
           result->score_q26,
           result->last_output_base,
           result->tail_hidden_base);
}

int main(void)
{
    qot_result_t result;
    int rc;

    printf("QMAP one-token AXI-Lite control smoke\n");
    printf("QOT_BASEADDR=0x%016" PRIxPTR "\n", (uintptr_t)QOT_BASEADDR);

    if ((uintptr_t)QOT_BASEADDR == (uintptr_t)0u) {
        printf("No QOT_BASEADDR is defined. After Vivado BD export, define QOT_BASEADDR\n");
        printf("from xparameters.h or the Vitis compiler command line before running this app.\n");
        return 1;
    }

    rc = qot_run_no_memory_validation_smoke((uintptr_t)QOT_BASEADDR,
                                            QOT_POLL_ATTEMPTS,
                                            &result);
    print_result(&result);

    if (rc != QOT_OK) {
        printf("FAIL qot_run_no_memory_validation_smoke rc=%d\n", rc);
        return 1;
    }

    printf("PASS qot_run_no_memory_validation_smoke\n");
    return 0;
}
