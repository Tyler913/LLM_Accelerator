#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__has_include)
#  if __has_include("xparameters.h")
#    include "xparameters.h"
#  endif
#endif

#include "qmap_one_token_runtime.h"

#if defined(QOT_MODEL_BOARD_SMOKE)
#include "qmap_model_config_generated.h"
#endif

#ifndef QOT_BASEADDR
#  if defined(XPAR_QMAP_ONE_TOKEN_AXI_BD_0_BASEADDR)
#    define QOT_BASEADDR ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXI_BD_0_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXI_BD_0_S_AXI_BASEADDR)
#    define QOT_BASEADDR ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXI_BD_0_S_AXI_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXI_BD_0_S_AXI_CTRL_BASEADDR)
#    define QOT_BASEADDR ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXI_BD_0_S_AXI_CTRL_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_BASEADDR)
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

#if defined(QOT_MODEL_BOARD_SMOKE)
#ifndef QOT_DDR4_STATUS_BASE
#define QOT_DDR4_STATUS_BASE ((uintptr_t)UINT64_C(0x00000000A0010000))
#endif

#ifndef QOT_MODEL_POLL_ATTEMPTS
#define QOT_MODEL_POLL_ATTEMPTS 200000000u
#endif

#ifndef QOT_MODEL_PROGRESS_INTERVAL
#define QOT_MODEL_PROGRESS_INTERVAL 1000000u
#endif

#define QOT_DDR4_CALIB_COMPLETE_MASK 0x00000001u
#define QOT_DDR4_UI_RESET_MASK       0x00000002u
#define QOT_DDR4_AXI_RESETN_MASK     0x00000004u
#define QOT_QMAP_MAGIC               0x50414D51u

#define QOT_MODEL_INPUT_TOKEN0       374u
#define QOT_MODEL_EXPECTED_TOKEN0    28458u
#define QOT_MODEL_EXPECTED_SCORE0    INT64_C(1227344433)
#define QOT_MODEL_EXPECTED_TOKEN1    64u
#define QOT_MODEL_EXPECTED_SCORE1    INT64_C(1015661901)
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

#if defined(QOT_MODEL_BOARD_SMOKE)
static uint32_t read_phys32(uint64_t address)
{
    return *(volatile const uint32_t *)(uintptr_t)address;
}

static int check_model_memory_ready(void)
{
    uint32_t ddr_status = read_phys32((uint64_t)QOT_DDR4_STATUS_BASE);
    uint32_t qkv_magic = read_phys32(qot_model_layer_qmap_bases[0].qkv);
    uint32_t input_norm_magic =
        read_phys32(qot_model_layer_qmap_bases[0].input_norm);
    uint32_t final_tail_magic = read_phys32(QOT_MODEL_FINAL_TAIL_QMAP_BASE);

    printf("DDR4 status=0x%08" PRIx32 "\n", ddr_status);
    if ((ddr_status & QOT_DDR4_CALIB_COMPLETE_MASK) == 0u ||
        (ddr_status & QOT_DDR4_UI_RESET_MASK) != 0u ||
        (ddr_status & QOT_DDR4_AXI_RESETN_MASK) == 0u) {
        printf("FAIL PL DDR4 is not calibrated and out of reset\n");
        return 1;
    }
    if (qkv_magic != QOT_QMAP_MAGIC ||
        input_norm_magic != QOT_QMAP_MAGIC ||
        final_tail_magic != QOT_QMAP_MAGIC) {
        printf(
            "FAIL runtime image sentinel qkv=0x%08" PRIx32
            " input_norm=0x%08" PRIx32 " final_tail=0x%08" PRIx32 "\n",
            qkv_magic,
            input_norm_magic,
            final_tail_magic);
        printf("Load pl_ddr_binary_segments.json before starting the model smoke.\n");
        return 1;
    }
    printf("PASS PL DDR4 and runtime image sentinels\n");
    return 0;
}

static int poll_model_done(uintptr_t base, qot_result_t *result)
{
    uint32_t poll;

    for (poll = 0u; poll < QOT_MODEL_POLL_ATTEMPTS; ++poll) {
        uint32_t status = qot_read32(base, QOT_REG_STATUS);

        if (QOT_STATUS_HAS_ANY_ERROR(status)) {
            (void)qot_read_result(base, result);
            return QOT_ERR_STATUS;
        }
        if ((status & QOT_STATUS_DONE_STICKY_MASK) != 0u) {
            return qot_read_result(base, result);
        }
        if (QOT_MODEL_PROGRESS_INTERVAL != 0u &&
            ((poll + 1u) % QOT_MODEL_PROGRESS_INTERVAL) == 0u) {
            printf(
                "poll=%" PRIu32 " status=0x%08" PRIx32
                " state=0x%02" PRIx32 " phase=0x%02" PRIx32 "\n",
                poll + 1u,
                status,
                qot_status_state(status),
                qot_status_phase(status));
        }
    }

    (void)qot_read_result(base, result);
    return QOT_ERR_TIMEOUT;
}

static int run_and_check_token(
    uintptr_t base,
    const qot_run_config_t *model_config,
    uint32_t input_token,
    uint32_t position,
    uint32_t expected_token,
    int64_t expected_score)
{
    qot_run_config_t config = *model_config;
    qot_result_t result = {0};
    int rc;

    config.input_token_id = input_token;
    config.position = position;
    printf(
        "Starting token position=%" PRIu32 " input=%" PRIu32 "\n",
        position,
        input_token);

    rc = qot_configure_run(base, &config);
    if (rc == QOT_OK) {
        rc = qot_start(base);
    }
    if (rc == QOT_OK) {
        rc = poll_model_done(base, &result);
    } else {
        (void)qot_read_result(base, &result);
    }
    print_result(&result);

    if (rc != QOT_OK) {
        printf("FAIL token position=%" PRIu32 " runtime rc=%d\n", position, rc);
        return 1;
    }
    if (result.token_id != expected_token || result.score_q26 != expected_score) {
        printf(
            "FAIL token position=%" PRIu32
            " expected token=%" PRIu32 " score=%" PRId64 "\n",
            position,
            expected_token,
            expected_score);
        return 1;
    }
    if (result.layers_started != QOT_MODEL_LAYER_COUNT ||
        result.layers_completed != QOT_MODEL_LAYER_COUNT ||
        result.layer_done_mask != UINT32_C(0x0FFFFFFF) ||
        result.layer_error_mask != 0u) {
        printf("FAIL token position=%" PRIu32 " layer completion contract\n", position);
        return 1;
    }

    printf(
        "PASS token position=%" PRIu32 " output=%" PRIu32
        " score=%" PRId64 "\n",
        position,
        result.token_id,
        result.score_q26);
    return 0;
}
#endif

int main(void)
{
#if defined(QOT_MODEL_BOARD_SMOKE)
    qot_run_config_t model_config;

    printf("Qwen3-0.6B full28 persistent two-token board smoke\n");
    printf("QOT_BASEADDR=0x%016" PRIxPTR "\n", (uintptr_t)QOT_BASEADDR);
    if ((uintptr_t)QOT_BASEADDR == (uintptr_t)0u) {
        printf("FAIL no one-token AXI-Lite base address in xparameters.h\n");
        return 1;
    }
    if (check_model_memory_ready() != 0) {
        return 1;
    }

    model_config = qot_model_default_run_config();
    if (run_and_check_token(
            (uintptr_t)QOT_BASEADDR,
            &model_config,
            QOT_MODEL_INPUT_TOKEN0,
            0u,
            QOT_MODEL_EXPECTED_TOKEN0,
            QOT_MODEL_EXPECTED_SCORE0) != 0) {
        return 1;
    }
    if (run_and_check_token(
            (uintptr_t)QOT_BASEADDR,
            &model_config,
            QOT_MODEL_EXPECTED_TOKEN0,
            1u,
            QOT_MODEL_EXPECTED_TOKEN1,
            QOT_MODEL_EXPECTED_SCORE1) != 0) {
        return 1;
    }

    printf("PASS Qwen3-0.6B full28 persistent two-token board smoke\n");
    return 0;
#else
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
#endif
}
