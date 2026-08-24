#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__has_include)
#  if __has_include("xparameters.h")
#    include "xparameters.h"
#  endif
#endif

#include "qmap_model_config_generated.h"
#include "qot_protocol.h"
#include "qot_session.h"
#include "qot_uart.h"
#include "qtk_text_tokenizer.h"
#include "qtk_tokenizer_runtime.h"

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

#ifndef QOT_DDR4_STATUS_BASE
#define QOT_DDR4_STATUS_BASE ((uintptr_t)UINT64_C(0x00000000A0010000))
#endif

#ifndef QOT_MODEL_POLL_ATTEMPTS
#define QOT_MODEL_POLL_ATTEMPTS UINT32_C(200000000)
#endif

#ifndef QOT_MODEL_PROGRESS_INTERVAL
#define QOT_MODEL_PROGRESS_INTERVAL UINT32_C(1000000)
#endif

#define QOT_DDR4_CALIB_COMPLETE_MASK UINT32_C(0x00000001)
#define QOT_DDR4_UI_RESET_MASK       UINT32_C(0x00000002)
#define QOT_DDR4_AXI_RESETN_MASK     UINT32_C(0x00000004)
#define QOT_QMAP_MAGIC               UINT32_C(0x50414D51)
#define QOT_FULL28_DONE_MASK         UINT32_C(0x0FFFFFFF)
#define QOT_TOKENIZER_WORKSPACE_CAPACITY \
    (QOT_PROTOCOL_MAX_LINE_LENGTH * 4u)

extern const uint8_t qot_tokenizer_asset_start[];
extern const uint8_t qot_tokenizer_asset_end[];

typedef struct qot_board_runner_context {
    uint32_t progress_interval;
} qot_board_runner_context_t;

static uint32_t qot_read_phys32(uint64_t address)
{
    return *(volatile const uint32_t *)(uintptr_t)address;
}

static int qot_check_model_memory_ready(void)
{
    uint32_t ddr_status = qot_read_phys32((uint64_t)QOT_DDR4_STATUS_BASE);
    uint32_t qkv_magic = qot_read_phys32(qot_model_layer_qmap_bases[0].qkv);
    uint32_t input_norm_magic =
        qot_read_phys32(qot_model_layer_qmap_bases[0].input_norm);
    uint32_t final_tail_magic =
        qot_read_phys32(QOT_MODEL_FINAL_TAIL_QMAP_BASE);

    printf("DDR4 status=0x%08" PRIx32 "\n", ddr_status);
    if ((ddr_status & QOT_DDR4_CALIB_COMPLETE_MASK) == 0u ||
        (ddr_status & QOT_DDR4_UI_RESET_MASK) != 0u ||
        (ddr_status & QOT_DDR4_AXI_RESETN_MASK) == 0u) {
        printf("ERROR DDR4_NOT_READY\n");
        return 1;
    }
    if (qkv_magic != QOT_QMAP_MAGIC ||
        input_norm_magic != QOT_QMAP_MAGIC ||
        final_tail_magic != QOT_QMAP_MAGIC) {
        printf("ERROR RUNTIME_SENTINEL qkv=0x%08" PRIx32
               " input_norm=0x%08" PRIx32
               " final_tail=0x%08" PRIx32 "\n",
               qkv_magic, input_norm_magic, final_tail_magic);
        return 1;
    }
    return 0;
}

/*
 * This runner owns one token's configure/start/poll sequence. A future Web
 * runner can use the same injection seam and pump lwIP inside this loop.
 */
static int qot_board_token_runner(
    void *runner_context,
    uintptr_t base,
    const qot_run_config_t *config,
    uint32_t input_token_id,
    uint32_t position,
    uint32_t max_polls,
    qot_result_t *result)
{
    qot_board_runner_context_t *context =
        (qot_board_runner_context_t *)runner_context;
    qot_run_config_t token_config;
    uint32_t poll = 0u;
    int rc;

    if (config == NULL || result == NULL) return QOT_ERR_NULL;
    token_config = *config;
    token_config.input_token_id = input_token_id;
    token_config.position = position;

    rc = qot_configure_run(base, &token_config);
    if (rc == QOT_OK) rc = qot_start(base);
    if (rc != QOT_OK) {
        (void)qot_read_result(base, result);
        return rc;
    }

    for (;;) {
        uint32_t status = qot_read32(base, QOT_REG_STATUS);

        if (QOT_STATUS_HAS_ANY_ERROR(status)) {
            (void)qot_read_result(base, result);
            return QOT_ERR_STATUS;
        }
        if ((status & QOT_STATUS_DONE_STICKY_MASK) != 0u) {
            rc = qot_read_result(base, result);
            if (rc != QOT_OK) return rc;
            if (result->layers_started != QOT_MODEL_LAYER_COUNT ||
                result->layers_completed != QOT_MODEL_LAYER_COUNT ||
                result->layer_done_mask != QOT_FULL28_DONE_MASK ||
                result->layer_error_mask != 0u) {
                return QOT_ERR_STATUS;
            }
            return QOT_OK;
        }
        ++poll;
        if (max_polls != 0u && poll >= max_polls) {
            (void)qot_read_result(base, result);
            return QOT_ERR_TIMEOUT;
        }
        if (context != NULL && context->progress_interval != 0u &&
            (poll % context->progress_interval) == 0u) {
            printf("BUSY position=%" PRIu32 " polls=%" PRIu32
                   " status=0x%08" PRIx32 "\n",
                   position, poll, status);
        }
    }
}

static int qot_emit_hex_bytes(
    void *context,
    const uint8_t *bytes,
    size_t length)
{
    size_t index;

    (void)context;
    for (index = 0u; index < length; ++index) {
        printf("%02" PRIx8, bytes[index]);
    }
    return 0;
}

static qtk_status_t qot_print_token_bytes(
    const qtk_asset_t *tokenizer,
    uint32_t generated_index,
    uint32_t token_id)
{
    qtk_token_slice_t slice;
    qtk_status_t status = qtk_token_slice(tokenizer, token_id, &slice);

    if (status == QTK_ERR_MODEL_ONLY_TOKEN) {
        printf("BYTES %" PRIu32 " UNDECODABLE\n", generated_index);
        return status;
    }
    if (status != QTK_OK) return status;
    if ((slice.flags & QTK_TOKEN_FLAG_SPECIAL) != 0u) {
        printf("BYTES %" PRIu32 " SPECIAL\n", generated_index);
        return QTK_OK;
    }

    printf("BYTES %" PRIu32 " ", generated_index);
    status = qtk_detokenize_ids(
        tokenizer, &token_id, 1u, 0, qot_emit_hex_bytes, NULL);
    printf("\n");
    return status;
}

static qtk_status_t qot_print_session_event(
    const qot_session_t *session,
    const qot_session_event_t *event,
    const qtk_asset_t *tokenizer)
{
    switch (event->kind) {
    case QOT_SESSION_EVENT_PREFILL:
        printf("PREFILL %" PRIu32 "/%" PRIu32 "\n",
               event->prompt_tokens_consumed, session->prompt_token_count);
        break;
    case QOT_SESSION_EVENT_TOKEN:
        printf("TOKEN %" PRIu32 " %" PRIu32 " %" PRId64 "\n",
               event->generated_index,
               event->result.token_id,
               event->result.score_q26);
        return qot_print_token_bytes(
            tokenizer, event->generated_index, event->result.token_id);
    case QOT_SESSION_EVENT_DONE:
        printf("DONE %" PRIu32 " %s\n",
               session->generated_count,
               qot_session_stop_reason_name(event->stop_reason));
        break;
    case QOT_SESSION_EVENT_ERROR:
        printf("ERROR HW_ERROR rc=%d position=%" PRIu32
               " input=%" PRIu32 " status=0x%08" PRIx32
               " layers=%" PRIu32 "/%" PRIu32
               " done_mask=0x%08" PRIx32
               " error_mask=0x%08" PRIx32 "\n",
               event->error_code,
               event->input_position,
               event->input_token_id,
               event->result.status,
               event->result.layers_started,
               event->result.layers_completed,
               event->result.layer_done_mask,
               event->result.layer_error_mask);
        break;
    default:
        break;
    }
    return QTK_OK;
}

int main(void)
{
    static char line[QOT_PROTOCOL_MAX_LINE_LENGTH + 1u];
    static qot_protocol_command_t command;
    static uint32_t tokenized_prompt_ids[QOT_MAX_CONTEXT];
    static uint32_t tokenizer_unicode_workspace[
        QOT_TOKENIZER_WORKSPACE_CAPACITY];
    static uint32_t tokenizer_piece_workspace[
        QOT_TOKENIZER_WORKSPACE_CAPACITY];
    qot_board_runner_context_t runner_context;
    qot_protocol_parse_result_t parse_result;
    qot_session_options_t options;
    qot_session_event_t event;
    qot_run_config_t model_config;
    qot_session_t session;
    qot_uart_t uart;
    qtk_asset_t tokenizer;
    size_t line_length;
    int rc;

    printf("Qwen3 text/token prompt demo\n");
    printf("QOT_BASEADDR=0x%016" PRIxPTR "\n", (uintptr_t)QOT_BASEADDR);
    if ((uintptr_t)QOT_BASEADDR == (uintptr_t)0u) {
        printf("ERROR NO_QOT_BASEADDR\n");
        return 1;
    }
    if (qot_check_model_memory_ready() != 0) return 1;

    {
        size_t tokenizer_size = (size_t)(
            qot_tokenizer_asset_end - qot_tokenizer_asset_start);
        qtk_status_t tokenizer_status = qtk_asset_init(
            &tokenizer, qot_tokenizer_asset_start, tokenizer_size);

        if (tokenizer_status != QTK_OK) {
            printf("ERROR TOKENIZER_ASSET rc=%d status=%s bytes=%u\n",
                   (int)tokenizer_status,
                   qtk_status_name(tokenizer_status),
                   (unsigned)tokenizer_size);
            return 1;
        }
        printf("TOKENIZER tokens=%" PRIu32 " model_vocab=%" PRIu32
               " eos=%" PRIu32 " bytes=%u\n",
               tokenizer.token_count,
               tokenizer.model_vocab_size,
               tokenizer.eos_token_id,
               (unsigned)tokenizer_size);
    }

    model_config = qot_model_default_run_config();
    runner_context.progress_interval = QOT_MODEL_PROGRESS_INTERVAL;
    rc = qot_session_init(&session,
                          (uintptr_t)QOT_BASEADDR,
                          &model_config,
                          qot_board_token_runner,
                          &runner_context);
    if (rc != QOT_OK) {
        printf("ERROR SESSION_INIT rc=%d\n", rc);
        return 1;
    }

    qot_uart_init_stdio(&uart, 1u);
    printf("READY vocab=%u context=%u\n", QOT_VOCAB_SIZE, QOT_MAX_CONTEXT);
    printf("Type HELP for the command syntax.\n");

    for (;;) {
        qot_uart_read_status_t read_status;

        printf("qot> ");
        (void)fflush(stdout);
        read_status = qot_uart_read_line(
            &uart, line, sizeof(line), &line_length);
        if (read_status == QOT_UART_READ_EOF) {
            printf("BYE\n");
            return 0;
        }
        if (read_status == QOT_UART_READ_OVERFLOW) {
            printf("ERROR LINE_TOO_LONG limit=%u\n",
                   QOT_PROTOCOL_MAX_LINE_LENGTH);
            continue;
        }
        if (read_status != QOT_UART_READ_OK) {
            printf("ERROR UART rc=%d\n", (int)read_status);
            continue;
        }

        parse_result = qot_protocol_parse(line, line_length, &command);
        if (parse_result.status != QOT_PROTOCOL_PARSE_OK) {
            printf("ERROR PARSE %s offset=%u\n",
                   qot_protocol_parse_status_name(parse_result.status),
                   (unsigned)parse_result.error_offset);
            continue;
        }
        if (command.kind == QOT_PROTOCOL_COMMAND_HELP) {
            printf("%s", qot_protocol_help_text());
            continue;
        }
        if (command.kind == QOT_PROTOCOL_COMMAND_PING) {
            printf("PONG\n");
            continue;
        }

        {
            const uint32_t *prompt_ids = command.token_ids;
            uint32_t prompt_count = command.token_count;

            if (command.kind == QOT_PROTOCOL_COMMAND_PROMPT) {
                qtk_tokenizer_requirements_t requirements;
                qtk_status_t tokenizer_status;
                size_t token_count = 0u;
                size_t token_index;

                tokenizer_status = qtk_tokenize_utf8(
                    &tokenizer,
                    (const uint8_t *)line + command.prompt_offset,
                    command.prompt_length,
                    tokenizer_unicode_workspace,
                    QOT_TOKENIZER_WORKSPACE_CAPACITY,
                    tokenizer_piece_workspace,
                    QOT_TOKENIZER_WORKSPACE_CAPACITY,
                    tokenized_prompt_ids,
                    QOT_MAX_CONTEXT,
                    &token_count,
                    &requirements);
                if (tokenizer_status != QTK_OK || token_count == 0u) {
                    printf("ERROR TOKENIZE rc=%d status=%s"
                           " unicode=%u piece=%u output=%u\n",
                           (int)tokenizer_status,
                           qtk_status_name(tokenizer_status),
                           (unsigned)requirements.unicode_codepoints,
                           (unsigned)requirements.piece_token_ids,
                           (unsigned)requirements.output_token_ids);
                    continue;
                }
                prompt_ids = tokenized_prompt_ids;
                prompt_count = (uint32_t)token_count;
                printf("PROMPT_IDS %" PRIu32, prompt_count);
                for (token_index = 0u; token_index < token_count; ++token_index) {
                    printf(" %" PRIu32, prompt_ids[token_index]);
                }
                printf("\n");
            }

            qot_session_default_options(&options);
            options.eos_token_id = tokenizer.eos_token_id;
            options.max_new_tokens = command.max_new_tokens;
            options.max_polls_per_token = QOT_MODEL_POLL_ATTEMPTS;
            rc = qot_session_begin(&session,
                                   prompt_ids,
                                   prompt_count,
                                   &options);
            if (rc != QOT_OK) {
                printf("ERROR SESSION_BEGIN rc=%d\n", rc);
                continue;
            }
            printf("START prompt=%" PRIu32 " max_new=%" PRIu32 "\n",
                   prompt_count, command.max_new_tokens);

            while (!qot_session_is_terminal(&session)) {
                qtk_status_t detokenize_status;

                rc = qot_session_step(&session, &event);
                detokenize_status = qot_print_session_event(
                    &session, &event, &tokenizer);
                if (detokenize_status != QTK_OK) {
                    printf("ERROR DETOKENIZE rc=%d status=%s token=%" PRIu32
                           "\n",
                           (int)detokenize_status,
                           qtk_status_name(detokenize_status),
                           event.result.token_id);
                    qot_session_reset(&session);
                    break;
                }
                if (rc != QOT_OK) break;
            }
        }
    }
}
