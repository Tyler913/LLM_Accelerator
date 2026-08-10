#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qot_session.h"
#include "tokenizer_runtime/qtk_text_tokenizer.h"

#define PROMPT_TOKEN_COUNT 5u
#define GENERATED_TOKEN_COUNT 2u
#define RUNNER_CALL_COUNT 6u
#define TOKEN_WORKSPACE_CAPACITY 128u

typedef struct {
    uint32_t input_tokens[RUNNER_CALL_COUNT];
    uint32_t positions[RUNNER_CALL_COUNT];
    uint32_t call_count;
} prompt_runner_t;

typedef struct {
    uint8_t *data;
    size_t capacity;
    size_t length;
} byte_collector_t;

static int expect(int condition, const char *message)
{
    if (condition != 0) return 0;
    printf("FAIL: %s\n", message);
    return 1;
}

static int read_file(const char *path, uint8_t **data, size_t *size)
{
    FILE *stream;
    long length;
    uint8_t *buffer;

    *data = NULL;
    *size = 0u;
    stream = fopen(path, "rb");
    if (stream == NULL) return 1;
    if (fseek(stream, 0L, SEEK_END) != 0) {
        (void)fclose(stream);
        return 1;
    }
    length = ftell(stream);
    if (length < 0L || fseek(stream, 0L, SEEK_SET) != 0) {
        (void)fclose(stream);
        return 1;
    }
    buffer = (uint8_t *)malloc(length == 0L ? 1u : (size_t)length);
    if (buffer == NULL) {
        (void)fclose(stream);
        return 1;
    }
    if (length != 0L &&
        fread(buffer, 1u, (size_t)length, stream) != (size_t)length) {
        free(buffer);
        (void)fclose(stream);
        return 1;
    }
    if (fclose(stream) != 0) {
        free(buffer);
        return 1;
    }
    *data = buffer;
    *size = (size_t)length;
    return 0;
}

static int collect_bytes(void *context, const uint8_t *bytes, size_t length)
{
    byte_collector_t *collector = (byte_collector_t *)context;

    if (collector->length > collector->capacity ||
        length > collector->capacity - collector->length) {
        return 1;
    }
    memcpy(collector->data + collector->length, bytes, length);
    collector->length += length;
    return 0;
}

static int prompt_runner(
    void *runner_context,
    uintptr_t base,
    const qot_run_config_t *config,
    uint32_t input_token_id,
    uint32_t position,
    uint32_t max_polls,
    qot_result_t *result)
{
    static const uint32_t outputs[RUNNER_CALL_COUNT] = {
        10u, 20u, 30u, 40u, 264u, 26291u
    };
    prompt_runner_t *runner = (prompt_runner_t *)runner_context;
    uint32_t call = runner->call_count;

    (void)base;
    (void)config;
    (void)max_polls;
    if (call >= RUNNER_CALL_COUNT) return QOT_ERR_CAPACITY;
    runner->input_tokens[call] = input_token_id;
    runner->positions[call] = position;
    memset(result, 0, sizeof(*result));
    result->token_id = outputs[call];
    result->score_q26 = (int64_t)(1000u + call);
    ++runner->call_count;
    return QOT_OK;
}

static qot_run_config_t make_config(void)
{
    qot_run_config_t config;

    memset(&config, 0, sizeof(config));
    config.runtime_context_enable = 1u;
    config.embedding_enable = 1u;
    return config;
}

static int test_prompt_chain(const qtk_asset_t *asset)
{
    static const uint8_t prompt_text[] = "The future of FPGA is";
    static const uint32_t expected_prompt[PROMPT_TOKEN_COUNT] = {
        785u, 3853u, 315u, 89462u, 374u
    };
    static const uint32_t expected_inputs[RUNNER_CALL_COUNT] = {
        785u, 3853u, 315u, 89462u, 374u, 264u
    };
    static const uint32_t expected_generated[GENERATED_TOKEN_COUNT] = {
        264u, 26291u
    };
    static const uint8_t expected_text[] = " a fascinating";
    uint32_t unicode_workspace[TOKEN_WORKSPACE_CAPACITY];
    uint32_t piece_workspace[TOKEN_WORKSPACE_CAPACITY];
    uint32_t prompt_tokens[TOKEN_WORKSPACE_CAPACITY];
    uint32_t generated[GENERATED_TOKEN_COUNT];
    qtk_tokenizer_requirements_t requirements;
    size_t prompt_count = 0u;
    qtk_status_t qtk_status;
    prompt_runner_t runner;
    qot_run_config_t config = make_config();
    qot_session_options_t options;
    qot_session_event_t event;
    qot_session_t session;
    uint32_t generated_count = 0u;
    uint32_t step;
    int rc;
    uint8_t decoded[sizeof(expected_text)];
    byte_collector_t collector;
    uint32_t model_only_token = 151935u;

    qtk_status = qtk_tokenize_utf8(
        asset, prompt_text, sizeof(prompt_text) - 1u,
        unicode_workspace, TOKEN_WORKSPACE_CAPACITY,
        piece_workspace, TOKEN_WORKSPACE_CAPACITY,
        prompt_tokens, TOKEN_WORKSPACE_CAPACITY,
        &prompt_count, &requirements);
    if (expect(qtk_status == QTK_OK, "tokenize prompt")) return 1;
    if (expect(prompt_count == PROMPT_TOKEN_COUNT,
               "prompt token count")) return 1;
    if (expect(memcmp(prompt_tokens, expected_prompt,
                      sizeof(expected_prompt)) == 0,
               "exact prompt token IDs")) return 1;

    memset(&runner, 0, sizeof(runner));
    rc = qot_session_init(&session, (uintptr_t)0u, &config,
                          prompt_runner, &runner);
    if (expect(rc == QOT_OK, "session init")) return 1;
    qot_session_default_options(&options);
    options.max_new_tokens = GENERATED_TOKEN_COUNT;
    rc = qot_session_begin(&session, prompt_tokens, (uint32_t)prompt_count,
                           &options);
    if (expect(rc == QOT_OK, "session begin")) return 1;

    for (step = 0u; step < RUNNER_CALL_COUNT; ++step) {
        rc = qot_session_step(&session, &event);
        if (expect(rc == QOT_OK, "session step")) return 1;
        if (step < PROMPT_TOKEN_COUNT - 1u) {
            if (expect(event.kind == QOT_SESSION_EVENT_PREFILL,
                       "prefill event")) return 1;
        } else {
            if (expect(event.kind == QOT_SESSION_EVENT_TOKEN,
                       "generated token event")) return 1;
            if (expect(generated_count < GENERATED_TOKEN_COUNT,
                       "generated token capacity")) return 1;
            generated[generated_count] = event.result.token_id;
            ++generated_count;
        }
    }
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_DONE &&
               event.stop_reason == QOT_SESSION_STOP_MAX_NEW,
               "terminal MAX_NEW event")) return 1;
    if (expect(runner.call_count == RUNNER_CALL_COUNT,
               "exact runner call count")) return 1;
    for (step = 0u; step < RUNNER_CALL_COUNT; ++step) {
        if (expect(runner.input_tokens[step] == expected_inputs[step],
                   "runner input sequence including dynamic feedback")) {
            return 1;
        }
        if (expect(runner.positions[step] == step,
                   "runner positions 0 through 5")) return 1;
    }
    if (expect(generated_count == GENERATED_TOKEN_COUNT &&
               memcmp(generated, expected_generated,
                      sizeof(expected_generated)) == 0,
               "generated token IDs")) return 1;

    collector.data = decoded;
    collector.capacity = sizeof(decoded);
    collector.length = 0u;
    qtk_status = qtk_detokenize_ids(
        asset, generated, GENERATED_TOKEN_COUNT, 0,
        collect_bytes, &collector);
    if (expect(qtk_status == QTK_OK, "detokenize generated IDs")) return 1;
    if (expect(collector.length == sizeof(expected_text) - 1u &&
               memcmp(decoded, expected_text, sizeof(expected_text) - 1u) == 0,
               "exact generated raw text")) return 1;

    collector.length = 0u;
    if (expect(model_only_token >= asset->token_count &&
               model_only_token < asset->model_vocab_size,
               "chosen ID is model-only")) return 1;
    qtk_status = qtk_detokenize_ids(
        asset, &model_only_token, 1u, 0, collect_bytes, &collector);
    if (expect(qtk_status == QTK_ERR_MODEL_ONLY_TOKEN &&
               collector.length == 0u,
               "model-only ID detokenization error")) return 1;

    return 0;
}

int main(int argc, char **argv)
{
    uint8_t *asset_data = NULL;
    size_t asset_size = 0u;
    qtk_asset_t asset;
    qtk_status_t status;
    int result = 1;

    if (argc != 2) {
        printf("usage: %s <qwen3_tokenizer.qtk>\n", argv[0]);
        return 2;
    }
    if (read_file(argv[1], &asset_data, &asset_size) != 0) {
        printf("FAIL: cannot read tokenizer asset: %s\n", argv[1]);
        return 1;
    }
    status = qtk_asset_init(&asset, asset_data, asset_size);
    if (status != QTK_OK) {
        printf("FAIL: tokenizer asset parse: %s\n", qtk_status_name(status));
        goto cleanup;
    }
    if (test_prompt_chain(&asset) != 0) goto cleanup;

    printf("PASS: PS prompt-to-generated-text host chain\n");
    printf("  prompt_ids=785,3853,315,89462,374\n");
    printf("  runner_inputs=785,3853,315,89462,374,264\n");
    printf("  generated_ids=264,26291\n");
    printf("  generated_text= a fascinating\n");
    printf("  model_only_detokenize=QTK_ERR_MODEL_ONLY_TOKEN\n");
    result = 0;

cleanup:
    free(asset_data);
    return result;
}
