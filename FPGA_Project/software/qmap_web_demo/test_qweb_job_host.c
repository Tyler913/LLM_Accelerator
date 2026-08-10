#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qweb_job.h"

#define ARRAY_SIZE(values) (sizeof(values) / sizeof((values)[0]))

typedef struct mock_runner {
    const uint32_t *expected_inputs;
    const uint32_t *outputs;
    const int64_t *scores;
    size_t count;
    size_t index;
    int fail_at;
    int mismatch;
} mock_runner_t;

static qweb_job_t test_job;

static int check(int condition, const char *message)
{
    if (condition) return 0;
    fprintf(stderr, "FAIL %s\n", message);
    return 1;
}

static int mock_run_token(
    void *runner_context,
    uintptr_t base,
    const qot_run_config_t *config,
    uint32_t input_token_id,
    uint32_t position,
    uint32_t max_polls,
    qot_result_t *result)
{
    mock_runner_t *runner = (mock_runner_t *)runner_context;

    (void)base;
    (void)config;
    (void)max_polls;
    if (runner == NULL || result == NULL || runner->index >= runner->count) {
        return QOT_ERR_STATUS;
    }
    if (input_token_id != runner->expected_inputs[runner->index] ||
        position != (uint32_t)runner->index) {
        runner->mismatch = 1;
        return QOT_ERR_STATUS;
    }
    if (runner->fail_at >= 0 &&
        runner->index == (size_t)runner->fail_at) {
        ++runner->index;
        return QOT_ERR_STATUS;
    }
    memset(result, 0, sizeof(*result));
    result->token_id = runner->outputs[runner->index];
    result->score_q26 = runner->scores[runner->index];
    result->layers_started = QOT_MAX_LAYERS;
    result->layers_completed = QOT_MAX_LAYERS;
    result->layer_done_mask = UINT32_C(0x0fffffff);
    ++runner->index;
    return QOT_OK;
}

static int load_tokenizer(
    const char *path,
    uint8_t **storage,
    qtk_asset_t *asset)
{
    FILE *stream;
    long length;
    size_t read_count;

    stream = fopen(path, "rb");
    if (stream == NULL) return 1;
    if (fseek(stream, 0L, SEEK_END) != 0) {
        fclose(stream);
        return 1;
    }
    length = ftell(stream);
    if (length <= 0L || fseek(stream, 0L, SEEK_SET) != 0) {
        fclose(stream);
        return 1;
    }
    *storage = (uint8_t *)malloc((size_t)length);
    if (*storage == NULL) {
        fclose(stream);
        return 1;
    }
    read_count = fread(*storage, 1u, (size_t)length, stream);
    fclose(stream);
    if (read_count != (size_t)length ||
        qtk_asset_init(asset, *storage, (size_t)length) != QTK_OK) {
        free(*storage);
        *storage = NULL;
        return 1;
    }
    return 0;
}

static qot_run_config_t test_config(void)
{
    qot_run_config_t config;

    memset(&config, 0, sizeof(config));
    config.runtime_context_enable = 1u;
    config.embedding_enable = 1u;
    return config;
}

static int run_to_terminal(qweb_job_t *job)
{
    size_t guard = 0u;

    while (qweb_job_is_active(job)) {
        if (qweb_job_step(job, NULL) != QWEB_JOB_OK) return 1;
        ++guard;
        if (guard > (size_t)(QOT_MAX_CONTEXT * 2u + 2u)) return 1;
    }
    return 0;
}

static int test_token_feedback(const qtk_asset_t *tokenizer)
{
    static const uint32_t expected_inputs[] = {374u, 28458u};
    static const uint32_t outputs[] = {28458u, 64u};
    static const int64_t scores[] = {
        INT64_C(1227344433), INT64_C(1015661901)
    };
    qweb_generate_request_t request;
    qot_run_config_t config = test_config();
    mock_runner_t runner = {
        expected_inputs, outputs, scores, ARRAY_SIZE(outputs), 0u, -1, 0
    };
    uint32_t job_id = 0u;
    int failed = 0;

    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_TOKENS;
    request.max_new_tokens = 2u;
    request.token_count = 1u;
    request.token_ids[0] = 374u;
    failed |= check(qweb_job_init(&test_job, 0u, &config, tokenizer,
                                  mock_run_token, &runner, 123u) == QWEB_JOB_OK,
                    "token feedback init");
    failed |= check(qweb_job_submit(&test_job, &request, &job_id) ==
                        QWEB_JOB_OK && job_id == 1u,
                    "token feedback submit");
    failed |= check(qweb_job_submit(&test_job, &request, NULL) ==
                        QWEB_JOB_ERR_BUSY,
                    "busy request rejected");
    failed |= check(run_to_terminal(&test_job) == 0,
                    "token feedback reaches terminal");
    failed |= check(test_job.state == QWEB_JOB_STATE_DONE,
                    "token feedback done state");
    failed |= check(test_job.generated_count == 2u &&
                        test_job.generated_token_ids[0] == 28458u &&
                        test_job.generated_token_ids[1] == 64u,
                    "actual result feedback sequence");
    failed |= check(test_job.generated_scores_q26[0] == scores[0] &&
                        test_job.generated_scores_q26[1] == scores[1],
                    "token feedback scores");
    failed |= check(test_job.stop_reason == QOT_SESSION_STOP_MAX_NEW,
                    "token feedback stop reason");
    failed |= check(test_job.output_length == 5u &&
                        memcmp(test_job.output_bytes, "aaaaa", 5u) == 0,
                    "token feedback detokenized bytes");
    failed |= check(runner.index == runner.count && runner.mismatch == 0,
                    "token feedback runner inputs");
    return failed;
}

static int test_text_prompt(const qtk_asset_t *tokenizer)
{
    static const uint8_t prompt[] = "The future of FPGA is";
    static const uint32_t expected_inputs[] = {
        785u, 3853u, 315u, 89462u, 374u, 264u
    };
    static const uint32_t outputs[] = {1u, 2u, 3u, 4u, 264u, 26291u};
    static const int64_t scores[] = {
        0, 0, 0, 0, INT64_C(1296911292), INT64_C(1225544557)
    };
    qweb_generate_request_t request;
    qot_run_config_t config = test_config();
    mock_runner_t runner = {
        expected_inputs, outputs, scores, ARRAY_SIZE(outputs), 0u, -1, 0
    };
    int failed = 0;

    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_PROMPT;
    request.max_new_tokens = 2u;
    request.prompt_length = sizeof(prompt) - 1u;
    memcpy(request.prompt, prompt, request.prompt_length);
    failed |= check(qweb_job_init(&test_job, 0u, &config, tokenizer,
                                  mock_run_token, &runner, 0u) == QWEB_JOB_OK,
                    "text prompt init");
    failed |= check(qweb_job_submit(&test_job, &request, NULL) == QWEB_JOB_OK,
                    "text prompt submit");
    failed |= check(test_job.prompt_token_count == 5u &&
                        memcmp(test_job.prompt_token_ids,
                               expected_inputs,
                               5u * sizeof(expected_inputs[0])) == 0,
                    "text prompt exact tokenization");
    failed |= check(run_to_terminal(&test_job) == 0,
                    "text prompt reaches terminal");
    failed |= check(test_job.generated_count == 2u &&
                        test_job.generated_token_ids[0] == 264u &&
                        test_job.generated_token_ids[1] == 26291u,
                    "text prompt generated IDs");
    failed |= check(test_job.output_length ==
                        sizeof(" a fascinating") - 1u &&
                        memcmp(test_job.output_bytes,
                               " a fascinating",
                               sizeof(" a fascinating") - 1u) == 0,
                    "text prompt output bytes");
    failed |= check(runner.index == runner.count && runner.mismatch == 0,
                    "text prompt runner sequence");
    return failed;
}

static int test_errors(const qtk_asset_t *tokenizer)
{
    static const uint32_t expected_inputs[] = {374u};
    static const uint32_t outputs[] = {28458u};
    static const int64_t scores[] = {INT64_C(1227344433)};
    qweb_generate_request_t request;
    qot_run_config_t config = test_config();
    mock_runner_t runner = {
        expected_inputs, outputs, scores, ARRAY_SIZE(outputs), 0u, 0, 0
    };
    int failed = 0;

    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_TOKENS;
    request.max_new_tokens = 1u;
    request.token_count = 1u;
    request.token_ids[0] = QOT_VOCAB_SIZE;
    failed |= check(qweb_job_init(&test_job, 0u, &config, tokenizer,
                                  mock_run_token, &runner, 0u) == QWEB_JOB_OK,
                    "error test init");
    failed |= check(qweb_job_init(&test_job, 0u, &config, tokenizer,
                                  NULL, &runner, 0u) == QWEB_JOB_ERR_NULL,
                    "Web job rejects a non-pumping default runner");
    failed |= check(qweb_job_submit(&test_job, &request, NULL) ==
                        QWEB_JOB_ERR_REQUEST,
                    "out-of-range token rejected before runner");
    request.token_ids[0] = 374u;
    failed |= check(qweb_job_submit(&test_job, &request, NULL) == QWEB_JOB_OK,
                    "hardware error request accepted");
    failed |= check(qweb_job_step(&test_job, NULL) == QWEB_JOB_ERR_SESSION,
                    "hardware error propagated");
    failed |= check(test_job.state == QWEB_JOB_STATE_ERROR &&
                        test_job.session_error_code == QOT_ERR_STATUS,
                    "hardware error state recorded");

    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_PROMPT;
    request.max_new_tokens = 1u;
    request.prompt_length = 1u;
    request.prompt[0] = UINT8_C(0xff);
    failed |= check(qweb_job_submit(&test_job, &request, NULL) ==
                        QWEB_JOB_ERR_TOKENIZE,
                    "invalid UTF-8 rejected by tokenizer");
    failed |= check(test_job.state == QWEB_JOB_STATE_ERROR &&
                        test_job.error_code == QWEB_JOB_ERR_SESSION &&
                        test_job.session_error_code == QOT_ERR_STATUS &&
                        test_job.tokenizer_status == QTK_OK,
                    "rejected request does not corrupt prior job record");

    runner.index = 0u;
    runner.fail_at = -1;
    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_TOKENS;
    request.max_new_tokens = 1u;
    request.token_count = 1u;
    request.token_ids[0] = 374u;
    failed |= check(qweb_job_submit(&test_job, &request, NULL) == QWEB_JOB_OK,
                    "fresh request accepted after prior job error");
    failed |= check(run_to_terminal(&test_job) == 0 &&
                        test_job.state == QWEB_JOB_STATE_DONE &&
                        test_job.generated_token_ids[0] == 28458u,
                    "fresh request completes after prior job error");
    return failed;
}

static int test_eos_stop(const qtk_asset_t *tokenizer)
{
    static const uint32_t expected_inputs[] = {374u};
    static const uint32_t outputs[] = {QOT_SESSION_DEFAULT_EOS_TOKEN_ID};
    static const int64_t scores[] = {0};
    qweb_generate_request_t request;
    qot_run_config_t config = test_config();
    mock_runner_t runner = {
        expected_inputs, outputs, scores, ARRAY_SIZE(outputs), 0u, -1, 0
    };
    int failed = 0;

    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_TOKENS;
    request.max_new_tokens = 2u;
    request.token_count = 1u;
    request.token_ids[0] = 374u;
    failed |= check(qweb_job_init(&test_job, 0u, &config, tokenizer,
                                  mock_run_token, &runner, 0u) == QWEB_JOB_OK,
                    "EOS test init");
    failed |= check(qweb_job_submit(&test_job, &request, NULL) == QWEB_JOB_OK,
                    "EOS request accepted");
    failed |= check(run_to_terminal(&test_job) == 0,
                    "EOS request reaches terminal");
    failed |= check(test_job.state == QWEB_JOB_STATE_DONE &&
                        test_job.stop_reason == QOT_SESSION_STOP_EOS &&
                        test_job.generated_count == 1u &&
                        test_job.generated_token_ids[0] ==
                            QOT_SESSION_DEFAULT_EOS_TOKEN_ID,
                    "EOS result and stop reason retained");
    failed |= check(test_job.output_length == 0u,
                    "EOS special token omitted from output bytes");
    failed |= check(runner.index == runner.count && runner.mismatch == 0,
                    "EOS runner stopped without a second model call");
    return failed;
}

static int test_split_utf8_output(const qtk_asset_t *tokenizer)
{
    static const uint32_t expected_inputs[] = {374u, 160u, 121u};
    static const uint32_t outputs[] = {160u, 121u, 254u};
    static const int64_t scores[] = {0, 0, 0};
    static const uint8_t expected_utf8[] = {0xe4u, 0xbdu, 0xa0u};
    qweb_generate_request_t request;
    qot_run_config_t config = test_config();
    mock_runner_t runner = {
        expected_inputs, outputs, scores, ARRAY_SIZE(outputs), 0u, -1, 0
    };
    int failed = 0;

    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_TOKENS;
    request.max_new_tokens = 3u;
    request.token_count = 1u;
    request.token_ids[0] = 374u;
    failed |= check(qweb_job_init(&test_job, 0u, &config, tokenizer,
                                  mock_run_token, &runner, 0u) == QWEB_JOB_OK,
                    "split UTF-8 test init");
    failed |= check(qweb_job_submit(&test_job, &request, NULL) == QWEB_JOB_OK,
                    "split UTF-8 request accepted");
    failed |= check(run_to_terminal(&test_job) == 0,
                    "split UTF-8 request reaches terminal");
    failed |= check(test_job.output_length == sizeof(expected_utf8) &&
                        memcmp(test_job.output_bytes,
                               expected_utf8,
                               sizeof(expected_utf8)) == 0,
                    "raw byte pieces join into one UTF-8 scalar");
    failed |= check(runner.index == runner.count && runner.mismatch == 0,
                    "split UTF-8 feedback sequence");
    return failed;
}

static int test_model_only_output_error(const qtk_asset_t *tokenizer)
{
    static const uint32_t expected_inputs[] = {374u};
    static const uint32_t outputs[] = {QOT_VOCAB_SIZE - 1u};
    static const int64_t scores[] = {0};
    qweb_generate_request_t request;
    qot_run_config_t config = test_config();
    mock_runner_t runner = {
        expected_inputs, outputs, scores, ARRAY_SIZE(outputs), 0u, -1, 0
    };
    int failed = 0;

    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_TOKENS;
    request.max_new_tokens = 1u;
    request.token_count = 1u;
    request.token_ids[0] = 374u;
    failed |= check(qweb_job_init(&test_job, 0u, &config, tokenizer,
                                  mock_run_token, &runner, 0u) == QWEB_JOB_OK,
                    "model-only output test init");
    failed |= check(qweb_job_submit(&test_job, &request, NULL) == QWEB_JOB_OK,
                    "model-only output request accepted");
    failed |= check(qweb_job_step(&test_job, NULL) ==
                        QWEB_JOB_ERR_DETOKENIZE,
                    "model-only output fails explicit detokenization");
    failed |= check(test_job.state == QWEB_JOB_STATE_ERROR &&
                        test_job.generated_count == 1u &&
                        test_job.generated_token_ids[0] == QOT_VOCAB_SIZE - 1u &&
                        test_job.tokenizer_status ==
                            QTK_ERR_MODEL_ONLY_TOKEN &&
                        test_job.stop_reason == QOT_SESSION_STOP_NONE,
                    "model-only output is not mislabeled as hardware error");
    return failed;
}

static int test_context_rejection_preserves_record(
    const qtk_asset_t *tokenizer)
{
    static const uint32_t expected_inputs[] = {374u};
    static const uint32_t outputs[] = {28458u};
    static const int64_t scores[] = {0};
    qweb_generate_request_t request;
    qot_run_config_t config = test_config();
    mock_runner_t runner = {
        expected_inputs, outputs, scores, ARRAY_SIZE(outputs), 0u, -1, 0
    };
    int failed = 0;

    config.position = QOT_MAX_CONTEXT - 1u;
    failed |= check(qweb_job_init(&test_job, 0u, &config, tokenizer,
                                  mock_run_token, &runner, 0u) == QWEB_JOB_OK,
                    "context rejection test init");
    test_job.state = QWEB_JOB_STATE_DONE;
    test_job.job_id = 7u;
    test_job.generated_count = 1u;
    test_job.generated_token_ids[0] = 64u;

    memset(&request, 0, sizeof(request));
    request.kind = QWEB_GENERATE_INPUT_TOKENS;
    request.max_new_tokens = 1u;
    request.token_count = 2u;
    request.token_ids[0] = 374u;
    request.token_ids[1] = 28458u;
    failed |= check(qweb_job_submit(&test_job, &request, NULL) ==
                        QWEB_JOB_ERR_REQUEST,
                    "prompt beyond remaining context rejected before begin");
    failed |= check(test_job.state == QWEB_JOB_STATE_DONE &&
                        test_job.job_id == 7u &&
                        test_job.generated_count == 1u &&
                        test_job.generated_token_ids[0] == 64u,
                    "context rejection preserves previous job record");
    failed |= check(runner.index == 0u,
                    "context rejection issues no model command");
    return failed;
}

int main(int argc, char **argv)
{
    qtk_asset_t tokenizer;
    uint8_t *storage = NULL;
    int failed = 0;

    if (argc != 2) {
        fprintf(stderr, "usage: %s qwen3_tokenizer.qtk\n", argv[0]);
        return 2;
    }
    if (load_tokenizer(argv[1], &storage, &tokenizer) != 0) {
        fprintf(stderr, "FAIL tokenizer load: %s\n", argv[1]);
        return 2;
    }

    failed |= test_token_feedback(&tokenizer);
    failed |= test_text_prompt(&tokenizer);
    failed |= test_errors(&tokenizer);
    failed |= test_eos_stop(&tokenizer);
    failed |= test_split_utf8_output(&tokenizer);
    failed |= test_model_only_output_error(&tokenizer);
    failed |= test_context_rejection_preserves_record(&tokenizer);
    free(storage);
    if (failed != 0) return 1;
    printf("PASS qweb job host tests\n");
    return 0;
}
