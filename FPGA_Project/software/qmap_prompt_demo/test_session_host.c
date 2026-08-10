#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "qot_session.h"

#define MOCK_MAX_CALLS 16u

typedef struct mock_runner {
    uint32_t outputs[MOCK_MAX_CALLS];
    int return_codes[MOCK_MAX_CALLS];
    uint32_t input_tokens[MOCK_MAX_CALLS];
    uint32_t positions[MOCK_MAX_CALLS];
    uint32_t call_count;
} mock_runner_t;

static int mock_run_token(
    void *runner_context,
    uintptr_t base,
    const qot_run_config_t *config,
    uint32_t input_token_id,
    uint32_t position,
    uint32_t max_polls,
    qot_result_t *result)
{
    mock_runner_t *mock = (mock_runner_t *)runner_context;
    uint32_t call = mock->call_count;

    (void)base;
    (void)config;
    (void)max_polls;
    if (call >= MOCK_MAX_CALLS) return QOT_ERR_CAPACITY;
    mock->input_tokens[call] = input_token_id;
    mock->positions[call] = position;
    memset(result, 0, sizeof(*result));
    result->token_id = mock->outputs[call];
    result->score_q26 = (int64_t)(1000u + call);
    ++mock->call_count;
    return mock->return_codes[call];
}

static qot_run_config_t make_config(uint32_t position)
{
    qot_run_config_t config;
    memset(&config, 0, sizeof(config));
    config.position = position;
    config.runtime_context_enable = 1u;
    config.embedding_enable = 1u;
    return config;
}

static int expect(int condition, const char *message)
{
    if (condition) return 0;
    printf("FAIL: %s\n", message);
    return 1;
}

static int test_prefill_and_real_feedback(void)
{
    static const uint32_t prompt[] = {10u, 11u};
    static const uint32_t expected_inputs[] = {10u, 11u, 20u, 21u};
    static const uint32_t expected_positions[] = {0u, 1u, 2u, 3u};
    mock_runner_t mock;
    qot_run_config_t config = make_config(0u);
    qot_session_options_t options;
    qot_session_event_t event;
    qot_session_t session;
    uint32_t index;
    int rc;

    memset(&mock, 0, sizeof(mock));
    mock.outputs[0] = 99u; /* Prediction while prompt prefill continues. */
    mock.outputs[1] = 20u; /* First emitted token. */
    mock.outputs[2] = 21u;
    mock.outputs[3] = 22u;

    rc = qot_session_init(&session, 0xA0040000u, &config,
                          mock_run_token, &mock);
    if (expect(rc == QOT_OK, "session init")) return 1;
    qot_session_default_options(&options);
    options.max_new_tokens = 3u;
    options.max_polls_per_token = 123u;
    rc = qot_session_begin(&session, prompt, 2u, &options);
    if (expect(rc == QOT_OK, "session begin")) return 1;

    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_PREFILL &&
               event.prompt_tokens_consumed == 1u,
               "first prompt token emits PREFILL")) return 1;

    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_TOKEN &&
               event.generated_index == 0u && event.result.token_id == 20u &&
               session.state == QOT_SESSION_STATE_DECODE,
               "last prompt result is first generated token")) return 1;

    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_TOKEN &&
               event.generated_index == 1u && event.result.token_id == 21u,
               "first argmax is fed back")) return 1;

    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_TOKEN &&
               event.generated_index == 2u && event.result.token_id == 22u &&
               event.stop_reason == QOT_SESSION_STOP_MAX_NEW &&
               session.state == QOT_SESSION_STATE_STOPPING,
               "max_new schedules a clean stop after final token")) return 1;

    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_DONE &&
               event.stop_reason == QOT_SESSION_STOP_MAX_NEW &&
               qot_session_is_terminal(&session),
               "max_new emits DONE")) return 1;

    if (expect(mock.call_count == 4u, "one runner call per hardware token")) {
        return 1;
    }
    for (index = 0u; index < mock.call_count; ++index) {
        if (expect(mock.input_tokens[index] == expected_inputs[index] &&
                   mock.positions[index] == expected_positions[index],
                   "runner input/position sequence")) return 1;
    }

    qot_session_reset(&session);
    return expect(session.state == QOT_SESSION_STATE_IDLE,
                  "reset returns initialized session to IDLE");
}

static int test_eos_and_im_end(void)
{
    static const uint32_t prompt[] = {7u};
    mock_runner_t mock;
    qot_run_config_t config = make_config(0u);
    qot_session_options_t options;
    qot_session_event_t event;
    qot_session_t session;
    int rc;

    memset(&mock, 0, sizeof(mock));
    mock.outputs[0] = QOT_SESSION_DEFAULT_EOS_TOKEN_ID;
    rc = qot_session_init(&session, 0u, &config, mock_run_token, &mock);
    if (expect(rc == QOT_OK, "EOS init")) return 1;
    qot_session_default_options(&options);
    options.max_new_tokens = 5u;
    rc = qot_session_begin(&session, prompt, 1u, &options);
    if (expect(rc == QOT_OK, "EOS begin")) return 1;
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_TOKEN &&
               event.stop_reason == QOT_SESSION_STOP_EOS,
               "EOS stops on first generated token")) return 1;
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_DONE &&
               event.stop_reason == QOT_SESSION_STOP_EOS &&
               mock.call_count == 1u,
               "EOS DONE does not invoke runner")) return 1;

    memset(&mock, 0, sizeof(mock));
    mock.outputs[0] = 8u;
    mock.outputs[1] = QOT_SESSION_DEFAULT_IM_END_TOKEN_ID;
    rc = qot_session_init(&session, 0u, &config, mock_run_token, &mock);
    if (expect(rc == QOT_OK, "IM_END init")) return 1;
    rc = qot_session_begin(&session, prompt, 1u, &options);
    if (expect(rc == QOT_OK, "IM_END begin")) return 1;
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.result.token_id == 8u &&
               event.stop_reason == QOT_SESSION_STOP_NONE,
               "ordinary token continues decode")) return 1;
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.result.token_id ==
               QOT_SESSION_DEFAULT_IM_END_TOKEN_ID &&
               event.stop_reason == QOT_SESSION_STOP_IM_END &&
               mock.input_tokens[1] == 8u,
               "IM_END stops after actual feedback")) return 1;
    return 0;
}

static int test_context_stop(void)
{
    static const uint32_t prompt[] = {1u, 2u};
    mock_runner_t mock;
    qot_run_config_t config = make_config(QOT_MAX_CONTEXT - 2u);
    qot_session_options_t options;
    qot_session_event_t event;
    qot_session_t session;
    int rc;

    memset(&mock, 0, sizeof(mock));
    mock.outputs[0] = 30u;
    mock.outputs[1] = 31u;
    rc = qot_session_init(&session, 0u, &config, mock_run_token, &mock);
    if (expect(rc == QOT_OK, "context init")) return 1;
    qot_session_default_options(&options);
    options.max_new_tokens = 8u;
    rc = qot_session_begin(&session, prompt, 2u, &options);
    if (expect(rc == QOT_OK, "prompt that exactly fills context is valid")) {
        return 1;
    }
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_PREFILL,
               "context prefill first token")) return 1;
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_TOKEN &&
               event.result.token_id == 31u &&
               event.stop_reason == QOT_SESSION_STOP_CONTEXT,
               "last legal position still produces one output token")) return 1;
    rc = qot_session_step(&session, &event);
    return expect(rc == QOT_OK && event.kind == QOT_SESSION_EVENT_DONE &&
                  event.stop_reason == QOT_SESSION_STOP_CONTEXT &&
                  mock.call_count == 2u,
                  "context stop performs no out-of-range runner call");
}

static int test_hardware_errors(void)
{
    static const uint32_t prompt[] = {5u};
    mock_runner_t mock;
    qot_run_config_t config = make_config(0u);
    qot_session_options_t options;
    qot_session_event_t event;
    qot_session_t session;
    int rc;

    memset(&mock, 0, sizeof(mock));
    mock.outputs[0] = 6u;
    mock.return_codes[1] = QOT_ERR_TIMEOUT;
    rc = qot_session_init(&session, 0u, &config, mock_run_token, &mock);
    if (expect(rc == QOT_OK, "hardware error init")) return 1;
    qot_session_default_options(&options);
    options.max_new_tokens = 3u;
    rc = qot_session_begin(&session, prompt, 1u, &options);
    if (expect(rc == QOT_OK, "hardware error begin")) return 1;
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_OK && event.result.token_id == 6u,
               "hardware error first token")) return 1;
    rc = qot_session_step(&session, &event);
    if (expect(rc == QOT_ERR_TIMEOUT &&
               event.kind == QOT_SESSION_EVENT_ERROR &&
               event.stop_reason == QOT_SESSION_STOP_HW_ERROR &&
               event.error_code == QOT_ERR_TIMEOUT &&
               session.state == QOT_SESSION_STATE_ERROR &&
               mock.input_tokens[1] == 6u,
               "runner error becomes terminal HW_ERROR")) return 1;

    memset(&mock, 0, sizeof(mock));
    mock.outputs[0] = QOT_VOCAB_SIZE;
    rc = qot_session_init(&session, 0u, &config, mock_run_token, &mock);
    if (expect(rc == QOT_OK, "bad output init")) return 1;
    rc = qot_session_begin(&session, prompt, 1u, &options);
    if (expect(rc == QOT_OK, "bad output begin")) return 1;
    rc = qot_session_step(&session, &event);
    return expect(rc == QOT_ERR_BAD_TOKEN &&
                  event.kind == QOT_SESSION_EVENT_ERROR &&
                  event.stop_reason == QOT_SESSION_STOP_HW_ERROR,
                  "out-of-vocabulary hardware output is rejected");
}

static int test_begin_validation(void)
{
    static const uint32_t valid_prompt[] = {1u, 2u};
    static const uint32_t invalid_prompt[] = {1u, QOT_VOCAB_SIZE};
    mock_runner_t mock;
    qot_run_config_t config = make_config(QOT_MAX_CONTEXT - 1u);
    qot_session_options_t options;
    qot_session_t session;
    int rc;

    memset(&mock, 0, sizeof(mock));
    rc = qot_session_init(&session, 0u, &config, mock_run_token, &mock);
    if (expect(rc == QOT_OK, "validation init")) return 1;
    qot_session_default_options(&options);
    rc = qot_session_begin(&session, invalid_prompt, 2u, &options);
    if (expect(rc == QOT_ERR_BAD_TOKEN && mock.call_count == 0u,
               "invalid prompt is prevalidated")) return 1;
    rc = qot_session_begin(&session, valid_prompt, 2u, &options);
    if (expect(rc == QOT_ERR_CONTEXT && mock.call_count == 0u,
               "prompt overflow is prevalidated")) return 1;
    options.max_new_tokens = 0u;
    rc = qot_session_begin(&session, valid_prompt, 1u, &options);
    return expect(rc == QOT_ERR_CAPACITY && mock.call_count == 0u,
                  "zero max_new is rejected");
}

int main(void)
{
    if (test_prefill_and_real_feedback() != 0) return 1;
    if (test_eos_and_im_end() != 0) return 1;
    if (test_context_stop() != 0) return 1;
    if (test_hardware_errors() != 0) return 1;
    if (test_begin_validation() != 0) return 1;
    printf("PASS: qot_session host tests\n");
    return 0;
}
