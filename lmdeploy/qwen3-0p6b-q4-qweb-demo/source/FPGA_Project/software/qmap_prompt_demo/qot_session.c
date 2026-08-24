#include "qot_session.h"

#include <string.h>

static int qot_session_default_runner(
    void *runner_context,
    uintptr_t base,
    const qot_run_config_t *config,
    uint32_t input_token_id,
    uint32_t position,
    uint32_t max_polls,
    qot_result_t *result)
{
    (void)runner_context;
    return qot_run_token(base, config, input_token_id, position, max_polls,
                         result);
}

static void qot_session_clear_event(qot_session_event_t *event)
{
    memset(event, 0, sizeof(*event));
    event->kind = QOT_SESSION_EVENT_NONE;
    event->stop_reason = QOT_SESSION_STOP_NONE;
    event->error_code = QOT_OK;
}

static int qot_session_fail(
    qot_session_t *session,
    qot_session_event_t *event,
    int error_code)
{
    session->last_error = error_code;
    session->stop_reason = QOT_SESSION_STOP_HW_ERROR;
    session->state = QOT_SESSION_STATE_ERROR;
    event->kind = QOT_SESSION_EVENT_ERROR;
    event->stop_reason = session->stop_reason;
    event->error_code = error_code;
    event->result = session->last_result;
    return error_code;
}

static qot_session_stop_reason_t qot_session_token_stop_reason(
    const qot_session_t *session,
    uint32_t token_id)
{
    if (session->options.stop_on_eos != 0u &&
        token_id == session->options.eos_token_id) {
        return QOT_SESSION_STOP_EOS;
    }
    if (session->options.stop_on_im_end != 0u &&
        token_id == session->options.im_end_token_id) {
        return QOT_SESSION_STOP_IM_END;
    }
    if (session->generated_count >= session->options.max_new_tokens) {
        return QOT_SESSION_STOP_MAX_NEW;
    }
    if (session->next_position >= QOT_MAX_CONTEXT) {
        return QOT_SESSION_STOP_CONTEXT;
    }
    return QOT_SESSION_STOP_NONE;
}

static void qot_session_emit_token(
    qot_session_t *session,
    qot_session_event_t *event,
    uint32_t input_token_id,
    uint32_t input_position)
{
    qot_session_stop_reason_t reason;

    event->kind = QOT_SESSION_EVENT_TOKEN;
    event->input_token_id = input_token_id;
    event->input_position = input_position;
    event->prompt_tokens_consumed = session->prompt_index;
    event->generated_index = session->generated_count - 1u;
    event->result = session->last_result;

    reason = qot_session_token_stop_reason(session,
                                           session->last_result.token_id);
    if (reason != QOT_SESSION_STOP_NONE) {
        session->stop_reason = reason;
        session->state = QOT_SESSION_STATE_STOPPING;
        event->stop_reason = reason;
    } else {
        session->state = QOT_SESSION_STATE_DECODE;
    }
}

void qot_session_default_options(qot_session_options_t *options)
{
    if (options == NULL) return;
    memset(options, 0, sizeof(*options));
    options->max_new_tokens = 1u;
    options->max_polls_per_token = 0u;
    options->eos_token_id = QOT_SESSION_DEFAULT_EOS_TOKEN_ID;
    options->im_end_token_id = QOT_SESSION_DEFAULT_IM_END_TOKEN_ID;
    options->stop_on_eos = 1u;
    options->stop_on_im_end = 1u;
}

int qot_session_init(
    qot_session_t *session,
    uintptr_t base,
    const qot_run_config_t *config,
    qot_session_runner_fn runner,
    void *runner_context)
{
    if (session == NULL || config == NULL) return QOT_ERR_NULL;
    if (config->runtime_context_enable == 0u ||
        config->embedding_enable == 0u ||
        config->final_hidden_override_valid != 0u ||
        config->position >= QOT_MAX_CONTEXT) {
        return QOT_ERR_CONTEXT;
    }

    memset(session, 0, sizeof(*session));
    session->base = base;
    session->config = *config;
    session->runner = (runner != NULL) ? runner : qot_session_default_runner;
    session->runner_context = runner_context;
    session->last_error = QOT_OK;
    session->state = QOT_SESSION_STATE_IDLE;
    session->stop_reason = QOT_SESSION_STOP_NONE;
    return QOT_OK;
}

int qot_session_begin(
    qot_session_t *session,
    const uint32_t *prompt_token_ids,
    uint32_t prompt_token_count,
    const qot_session_options_t *options)
{
    qot_session_options_t selected_options;
    uint32_t prompt_index;
    uint64_t prompt_end;

    if (session == NULL || prompt_token_ids == NULL) return QOT_ERR_NULL;
    if (session->state == QOT_SESSION_STATE_UNINITIALIZED ||
        session->runner == NULL) {
        return QOT_SESSION_ERR_STATE;
    }
    if (prompt_token_count == 0u) return QOT_ERR_CAPACITY;

    qot_session_default_options(&selected_options);
    if (options != NULL) selected_options = *options;
    if (selected_options.max_new_tokens == 0u) return QOT_ERR_CAPACITY;
    if ((selected_options.stop_on_eos != 0u &&
         selected_options.eos_token_id >= QOT_VOCAB_SIZE) ||
        (selected_options.stop_on_im_end != 0u &&
         selected_options.im_end_token_id >= QOT_VOCAB_SIZE)) {
        return QOT_ERR_BAD_TOKEN;
    }

    for (prompt_index = 0u; prompt_index < prompt_token_count; ++prompt_index) {
        if (prompt_token_ids[prompt_index] >= QOT_VOCAB_SIZE) {
            return QOT_ERR_BAD_TOKEN;
        }
    }

    prompt_end = (uint64_t)session->config.position +
                 (uint64_t)prompt_token_count;
    if (prompt_end > (uint64_t)QOT_MAX_CONTEXT) return QOT_ERR_CONTEXT;

    session->prompt_token_ids = prompt_token_ids;
    session->prompt_token_count = prompt_token_count;
    session->prompt_index = 0u;
    session->next_position = session->config.position;
    session->pending_token_id = 0u;
    session->generated_count = 0u;
    session->options = selected_options;
    memset(&session->last_result, 0, sizeof(session->last_result));
    session->last_error = QOT_OK;
    session->stop_reason = QOT_SESSION_STOP_NONE;
    session->state = QOT_SESSION_STATE_PREFILL;
    return QOT_OK;
}

int qot_session_step(qot_session_t *session, qot_session_event_t *event)
{
    uint32_t input_token_id;
    uint32_t input_position;
    int rc;

    if (session == NULL || event == NULL) return QOT_ERR_NULL;
    qot_session_clear_event(event);

    if (session->state == QOT_SESSION_STATE_STOPPING) {
        session->state = QOT_SESSION_STATE_DONE;
        event->kind = QOT_SESSION_EVENT_DONE;
        event->stop_reason = session->stop_reason;
        event->prompt_tokens_consumed = session->prompt_index;
        event->generated_index = session->generated_count;
        event->result = session->last_result;
        return QOT_OK;
    }
    if (session->state == QOT_SESSION_STATE_IDLE ||
        session->state == QOT_SESSION_STATE_UNINITIALIZED ||
        session->state == QOT_SESSION_STATE_DONE ||
        session->state == QOT_SESSION_STATE_ERROR) {
        return QOT_SESSION_ERR_STATE;
    }

    if (session->state == QOT_SESSION_STATE_PREFILL) {
        input_token_id = session->prompt_token_ids[session->prompt_index];
    } else {
        if (session->next_position >= QOT_MAX_CONTEXT) {
            session->stop_reason = QOT_SESSION_STOP_CONTEXT;
            session->state = QOT_SESSION_STATE_DONE;
            event->kind = QOT_SESSION_EVENT_DONE;
            event->stop_reason = session->stop_reason;
            event->prompt_tokens_consumed = session->prompt_index;
            event->generated_index = session->generated_count;
            event->result = session->last_result;
            return QOT_OK;
        }
        input_token_id = session->pending_token_id;
    }
    input_position = session->next_position;

    memset(&session->last_result, 0, sizeof(session->last_result));
    rc = session->runner(session->runner_context,
                         session->base,
                         &session->config,
                         input_token_id,
                         input_position,
                         session->options.max_polls_per_token,
                         &session->last_result);
    if (rc != QOT_OK) {
        event->input_token_id = input_token_id;
        event->input_position = input_position;
        return qot_session_fail(session, event, rc);
    }
    if (session->last_result.token_id >= QOT_VOCAB_SIZE) {
        event->input_token_id = input_token_id;
        event->input_position = input_position;
        return qot_session_fail(session, event, QOT_ERR_BAD_TOKEN);
    }

    ++session->next_position;
    if (session->state == QOT_SESSION_STATE_PREFILL) {
        ++session->prompt_index;
        if (session->prompt_index < session->prompt_token_count) {
            event->kind = QOT_SESSION_EVENT_PREFILL;
            event->input_token_id = input_token_id;
            event->input_position = input_position;
            event->prompt_tokens_consumed = session->prompt_index;
            event->result = session->last_result;
            return QOT_OK;
        }
    }

    session->pending_token_id = session->last_result.token_id;
    ++session->generated_count;
    qot_session_emit_token(session, event, input_token_id, input_position);
    return QOT_OK;
}

void qot_session_reset(qot_session_t *session)
{
    if (session == NULL ||
        session->state == QOT_SESSION_STATE_UNINITIALIZED) {
        return;
    }
    session->prompt_token_ids = NULL;
    session->prompt_token_count = 0u;
    session->prompt_index = 0u;
    session->next_position = session->config.position;
    session->pending_token_id = 0u;
    session->generated_count = 0u;
    memset(&session->last_result, 0, sizeof(session->last_result));
    session->last_error = QOT_OK;
    session->state = QOT_SESSION_STATE_IDLE;
    session->stop_reason = QOT_SESSION_STOP_NONE;
}

int qot_session_is_terminal(const qot_session_t *session)
{
    if (session == NULL) return 1;
    return session->state == QOT_SESSION_STATE_DONE ||
           session->state == QOT_SESSION_STATE_ERROR;
}

const char *qot_session_state_name(qot_session_state_t state)
{
    switch (state) {
    case QOT_SESSION_STATE_UNINITIALIZED: return "UNINITIALIZED";
    case QOT_SESSION_STATE_IDLE: return "IDLE";
    case QOT_SESSION_STATE_PREFILL: return "PREFILL";
    case QOT_SESSION_STATE_DECODE: return "DECODE";
    case QOT_SESSION_STATE_STOPPING: return "STOPPING";
    case QOT_SESSION_STATE_DONE: return "DONE";
    case QOT_SESSION_STATE_ERROR: return "ERROR";
    default: return "UNKNOWN";
    }
}

const char *qot_session_stop_reason_name(qot_session_stop_reason_t reason)
{
    switch (reason) {
    case QOT_SESSION_STOP_NONE: return "NONE";
    case QOT_SESSION_STOP_EOS: return "EOS";
    case QOT_SESSION_STOP_IM_END: return "IM_END";
    case QOT_SESSION_STOP_MAX_NEW: return "MAX_NEW";
    case QOT_SESSION_STOP_CONTEXT: return "CONTEXT";
    case QOT_SESSION_STOP_HW_ERROR: return "HW_ERROR";
    default: return "UNKNOWN";
    }
}
