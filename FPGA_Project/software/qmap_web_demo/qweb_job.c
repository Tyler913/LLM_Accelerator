#include "qweb_job.h"

#include <string.h>

#if QWEB_API_MAX_TOKENS != QOT_MAX_CONTEXT
#error "Web API token capacity must match the QOT context capacity"
#endif

#if QWEB_API_MODEL_VOCAB_SIZE != QOT_VOCAB_SIZE
#error "Web API vocabulary must match the QOT model vocabulary"
#endif

typedef struct qweb_output_sink {
    qweb_job_t *job;
    int overflow;
} qweb_output_sink_t;

static void qweb_job_clear_execution(qweb_job_t *job)
{
    job->job_id = 0u;
    job->prompt_token_count = 0u;
    job->prompt_tokens_consumed = 0u;
    job->generated_count = 0u;
    memset(job->prompt_token_ids, 0, sizeof(job->prompt_token_ids));
    memset(job->generated_token_ids, 0, sizeof(job->generated_token_ids));
    memset(job->generated_scores_q26, 0, sizeof(job->generated_scores_q26));
    memset(job->output_bytes, 0, sizeof(job->output_bytes));
    job->output_length = 0u;
    job->error_code = QWEB_JOB_OK;
    job->session_error_code = QOT_OK;
    job->tokenizer_status = QTK_OK;
    memset(&job->tokenizer_requirements, 0,
           sizeof(job->tokenizer_requirements));
    job->stop_reason = QOT_SESSION_STOP_NONE;
    qot_session_reset(&job->session);
}

static int qweb_job_validate_request(const qweb_generate_request_t *request)
{
    uint32_t index;

    if (request == NULL) return QWEB_JOB_ERR_NULL;
    if (request->max_new_tokens == 0u ||
        request->max_new_tokens > QWEB_API_MAX_TOKENS) {
        return QWEB_JOB_ERR_REQUEST;
    }
    if (request->kind == QWEB_GENERATE_INPUT_PROMPT) {
        if (request->prompt_length == 0u ||
            request->prompt_length > QWEB_API_MAX_PROMPT_BYTES ||
            request->token_count != 0u) {
            return QWEB_JOB_ERR_REQUEST;
        }
        return QWEB_JOB_OK;
    }
    if (request->kind != QWEB_GENERATE_INPUT_TOKENS ||
        request->prompt_length != 0u ||
        request->token_count == 0u ||
        request->token_count > QWEB_API_MAX_TOKENS) {
        return QWEB_JOB_ERR_REQUEST;
    }
    for (index = 0u; index < request->token_count; ++index) {
        if (request->token_ids[index] >= QOT_VOCAB_SIZE) {
            return QWEB_JOB_ERR_REQUEST;
        }
    }
    return QWEB_JOB_OK;
}

static int qweb_job_emit_output(
    void *context,
    const uint8_t *bytes,
    size_t length)
{
    qweb_output_sink_t *sink = (qweb_output_sink_t *)context;
    qweb_job_t *job;

    if (sink == NULL || sink->job == NULL ||
        (bytes == NULL && length != 0u)) {
        return 1;
    }
    job = sink->job;
    if (length > sizeof(job->output_bytes) - job->output_length) {
        sink->overflow = 1;
        return 1;
    }
    if (length != 0u) {
        memcpy(job->output_bytes + job->output_length, bytes, length);
        job->output_length += length;
    }
    return 0;
}

static int qweb_job_fail(
    qweb_job_t *job,
    int error_code,
    int session_error_code,
    qtk_status_t tokenizer_status)
{
    job->state = QWEB_JOB_STATE_ERROR;
    job->error_code = error_code;
    job->session_error_code = session_error_code;
    job->tokenizer_status = tokenizer_status;
    job->stop_reason = (error_code == QWEB_JOB_ERR_SESSION)
                           ? job->session.stop_reason
                           : QOT_SESSION_STOP_NONE;
    return error_code;
}

int qweb_job_init(
    qweb_job_t *job,
    uintptr_t qot_base,
    const qot_run_config_t *config,
    const qtk_asset_t *tokenizer,
    qot_session_runner_fn runner,
    void *runner_context,
    uint32_t max_polls_per_token)
{
    int rc;

    if (job == NULL || config == NULL || tokenizer == NULL || runner == NULL) {
        return QWEB_JOB_ERR_NULL;
    }
    if (tokenizer->model_vocab_size != QOT_VOCAB_SIZE ||
        tokenizer->eos_token_id >= QOT_VOCAB_SIZE) {
        return QWEB_JOB_ERR_REQUEST;
    }

    memset(job, 0, sizeof(*job));
    rc = qot_session_init(&job->session,
                          qot_base,
                          config,
                          runner,
                          runner_context);
    if (rc != QOT_OK) {
        job->state = QWEB_JOB_STATE_ERROR;
        job->error_code = QWEB_JOB_ERR_SESSION;
        job->session_error_code = rc;
        return QWEB_JOB_ERR_SESSION;
    }
    job->tokenizer = tokenizer;
    job->max_polls_per_token = max_polls_per_token;
    job->next_job_id = 1u;
    job->state = QWEB_JOB_STATE_IDLE;
    return QWEB_JOB_OK;
}

int qweb_job_submit(
    qweb_job_t *job,
    const qweb_generate_request_t *request,
    uint32_t *accepted_job_id)
{
    uint32_t candidate_count;
    qtk_status_t candidate_tokenizer_status = QTK_OK;
    qtk_tokenizer_requirements_t candidate_requirements;
    qot_session_options_t options;
    size_t tokenized_count = 0u;
    int rc;

    if (accepted_job_id != NULL) *accepted_job_id = 0u;
    if (job == NULL || request == NULL) return QWEB_JOB_ERR_NULL;
    if (job->state == QWEB_JOB_STATE_UNINITIALIZED ||
        job->tokenizer == NULL) {
        return QWEB_JOB_ERR_STATE;
    }
    if (qweb_job_is_active(job)) return QWEB_JOB_ERR_BUSY;

    rc = qweb_job_validate_request(request);
    if (rc != QWEB_JOB_OK) return rc;

    memset(job->submit_token_workspace, 0,
           sizeof(job->submit_token_workspace));
    memset(&candidate_requirements, 0, sizeof(candidate_requirements));
    if (request->kind == QWEB_GENERATE_INPUT_PROMPT) {
        candidate_tokenizer_status = qtk_tokenize_utf8(
            job->tokenizer,
            request->prompt,
            request->prompt_length,
            job->tokenizer_unicode_workspace,
            QWEB_JOB_TOKENIZER_WORKSPACE_CAPACITY,
            job->tokenizer_piece_workspace,
            QWEB_JOB_TOKENIZER_WORKSPACE_CAPACITY,
            job->submit_token_workspace,
            QWEB_API_MAX_TOKENS,
            &tokenized_count,
            &candidate_requirements);
        if (candidate_tokenizer_status != QTK_OK || tokenized_count == 0u) {
            return QWEB_JOB_ERR_TOKENIZE;
        }
        candidate_count = (uint32_t)tokenized_count;
    } else {
        candidate_count = request->token_count;
        memcpy(job->submit_token_workspace,
               request->token_ids,
               (size_t)candidate_count *
                   sizeof(job->submit_token_workspace[0]));
    }
    if ((uint64_t)job->session.config.position +
            (uint64_t)candidate_count >
        (uint64_t)QOT_MAX_CONTEXT) {
        return QWEB_JOB_ERR_REQUEST;
    }

    qweb_job_clear_execution(job);
    job->tokenizer_status = candidate_tokenizer_status;
    job->tokenizer_requirements = candidate_requirements;
    memcpy(job->prompt_token_ids,
           job->submit_token_workspace,
           (size_t)candidate_count * sizeof(job->prompt_token_ids[0]));
    job->prompt_token_count = candidate_count;

    qot_session_default_options(&options);
    options.max_new_tokens = request->max_new_tokens;
    options.max_polls_per_token = job->max_polls_per_token;
    options.eos_token_id = job->tokenizer->eos_token_id;
    rc = qot_session_begin(&job->session,
                           job->prompt_token_ids,
                           job->prompt_token_count,
                           &options);
    if (rc != QOT_OK) {
        return qweb_job_fail(job,
                             QWEB_JOB_ERR_SESSION,
                             rc,
                             QTK_OK);
    }

    job->job_id = job->next_job_id;
    ++job->next_job_id;
    if (job->next_job_id == 0u) job->next_job_id = 1u;
    job->state = QWEB_JOB_STATE_QUEUED;
    if (accepted_job_id != NULL) *accepted_job_id = job->job_id;
    return QWEB_JOB_OK;
}

int qweb_job_step(qweb_job_t *job, qot_session_event_t *event)
{
    qot_session_event_t local_event;
    qweb_output_sink_t sink;
    qtk_status_t detokenize_status;
    int rc;

    if (job == NULL) return QWEB_JOB_ERR_NULL;
    if (job->state != QWEB_JOB_STATE_QUEUED &&
        job->state != QWEB_JOB_STATE_RUNNING) {
        return QWEB_JOB_ERR_STATE;
    }
    if (event == NULL) event = &local_event;
    job->state = QWEB_JOB_STATE_RUNNING;

    rc = qot_session_step(&job->session, event);
    if (rc != QOT_OK || event->kind == QOT_SESSION_EVENT_ERROR) {
        int session_rc = (rc != QOT_OK) ? rc : event->error_code;
        return qweb_job_fail(job,
                             QWEB_JOB_ERR_SESSION,
                             session_rc,
                             QTK_OK);
    }

    if (event->kind == QOT_SESSION_EVENT_PREFILL) {
        job->prompt_tokens_consumed = event->prompt_tokens_consumed;
        return QWEB_JOB_OK;
    }
    if (event->kind == QOT_SESSION_EVENT_TOKEN) {
        uint32_t index = job->generated_count;

        if (index >= QWEB_API_MAX_TOKENS ||
            event->generated_index != index) {
            return qweb_job_fail(job,
                                 QWEB_JOB_ERR_CAPACITY,
                                 QOT_OK,
                                 QTK_OK);
        }
        job->generated_token_ids[index] = event->result.token_id;
        job->generated_scores_q26[index] = event->result.score_q26;
        ++job->generated_count;
        job->prompt_tokens_consumed = event->prompt_tokens_consumed;

        sink.job = job;
        sink.overflow = 0;
        detokenize_status = qtk_detokenize_ids(
            job->tokenizer,
            &event->result.token_id,
            1u,
            1,
            qweb_job_emit_output,
            &sink);
        if (detokenize_status != QTK_OK || sink.overflow != 0) {
            int job_error = (sink.overflow != 0)
                                ? QWEB_JOB_ERR_CAPACITY
                                : QWEB_JOB_ERR_DETOKENIZE;
            return qweb_job_fail(job,
                                 job_error,
                                 QOT_OK,
                                 detokenize_status);
        }
        return QWEB_JOB_OK;
    }
    if (event->kind == QOT_SESSION_EVENT_DONE) {
        job->prompt_tokens_consumed = event->prompt_tokens_consumed;
        job->stop_reason = event->stop_reason;
        job->state = QWEB_JOB_STATE_DONE;
        return QWEB_JOB_OK;
    }
    return qweb_job_fail(job,
                         QWEB_JOB_ERR_SESSION,
                         QOT_SESSION_ERR_STATE,
                         QTK_OK);
}

int qweb_job_is_active(const qweb_job_t *job)
{
    if (job == NULL) return 0;
    return job->state == QWEB_JOB_STATE_QUEUED ||
           job->state == QWEB_JOB_STATE_RUNNING;
}

const char *qweb_job_state_name(qweb_job_state_t state)
{
    switch (state) {
    case QWEB_JOB_STATE_UNINITIALIZED: return "uninitialized";
    case QWEB_JOB_STATE_IDLE: return "idle";
    case QWEB_JOB_STATE_QUEUED: return "queued";
    case QWEB_JOB_STATE_RUNNING: return "running";
    case QWEB_JOB_STATE_DONE: return "done";
    case QWEB_JOB_STATE_ERROR: return "error";
    default: return "unknown";
    }
}
