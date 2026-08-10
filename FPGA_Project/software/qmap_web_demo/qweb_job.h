#ifndef QWEB_JOB_H
#define QWEB_JOB_H

#include <stddef.h>
#include <stdint.h>

#include "qot_session.h"
#include "qtk_text_tokenizer.h"
#include "qweb_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QWEB_JOB_TOKENIZER_WORKSPACE_CAPACITY \
    (QWEB_API_MAX_PROMPT_BYTES * 4u)
#define QWEB_JOB_MAX_OUTPUT_BYTES (QWEB_API_MAX_TOKENS * 128u)

enum {
    QWEB_JOB_OK = 0,
    QWEB_JOB_ERR_NULL = -1,
    QWEB_JOB_ERR_STATE = -2,
    QWEB_JOB_ERR_BUSY = -3,
    QWEB_JOB_ERR_REQUEST = -4,
    QWEB_JOB_ERR_TOKENIZE = -5,
    QWEB_JOB_ERR_SESSION = -6,
    QWEB_JOB_ERR_CAPACITY = -7,
    QWEB_JOB_ERR_DETOKENIZE = -8
};

typedef enum qweb_job_state {
    QWEB_JOB_STATE_UNINITIALIZED = 0,
    QWEB_JOB_STATE_IDLE,
    QWEB_JOB_STATE_QUEUED,
    QWEB_JOB_STATE_RUNNING,
    QWEB_JOB_STATE_DONE,
    QWEB_JOB_STATE_ERROR
} qweb_job_state_t;

/*
 * This object is intentionally large and allocation-free. Bare-metal callers
 * must place it in static/global storage rather than on the small BSP stack.
 */
typedef struct qweb_job {
    qweb_job_state_t state;
    uint32_t job_id;
    uint32_t next_job_id;
    uint32_t max_polls_per_token;
    uint32_t prompt_token_count;
    uint32_t prompt_tokens_consumed;
    uint32_t generated_count;
    uint32_t prompt_token_ids[QWEB_API_MAX_TOKENS];
    uint32_t generated_token_ids[QWEB_API_MAX_TOKENS];
    int64_t generated_scores_q26[QWEB_API_MAX_TOKENS];
    uint8_t output_bytes[QWEB_JOB_MAX_OUTPUT_BYTES];
    size_t output_length;
    int error_code;
    int session_error_code;
    qtk_status_t tokenizer_status;
    qtk_tokenizer_requirements_t tokenizer_requirements;
    qot_session_stop_reason_t stop_reason;
    qot_session_t session;
    const qtk_asset_t *tokenizer;
    uint32_t tokenizer_unicode_workspace[
        QWEB_JOB_TOKENIZER_WORKSPACE_CAPACITY];
    uint32_t tokenizer_piece_workspace[
        QWEB_JOB_TOKENIZER_WORKSPACE_CAPACITY];
    uint32_t submit_token_workspace[QWEB_API_MAX_TOKENS];
} qweb_job_t;

/*
 * runner is mandatory for the Web job. The board implementation must service
 * lwIP input and TTC timers at bounded intervals while it polls one PL token;
 * passing NULL is rejected so the non-pumping default runtime loop cannot be
 * selected accidentally.
 */
int qweb_job_init(
    qweb_job_t *job,
    uintptr_t qot_base,
    const qot_run_config_t *config,
    const qtk_asset_t *tokenizer,
    qot_session_runner_fn runner,
    void *runner_context,
    uint32_t max_polls_per_token);

/*
 * Validate, tokenize/copy, and enqueue exactly one bounded generation job.
 * DONE/ERROR results remain queryable until the next successful submission;
 * this is deliberately a one-client, one-retained-record demo contract.
 * Rejected submissions do not modify the retained record.
 */
int qweb_job_submit(
    qweb_job_t *job,
    const qweb_generate_request_t *request,
    uint32_t *accepted_job_id);

/* Advance at most one complete PL token invocation. */
int qweb_job_step(qweb_job_t *job, qot_session_event_t *event);

int qweb_job_is_active(const qweb_job_t *job);
const char *qweb_job_state_name(qweb_job_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* QWEB_JOB_H */
