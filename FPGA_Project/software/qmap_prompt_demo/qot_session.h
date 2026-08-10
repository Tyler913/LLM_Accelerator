#ifndef QOT_SESSION_H
#define QOT_SESSION_H

#include <stdint.h>

#include "qmap_one_token_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QOT_SESSION_DEFAULT_EOS_TOKEN_ID    UINT32_C(151643)
#define QOT_SESSION_DEFAULT_IM_END_TOKEN_ID UINT32_C(151645)

enum {
    QOT_SESSION_ERR_STATE = -100
};

typedef enum qot_session_state {
    QOT_SESSION_STATE_UNINITIALIZED = 0,
    QOT_SESSION_STATE_IDLE,
    QOT_SESSION_STATE_PREFILL,
    QOT_SESSION_STATE_DECODE,
    QOT_SESSION_STATE_STOPPING,
    QOT_SESSION_STATE_DONE,
    QOT_SESSION_STATE_ERROR
} qot_session_state_t;

typedef enum qot_session_stop_reason {
    QOT_SESSION_STOP_NONE = 0,
    QOT_SESSION_STOP_EOS,
    QOT_SESSION_STOP_IM_END,
    QOT_SESSION_STOP_MAX_NEW,
    QOT_SESSION_STOP_CONTEXT,
    QOT_SESSION_STOP_HW_ERROR
} qot_session_stop_reason_t;

typedef enum qot_session_event_kind {
    QOT_SESSION_EVENT_NONE = 0,
    QOT_SESSION_EVENT_PREFILL,
    QOT_SESSION_EVENT_TOKEN,
    QOT_SESSION_EVENT_DONE,
    QOT_SESSION_EVENT_ERROR
} qot_session_event_kind_t;

/*
 * A runner performs exactly one full-model token invocation. The injectable
 * context keeps host tests independent of MMIO while the default runner calls
 * qot_run_token() unchanged.
 */
typedef int (*qot_session_runner_fn)(
    void *runner_context,
    uintptr_t base,
    const qot_run_config_t *config,
    uint32_t input_token_id,
    uint32_t position,
    uint32_t max_polls,
    qot_result_t *result);

typedef struct qot_session_options {
    uint32_t max_new_tokens;
    uint32_t max_polls_per_token;
    uint32_t eos_token_id;
    uint32_t im_end_token_id;
    uint32_t stop_on_eos;
    uint32_t stop_on_im_end;
} qot_session_options_t;

typedef struct qot_session_event {
    qot_session_event_kind_t kind;
    qot_session_stop_reason_t stop_reason;
    uint32_t input_token_id;
    uint32_t input_position;
    uint32_t prompt_tokens_consumed;
    uint32_t generated_index;
    int error_code;
    qot_result_t result;
} qot_session_event_t;

typedef struct qot_session {
    uintptr_t base;
    qot_run_config_t config;
    qot_session_runner_fn runner;
    void *runner_context;
    const uint32_t *prompt_token_ids;
    uint32_t prompt_token_count;
    uint32_t prompt_index;
    uint32_t next_position;
    uint32_t pending_token_id;
    uint32_t generated_count;
    qot_session_options_t options;
    qot_result_t last_result;
    int last_error;
    qot_session_state_t state;
    qot_session_stop_reason_t stop_reason;
} qot_session_t;

void qot_session_default_options(qot_session_options_t *options);

int qot_session_init(
    qot_session_t *session,
    uintptr_t base,
    const qot_run_config_t *config,
    qot_session_runner_fn runner,
    void *runner_context);

/*
 * Begin a fresh sequence. prompt_token_ids must remain valid until the session
 * reaches DONE or ERROR. A new begin call resets all prior software state; PL
 * attention correctness relies on the hardware reading KV entries only through
 * the current position.
 */
int qot_session_begin(
    qot_session_t *session,
    const uint32_t *prompt_token_ids,
    uint32_t prompt_token_count,
    const qot_session_options_t *options);

/* Execute at most one model invocation and report one state-machine event. */
int qot_session_step(qot_session_t *session, qot_session_event_t *event);

void qot_session_reset(qot_session_t *session);
int qot_session_is_terminal(const qot_session_t *session);

const char *qot_session_state_name(qot_session_state_t state);
const char *qot_session_stop_reason_name(qot_session_stop_reason_t reason);

#ifdef __cplusplus
}
#endif

#endif /* QOT_SESSION_H */
