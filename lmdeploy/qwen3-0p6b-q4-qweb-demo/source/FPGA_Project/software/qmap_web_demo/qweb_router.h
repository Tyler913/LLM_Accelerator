#ifndef QWEB_ROUTER_H
#define QWEB_ROUTER_H

#include <stddef.h>
#include <stdint.h>

#include "qweb_http.h"
#include "qweb_job.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Large enough for 256 token ids and 256 signed Q26 scores in one status. */
#define QWEB_ROUTER_MAX_JSON_BODY_BYTES 12288u

enum {
    QWEB_ROUTER_OK = 0,
    QWEB_ROUTER_ERR_NULL = -1,
    QWEB_ROUTER_ERR_CAPACITY = -2,
    QWEB_ROUTER_ERR_STATE = -3
};

/*
 * The request workspace keeps the decoded JSON object off the BSP stack.
 * Bare-metal callers should place qweb_http_parser_t, qweb_job_t, this router,
 * and the JSON response scratch buffer in static/global storage.
 */
typedef struct qweb_router {
    qweb_job_t *job;
    qweb_generate_request_t request_workspace;
} qweb_router_t;

/*
 * For JSON responses body points into the caller-owned scratch buffer. For the
 * output endpoint it aliases job->output_bytes and may contain arbitrary
 * octets, including NUL and incomplete UTF-8. The pointer remains valid until
 * the scratch buffer or generation job is reused.
 */
typedef struct qweb_route_response {
    uint16_t status_code;
    qweb_http_content_type_t content_type;
    const uint8_t *body;
    size_t body_length;
} qweb_route_response_t;

int qweb_router_init(qweb_router_t *router, qweb_job_t *job);

/*
 * Route one complete qweb_http_request_t. The lwIP adapter remains responsible
 * for incremental parsing and for passing this response to
 * qweb_http_format_response() or an equivalent bounded sender.
 */
int qweb_router_handle(
    qweb_router_t *router,
    const qweb_http_request_t *request,
    uint8_t *json_scratch,
    size_t json_scratch_capacity,
    qweb_route_response_t *response);

#ifdef __cplusplus
}
#endif

#endif /* QWEB_ROUTER_H */
