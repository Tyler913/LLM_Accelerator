#include "qweb_router.h"

#include <string.h>

#include "qweb_api.h"

typedef enum qweb_generation_route_kind {
    QWEB_GENERATION_ROUTE_INVALID = 0,
    QWEB_GENERATION_ROUTE_STATUS,
    QWEB_GENERATION_ROUTE_OUTPUT
} qweb_generation_route_kind_t;

typedef struct qweb_generation_route {
    qweb_generation_route_kind_t kind;
    uint32_t job_id;
} qweb_generation_route_t;

static int qweb_router_bytes_equal(
    const char *bytes,
    size_t length,
    const char *literal,
    size_t literal_length)
{
    return length == literal_length &&
           memcmp(bytes, literal, literal_length) == 0;
}

static unsigned char qweb_router_ascii_lower(unsigned char value)
{
    if (value >= (unsigned char)'A' && value <= (unsigned char)'Z') {
        return (unsigned char)(value +
                               ((unsigned char)'a' - (unsigned char)'A'));
    }
    return value;
}

static int qweb_router_content_type_is_json(
    const qweb_http_request_t *request)
{
    static const char expected[] = "application/json";
    size_t index;

    if (request->has_content_type == 0u ||
        request->content_type_length != sizeof(expected) - 1u) {
        return 0;
    }
    for (index = 0u; index < sizeof(expected) - 1u; ++index) {
        if (qweb_router_ascii_lower(
                (unsigned char)request->content_type[index]) !=
            (unsigned char)expected[index]) {
            return 0;
        }
    }
    return 1;
}

static int qweb_router_parse_u32(
    const char *bytes,
    size_t length,
    uint32_t *value)
{
    uint32_t parsed = 0u;
    size_t index;

    if (bytes == NULL || value == NULL || length == 0u) return 0;
    for (index = 0u; index < length; ++index) {
        unsigned char current = (unsigned char)bytes[index];
        uint32_t digit;

        if (current < (unsigned char)'0' ||
            current > (unsigned char)'9') {
            return 0;
        }
        digit = (uint32_t)(current - (unsigned char)'0');
        if (parsed > (UINT32_MAX - digit) / UINT32_C(10)) return 0;
        parsed = parsed * UINT32_C(10) + digit;
    }
    if (parsed == 0u) return 0;
    *value = parsed;
    return 1;
}

static qweb_generation_route_t qweb_router_parse_generation_route(
    const qweb_http_request_t *request)
{
    static const char prefix[] = "/api/generate/";
    static const char output_suffix[] = "/output";
    qweb_generation_route_t route;
    size_t identifier_length;
    size_t remaining;
    size_t slash;

    route.kind = QWEB_GENERATION_ROUTE_INVALID;
    route.job_id = 0u;
    if (request->target_length <= sizeof(prefix) - 1u ||
        memcmp(request->target, prefix, sizeof(prefix) - 1u) != 0) {
        return route;
    }

    remaining = request->target_length - (sizeof(prefix) - 1u);
    slash = 0u;
    while (slash < remaining &&
           request->target[(sizeof(prefix) - 1u) + slash] != '/') {
        ++slash;
    }
    identifier_length = slash;
    if (!qweb_router_parse_u32(request->target + sizeof(prefix) - 1u,
                               identifier_length,
                               &route.job_id)) {
        route.job_id = 0u;
        return route;
    }
    if (slash == remaining) {
        route.kind = QWEB_GENERATION_ROUTE_STATUS;
        return route;
    }
    if (remaining - slash == sizeof(output_suffix) - 1u &&
        memcmp(request->target + sizeof(prefix) - 1u + slash,
               output_suffix,
               sizeof(output_suffix) - 1u) == 0) {
        route.kind = QWEB_GENERATION_ROUTE_OUTPUT;
        return route;
    }
    route.job_id = 0u;
    return route;
}

static int qweb_router_set_json_response(
    uint16_t status_code,
    uint8_t *json_scratch,
    size_t body_length,
    qweb_route_response_t *response)
{
    response->status_code = status_code;
    response->content_type = QWEB_HTTP_CONTENT_JSON;
    response->body = json_scratch;
    response->body_length = body_length;
    return QWEB_ROUTER_OK;
}

static int qweb_router_error_response(
    uint16_t status_code,
    const char *code,
    const char *message,
    uint8_t *json_scratch,
    size_t json_scratch_capacity,
    qweb_route_response_t *response)
{
    size_t body_length = 0u;
    qweb_json_write_status_t status = qweb_api_format_error_json(
        json_scratch,
        json_scratch_capacity,
        code,
        message,
        &body_length);

    if (status == QWEB_JSON_WRITE_ERR_NULL) return QWEB_ROUTER_ERR_NULL;
    if (status != QWEB_JSON_WRITE_OK) return QWEB_ROUTER_ERR_CAPACITY;
    return qweb_router_set_json_response(status_code,
                                         json_scratch,
                                         body_length,
                                         response);
}

static int qweb_router_write_health(
    const qweb_router_t *router,
    uint8_t *json_scratch,
    size_t json_scratch_capacity,
    qweb_route_response_t *response)
{
    qweb_json_writer_t writer;
    int ready = router->job->state != QWEB_JOB_STATE_UNINITIALIZED &&
                router->job->tokenizer != NULL;

    qweb_json_writer_init(&writer, json_scratch, json_scratch_capacity);
    (void)qweb_json_writer_append_cstr(
        &writer, "{\"service\":\"qmap-web\",\"ready\":");
    (void)qweb_json_writer_append_cstr(&writer, ready ? "true" : "false");
    (void)qweb_json_writer_append_cstr(&writer, ",\"job_state\":");
    (void)qweb_json_writer_append_string(
        &writer,
        (const uint8_t *)qweb_job_state_name(router->job->state),
        strlen(qweb_job_state_name(router->job->state)));
    (void)qweb_json_writer_append_cstr(&writer, "}");
    if (writer.status == QWEB_JSON_WRITE_ERR_NULL) {
        return QWEB_ROUTER_ERR_NULL;
    }
    if (writer.status != QWEB_JSON_WRITE_OK) {
        return QWEB_ROUTER_ERR_CAPACITY;
    }
    return qweb_router_set_json_response(200u,
                                         json_scratch,
                                         writer.length,
                                         response);
}

static int qweb_router_write_accepted(
    const qweb_job_t *job,
    uint8_t *json_scratch,
    size_t json_scratch_capacity,
    qweb_route_response_t *response)
{
    qweb_json_writer_t writer;
    const char *state_name = qweb_job_state_name(job->state);

    qweb_json_writer_init(&writer, json_scratch, json_scratch_capacity);
    (void)qweb_json_writer_append_cstr(&writer, "{\"job_id\":");
    (void)qweb_json_writer_append_u32(&writer, job->job_id);
    (void)qweb_json_writer_append_cstr(&writer, ",\"state\":");
    (void)qweb_json_writer_append_string(
        &writer, (const uint8_t *)state_name, strlen(state_name));
    (void)qweb_json_writer_append_cstr(&writer, "}");
    if (writer.status == QWEB_JSON_WRITE_ERR_NULL) {
        return QWEB_ROUTER_ERR_NULL;
    }
    if (writer.status != QWEB_JSON_WRITE_OK) {
        return QWEB_ROUTER_ERR_CAPACITY;
    }
    return qweb_router_set_json_response(202u,
                                         json_scratch,
                                         writer.length,
                                         response);
}

static int qweb_router_write_u32_array(
    qweb_json_writer_t *writer,
    const uint32_t *values,
    uint32_t count)
{
    uint32_t index;

    (void)qweb_json_writer_append_cstr(writer, "[");
    for (index = 0u; index < count; ++index) {
        if (index != 0u) (void)qweb_json_writer_append_cstr(writer, ",");
        (void)qweb_json_writer_append_u32(writer, values[index]);
    }
    (void)qweb_json_writer_append_cstr(writer, "]");
    return writer->status == QWEB_JSON_WRITE_OK ? QWEB_ROUTER_OK
                                                : QWEB_ROUTER_ERR_CAPACITY;
}

static int qweb_router_write_i64_array(
    qweb_json_writer_t *writer,
    const int64_t *values,
    uint32_t count)
{
    uint32_t index;

    (void)qweb_json_writer_append_cstr(writer, "[");
    for (index = 0u; index < count; ++index) {
        if (index != 0u) (void)qweb_json_writer_append_cstr(writer, ",");
        (void)qweb_json_writer_append_i64(writer, values[index]);
    }
    (void)qweb_json_writer_append_cstr(writer, "]");
    return writer->status == QWEB_JSON_WRITE_OK ? QWEB_ROUTER_OK
                                                : QWEB_ROUTER_ERR_CAPACITY;
}

static int qweb_router_write_status(
    const qweb_job_t *job,
    uint8_t *json_scratch,
    size_t json_scratch_capacity,
    qweb_route_response_t *response)
{
    qweb_json_writer_t writer;
    const char *state_name;
    const char *stop_name;

    if (job->prompt_token_count > QWEB_API_MAX_TOKENS ||
        job->prompt_tokens_consumed > job->prompt_token_count ||
        job->generated_count > QWEB_API_MAX_TOKENS ||
        job->output_length > QWEB_JOB_MAX_OUTPUT_BYTES) {
        return qweb_router_error_response(500u,
                                          "job_state_corrupt",
                                          "job counters exceed bounds",
                                          json_scratch,
                                          json_scratch_capacity,
                                          response);
    }

    state_name = qweb_job_state_name(job->state);
    stop_name = qot_session_stop_reason_name(job->stop_reason);
    qweb_json_writer_init(&writer, json_scratch, json_scratch_capacity);
    (void)qweb_json_writer_append_cstr(&writer, "{\"job_id\":");
    (void)qweb_json_writer_append_u32(&writer, job->job_id);
    (void)qweb_json_writer_append_cstr(&writer, ",\"state\":");
    (void)qweb_json_writer_append_string(
        &writer, (const uint8_t *)state_name, strlen(state_name));
    (void)qweb_json_writer_append_cstr(
        &writer, ",\"prompt_token_count\":");
    (void)qweb_json_writer_append_u32(&writer, job->prompt_token_count);
    (void)qweb_json_writer_append_cstr(
        &writer, ",\"prompt_tokens_consumed\":");
    (void)qweb_json_writer_append_u32(&writer, job->prompt_tokens_consumed);
    (void)qweb_json_writer_append_cstr(&writer, ",\"generated_count\":");
    (void)qweb_json_writer_append_u32(&writer, job->generated_count);
    (void)qweb_json_writer_append_cstr(
        &writer, ",\"generated_token_ids\":");
    (void)qweb_router_write_u32_array(&writer,
                                      job->generated_token_ids,
                                      job->generated_count);
    (void)qweb_json_writer_append_cstr(
        &writer, ",\"generated_scores_q26\":");
    (void)qweb_router_write_i64_array(&writer,
                                      job->generated_scores_q26,
                                      job->generated_count);
    (void)qweb_json_writer_append_cstr(&writer, ",\"output_length\":");
    if (job->output_length > (size_t)UINT32_MAX) {
        return qweb_router_error_response(500u,
                                          "job_state_corrupt",
                                          "output length cannot be reported",
                                          json_scratch,
                                          json_scratch_capacity,
                                          response);
    }
    (void)qweb_json_writer_append_u32(&writer,
                                      (uint32_t)job->output_length);
    (void)qweb_json_writer_append_cstr(&writer, ",\"stop_reason\":");
    (void)qweb_json_writer_append_string(
        &writer, (const uint8_t *)stop_name, strlen(stop_name));
    (void)qweb_json_writer_append_cstr(
        &writer, ",\"error\":{\"job_code\":");
    (void)qweb_json_writer_append_i64(&writer, (int64_t)job->error_code);
    (void)qweb_json_writer_append_cstr(&writer, ",\"session_code\":");
    (void)qweb_json_writer_append_i64(&writer,
                                      (int64_t)job->session_error_code);
    (void)qweb_json_writer_append_cstr(&writer, ",\"tokenizer_code\":");
    (void)qweb_json_writer_append_i64(&writer,
                                      (int64_t)job->tokenizer_status);
    (void)qweb_json_writer_append_cstr(&writer, "}}");

    if (writer.status == QWEB_JSON_WRITE_ERR_NULL) {
        return QWEB_ROUTER_ERR_NULL;
    }
    if (writer.status != QWEB_JSON_WRITE_OK) {
        return QWEB_ROUTER_ERR_CAPACITY;
    }
    return qweb_router_set_json_response(200u,
                                         json_scratch,
                                         writer.length,
                                         response);
}

static int qweb_router_handle_submit(
    qweb_router_t *router,
    const qweb_http_request_t *request,
    uint8_t *json_scratch,
    size_t json_scratch_capacity,
    qweb_route_response_t *response)
{
    qweb_api_parse_result_t parse_result;
    uint32_t accepted_job_id = 0u;
    int job_result;

    if (!qweb_router_content_type_is_json(request)) {
        return qweb_router_error_response(415u,
                                          "unsupported_media_type",
                                          "Content-Type must be application/json",
                                          json_scratch,
                                          json_scratch_capacity,
                                          response);
    }
    parse_result = qweb_api_parse_generate_request(
        request->body,
        request->body_length,
        &router->request_workspace);
    if (parse_result.status != QWEB_API_PARSE_OK) {
        return qweb_router_error_response(
            422u,
            "invalid_generation_request",
            qweb_api_parse_status_name(parse_result.status),
            json_scratch,
            json_scratch_capacity,
            response);
    }

    job_result = qweb_job_submit(router->job,
                                 &router->request_workspace,
                                 &accepted_job_id);
    if (job_result == QWEB_JOB_ERR_BUSY) {
        return qweb_router_error_response(409u,
                                          "job_busy",
                                          "one generation job is already active",
                                          json_scratch,
                                          json_scratch_capacity,
                                          response);
    }
    if (job_result == QWEB_JOB_ERR_STATE) {
        return qweb_router_error_response(503u,
                                          "job_unavailable",
                                          "generation runtime is unavailable",
                                          json_scratch,
                                          json_scratch_capacity,
                                          response);
    }
    if (job_result == QWEB_JOB_ERR_REQUEST ||
        job_result == QWEB_JOB_ERR_TOKENIZE) {
        return qweb_router_error_response(422u,
                                          "generation_rejected",
                                          "generation request was rejected",
                                          json_scratch,
                                          json_scratch_capacity,
                                          response);
    }
    if (job_result != QWEB_JOB_OK || accepted_job_id == 0u ||
        accepted_job_id != router->job->job_id) {
        return qweb_router_error_response(500u,
                                          "generation_internal_error",
                                          "generation job could not be queued",
                                          json_scratch,
                                          json_scratch_capacity,
                                          response);
    }
    return qweb_router_write_accepted(router->job,
                                      json_scratch,
                                      json_scratch_capacity,
                                      response);
}

int qweb_router_init(qweb_router_t *router, qweb_job_t *job)
{
    if (router == NULL || job == NULL) return QWEB_ROUTER_ERR_NULL;
    memset(router, 0, sizeof(*router));
    router->job = job;
    return QWEB_ROUTER_OK;
}

int qweb_router_handle(
    qweb_router_t *router,
    const qweb_http_request_t *request,
    uint8_t *json_scratch,
    size_t json_scratch_capacity,
    qweb_route_response_t *response)
{
    static const char health_path[] = "/api/health";
    static const char generate_path[] = "/api/generate";
    qweb_generation_route_t generation_route;

    if (router == NULL || router->job == NULL || request == NULL ||
        response == NULL ||
        (json_scratch == NULL && json_scratch_capacity != 0u)) {
        return QWEB_ROUTER_ERR_NULL;
    }
    memset(response, 0, sizeof(*response));
    if (request->body_length != request->content_length ||
        request->body_length > QWEB_HTTP_MAX_BODY_BYTES) {
        return qweb_router_error_response(400u,
                                          "incomplete_request",
                                          "HTTP request body is incomplete",
                                          json_scratch,
                                          json_scratch_capacity,
                                          response);
    }

    if (qweb_router_bytes_equal(request->target,
                                request->target_length,
                                health_path,
                                sizeof(health_path) - 1u)) {
        if (!qweb_router_bytes_equal(request->method,
                                     request->method_length,
                                     "GET",
                                     3u)) {
            return qweb_router_error_response(405u,
                                              "method_not_allowed",
                                              "GET is required for this route",
                                              json_scratch,
                                              json_scratch_capacity,
                                              response);
        }
        return qweb_router_write_health(router,
                                        json_scratch,
                                        json_scratch_capacity,
                                        response);
    }

    if (qweb_router_bytes_equal(request->target,
                                request->target_length,
                                generate_path,
                                sizeof(generate_path) - 1u)) {
        if (!qweb_router_bytes_equal(request->method,
                                     request->method_length,
                                     "POST",
                                     4u)) {
            return qweb_router_error_response(405u,
                                              "method_not_allowed",
                                              "POST is required for this route",
                                              json_scratch,
                                              json_scratch_capacity,
                                              response);
        }
        return qweb_router_handle_submit(router,
                                         request,
                                         json_scratch,
                                         json_scratch_capacity,
                                         response);
    }

    generation_route = qweb_router_parse_generation_route(request);
    if (generation_route.kind != QWEB_GENERATION_ROUTE_INVALID) {
        if (!qweb_router_bytes_equal(request->method,
                                     request->method_length,
                                     "GET",
                                     3u)) {
            return qweb_router_error_response(405u,
                                              "method_not_allowed",
                                              "GET is required for this route",
                                              json_scratch,
                                              json_scratch_capacity,
                                              response);
        }
        if (router->job->job_id == 0u ||
            generation_route.job_id != router->job->job_id) {
            return qweb_router_error_response(404u,
                                              "job_not_found",
                                              "generation job is not retained",
                                              json_scratch,
                                              json_scratch_capacity,
                                              response);
        }
        if (generation_route.kind == QWEB_GENERATION_ROUTE_OUTPUT) {
            if (router->job->output_length > QWEB_JOB_MAX_OUTPUT_BYTES) {
                return qweb_router_error_response(
                    500u,
                    "job_state_corrupt",
                    "output length exceeds its fixed buffer",
                    json_scratch,
                    json_scratch_capacity,
                    response);
            }
            response->status_code = 200u;
            response->content_type = QWEB_HTTP_CONTENT_OCTETS;
            response->body = router->job->output_bytes;
            response->body_length = router->job->output_length;
            return QWEB_ROUTER_OK;
        }
        return qweb_router_write_status(router->job,
                                        json_scratch,
                                        json_scratch_capacity,
                                        response);
    }

    return qweb_router_error_response(404u,
                                      "route_not_found",
                                      "HTTP route was not found",
                                      json_scratch,
                                      json_scratch_capacity,
                                      response);
}
