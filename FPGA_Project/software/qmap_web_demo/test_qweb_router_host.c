#include "qweb_router.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

typedef int (*test_fn_t)(void);

static qweb_job_t g_job;
static qweb_router_t g_router;
static qtk_asset_t g_tokenizer;
static uint8_t g_json_scratch[QWEB_ROUTER_MAX_JSON_BODY_BYTES];
static unsigned g_tests_run;
static unsigned g_tests_failed;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n",                     \
                    __FILE__, __LINE__, #condition);                            \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static void run_test(const char *name, test_fn_t function)
{
    int result;

    ++g_tests_run;
    result = function();
    if (result != 0) {
        ++g_tests_failed;
        fprintf(stderr, "FAIL %s\n", name);
    } else {
        printf("PASS %s\n", name);
    }
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
    (void)runner_context;
    (void)base;
    (void)config;
    (void)input_token_id;
    (void)position;
    (void)max_polls;
    if (result == NULL) return QOT_ERR_NULL;
    memset(result, 0, sizeof(*result));
    return QOT_OK;
}

static int initialize_router(void)
{
    qot_run_config_t config;
    int rc;

    memset(&config, 0, sizeof(config));
    config.runtime_context_enable = 1u;
    config.embedding_enable = 1u;
    memset(&g_tokenizer, 0, sizeof(g_tokenizer));
    g_tokenizer.model_vocab_size = QOT_VOCAB_SIZE;
    g_tokenizer.eos_token_id = QOT_SESSION_DEFAULT_EOS_TOKEN_ID;

    rc = qweb_job_init(&g_job,
                       (uintptr_t)0u,
                       &config,
                       &g_tokenizer,
                       mock_run_token,
                       NULL,
                       100u);
    if (rc != QWEB_JOB_OK) return rc;
    return qweb_router_init(&g_router, &g_job);
}

static void make_request(
    qweb_http_request_t *request,
    const char *method,
    const char *target,
    const char *content_type,
    const uint8_t *body,
    size_t body_length)
{
    size_t method_length = strlen(method);
    size_t target_length = strlen(target);

    memset(request, 0, sizeof(*request));
    if (method_length <= QWEB_HTTP_MAX_METHOD_BYTES) {
        memcpy(request->method, method, method_length);
        request->method_length = method_length;
    }
    if (target_length <= QWEB_HTTP_MAX_TARGET_BYTES) {
        memcpy(request->target, target, target_length);
        request->target_length = target_length;
    }
    if (content_type != NULL) {
        size_t content_type_length = strlen(content_type);

        if (content_type_length <= QWEB_HTTP_MAX_CONTENT_TYPE_BYTES) {
            memcpy(request->content_type, content_type, content_type_length);
            request->content_type_length = content_type_length;
            request->has_content_type = 1u;
        }
    }
    if (body != NULL && body_length <= sizeof(request->body)) {
        memcpy(request->body, body, body_length);
        request->body_length = body_length;
        request->content_length = body_length;
    }
}

static int body_equals(
    const qweb_route_response_t *response,
    const uint8_t *expected,
    size_t expected_length)
{
    return response->body_length == expected_length &&
           memcmp(response->body, expected, expected_length) == 0;
}

static int body_contains(
    const qweb_route_response_t *response,
    const char *needle)
{
    size_t needle_length = strlen(needle);
    size_t offset;

    if (needle_length == 0u) return 1;
    if (needle_length > response->body_length) return 0;
    for (offset = 0u;
         offset <= response->body_length - needle_length;
         ++offset) {
        if (memcmp(response->body + offset, needle, needle_length) == 0) {
            return 1;
        }
    }
    return 0;
}

static int submit_one_token(qweb_route_response_t *response)
{
    static const uint8_t body[] =
        "{\"tokens\":[374],\"max_new_tokens\":2}";
    qweb_http_request_t request;

    make_request(&request,
                 "POST",
                 "/api/generate",
                 "application/json",
                 body,
                 sizeof(body) - 1u);
    return qweb_router_handle(&g_router,
                              &request,
                              g_json_scratch,
                              sizeof(g_json_scratch),
                              response);
}

static int test_init_and_health(void)
{
    static const uint8_t expected[] =
        "{\"service\":\"qmap-web\",\"ready\":true,\"job_state\":\"idle\"}";
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(qweb_router_init(NULL, &g_job) == QWEB_ROUTER_ERR_NULL);
    CHECK(qweb_router_init(&g_router, NULL) == QWEB_ROUTER_ERR_NULL);
    CHECK(initialize_router() == QWEB_ROUTER_OK);
    make_request(&request, "GET", "/api/health", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 200u);
    CHECK(response.content_type == QWEB_HTTP_CONTENT_JSON);
    CHECK(body_equals(&response, expected, sizeof(expected) - 1u));
    return 0;
}

static int test_http_parser_to_accepted_job(void)
{
    static const uint8_t raw_request[] =
        "POST /api/generate HTTP/1.1\r\n"
        "Host: 192.168.10.2\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 39\r\n"
        "Connection: close\r\n"
        "\r\n"
        "{\"tokens\":[785,374],\"max_new_tokens\":2}";
    static const uint8_t expected[] = "{\"job_id\":1,\"state\":\"queued\"}";
    qweb_http_parser_t parser;
    qweb_http_feed_result_t feed;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    qweb_http_parser_init(&parser);
    feed = qweb_http_parser_feed(&parser,
                                 raw_request,
                                 sizeof(raw_request) - 1u);
    CHECK(feed.status == QWEB_HTTP_PARSE_COMPLETE);
    CHECK(feed.consumed == sizeof(raw_request) - 1u);
    CHECK(qweb_router_handle(&g_router,
                             &parser.request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 202u);
    CHECK(body_equals(&response, expected, sizeof(expected) - 1u));
    CHECK(g_job.state == QWEB_JOB_STATE_QUEUED);
    CHECK(g_job.job_id == 1u);
    CHECK(g_job.prompt_token_count == 2u);
    CHECK(g_job.prompt_token_ids[0] == 785u);
    CHECK(g_job.prompt_token_ids[1] == 374u);
    return 0;
}

static int test_strict_media_type_and_json(void)
{
    static const uint8_t valid_body[] =
        "{\"tokens\":[374],\"max_new_tokens\":2}";
    static const uint8_t invalid_body[] = "{";
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    make_request(&request,
                 "POST",
                 "/api/generate",
                 "application/json; charset=utf-8",
                 valid_body,
                 sizeof(valid_body) - 1u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 415u);
    CHECK(body_contains(&response, "unsupported_media_type"));

    make_request(&request,
                 "POST",
                 "/api/generate",
                 "Application/JSON",
                 invalid_body,
                 sizeof(invalid_body) - 1u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 422u);
    CHECK(body_contains(&response, "invalid_generation_request"));
    CHECK(body_contains(&response, "SYNTAX"));
    return 0;
}

static int test_busy_conflict_after_strict_parse(void)
{
    static const uint8_t malformed[] = "{";
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    CHECK(submit_one_token(&response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 202u);
    CHECK(submit_one_token(&response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 409u);
    CHECK(body_contains(&response, "job_busy"));

    make_request(&request,
                 "POST",
                 "/api/generate",
                 "application/json",
                 malformed,
                 sizeof(malformed) - 1u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 422u);
    return 0;
}

static int test_method_and_route_errors(void)
{
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    make_request(&request, "GET", "/api/generate", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 405u);

    make_request(&request, "POST", "/api/health", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 405u);

    make_request(&request, "GET", "/no-such-route", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 404u);

    make_request(&request,
                 "GET",
                 "/api/generate/4294967296",
                 NULL,
                 NULL,
                 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 404u);

    make_request(&request, "GET", "/api/generate/0", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 404u);
    return 0;
}

static int test_done_status_exact_json(void)
{
    static const uint8_t expected[] =
        "{\"job_id\":1,\"state\":\"done\",\"prompt_token_count\":1,"
        "\"prompt_tokens_consumed\":1,\"generated_count\":2,"
        "\"generated_token_ids\":[28458,64],"
        "\"generated_scores_q26\":[1227344433,-1015661901],"
        "\"output_length\":4,\"stop_reason\":\"MAX_NEW\","
        "\"error\":{\"job_code\":0,\"session_code\":0,"
        "\"tokenizer_code\":0}}";
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    CHECK(submit_one_token(&response) == QWEB_ROUTER_OK);
    g_job.state = QWEB_JOB_STATE_DONE;
    g_job.prompt_tokens_consumed = 1u;
    g_job.generated_count = 2u;
    g_job.generated_token_ids[0] = 28458u;
    g_job.generated_token_ids[1] = 64u;
    g_job.generated_scores_q26[0] = INT64_C(1227344433);
    g_job.generated_scores_q26[1] = -INT64_C(1015661901);
    g_job.output_length = 4u;
    g_job.stop_reason = QOT_SESSION_STOP_MAX_NEW;
    make_request(&request, "GET", "/api/generate/1", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 200u);
    CHECK(body_equals(&response, expected, sizeof(expected) - 1u));
    return 0;
}

static int test_error_status_fields(void)
{
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    CHECK(submit_one_token(&response) == QWEB_ROUTER_OK);
    g_job.state = QWEB_JOB_STATE_ERROR;
    g_job.error_code = QWEB_JOB_ERR_SESSION;
    g_job.session_error_code = QOT_ERR_STATUS;
    g_job.tokenizer_status = QTK_ERR_UTF8;
    g_job.stop_reason = QOT_SESSION_STOP_HW_ERROR;
    make_request(&request, "GET", "/api/generate/1", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 200u);
    CHECK(body_contains(&response, "\"state\":\"error\""));
    CHECK(body_contains(&response, "\"stop_reason\":\"HW_ERROR\""));
    CHECK(body_contains(&response, "\"job_code\":-6"));
    CHECK(body_contains(&response, "\"tokenizer_code\":-14"));
    return 0;
}

static int test_output_preserves_arbitrary_octets(void)
{
    static const uint8_t arbitrary_output[] = {0xffu, 0x00u, 0xc3u, 0x28u};
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    CHECK(submit_one_token(&response) == QWEB_ROUTER_OK);
    memcpy(g_job.output_bytes, arbitrary_output, sizeof(arbitrary_output));
    g_job.output_length = sizeof(arbitrary_output);
    make_request(&request,
                 "GET",
                 "/api/generate/1/output",
                 NULL,
                 NULL,
                 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             NULL,
                             0u,
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 200u);
    CHECK(response.content_type == QWEB_HTTP_CONTENT_OCTETS);
    CHECK(response.body == g_job.output_bytes);
    CHECK(body_equals(&response,
                      arbitrary_output,
                      sizeof(arbitrary_output)));
    return 0;
}

static int test_unknown_job_and_incomplete_body(void)
{
    static const uint8_t body[] =
        "{\"tokens\":[374],\"max_new_tokens\":2}";
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    make_request(&request, "GET", "/api/generate/1", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 404u);

    make_request(&request,
                 "POST",
                 "/api/generate",
                 "application/json",
                 body,
                 sizeof(body) - 1u);
    request.content_length += 1u;
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 400u);
    CHECK(body_contains(&response, "incomplete_request"));
    return 0;
}

static int test_status_worst_case_capacity_and_small_scratch(void)
{
    qweb_http_request_t request;
    qweb_route_response_t response;
    uint8_t tiny[8];
    uint32_t index;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    CHECK(submit_one_token(&response) == QWEB_ROUTER_OK);
    g_job.state = QWEB_JOB_STATE_DONE;
    g_job.prompt_token_count = QWEB_API_MAX_TOKENS;
    g_job.prompt_tokens_consumed = QWEB_API_MAX_TOKENS;
    g_job.generated_count = QWEB_API_MAX_TOKENS;
    g_job.output_length = QWEB_JOB_MAX_OUTPUT_BYTES;
    g_job.stop_reason = QOT_SESSION_STOP_CONTEXT;
    for (index = 0u; index < QWEB_API_MAX_TOKENS; ++index) {
        g_job.generated_token_ids[index] = QWEB_API_MODEL_VOCAB_SIZE - 1u;
        g_job.generated_scores_q26[index] =
            (index & 1u) == 0u ? INT64_MIN : INT64_MAX;
    }
    make_request(&request, "GET", "/api/generate/1", NULL, NULL, 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 200u);
    CHECK(response.body_length <= QWEB_ROUTER_MAX_JSON_BODY_BYTES);
    CHECK(body_contains(&response, "-9223372036854775808"));
    CHECK(body_contains(&response, "9223372036854775807"));

    CHECK(qweb_router_handle(&g_router,
                             &request,
                             tiny,
                             sizeof(tiny),
                             &response) == QWEB_ROUTER_ERR_CAPACITY);
    return 0;
}

static int test_corrupt_output_length_fails_closed(void)
{
    qweb_http_request_t request;
    qweb_route_response_t response;

    CHECK(initialize_router() == QWEB_ROUTER_OK);
    CHECK(submit_one_token(&response) == QWEB_ROUTER_OK);
    g_job.output_length = QWEB_JOB_MAX_OUTPUT_BYTES + 1u;
    make_request(&request,
                 "GET",
                 "/api/generate/1/output",
                 NULL,
                 NULL,
                 0u);
    CHECK(qweb_router_handle(&g_router,
                             &request,
                             g_json_scratch,
                             sizeof(g_json_scratch),
                             &response) == QWEB_ROUTER_OK);
    CHECK(response.status_code == 500u);
    CHECK(body_contains(&response, "job_state_corrupt"));
    return 0;
}

int main(void)
{
    run_test("init_and_health", test_init_and_health);
    run_test("http_parser_to_accepted_job",
             test_http_parser_to_accepted_job);
    run_test("strict_media_type_and_json",
             test_strict_media_type_and_json);
    run_test("busy_conflict_after_strict_parse",
             test_busy_conflict_after_strict_parse);
    run_test("method_and_route_errors", test_method_and_route_errors);
    run_test("done_status_exact_json", test_done_status_exact_json);
    run_test("error_status_fields", test_error_status_fields);
    run_test("output_preserves_arbitrary_octets",
             test_output_preserves_arbitrary_octets);
    run_test("unknown_job_and_incomplete_body",
             test_unknown_job_and_incomplete_body);
    run_test("status_worst_case_capacity_and_small_scratch",
             test_status_worst_case_capacity_and_small_scratch);
    run_test("corrupt_output_length_fails_closed",
             test_corrupt_output_length_fails_closed);

    printf("qweb router host tests: %u run, %u failed\n",
           g_tests_run,
           g_tests_failed);
    return g_tests_failed == 0u ? 0 : 1;
}
