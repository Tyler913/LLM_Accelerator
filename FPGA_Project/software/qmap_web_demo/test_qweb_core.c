#include "qweb_api.h"
#include "qweb_http.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*test_fn_t)(void);

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

static size_t make_post_request(
    const uint8_t *body,
    size_t body_length,
    uint8_t *output,
    size_t capacity)
{
    int header_length = snprintf(
        (char *)output,
        capacity,
        "POST /api/generate HTTP/1.1\r\n"
        "Host: 192.168.10.2\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        body_length);

    if (header_length < 0 || (size_t)header_length > capacity ||
        body_length > capacity - (size_t)header_length) {
        return 0u;
    }
    memcpy(output + (size_t)header_length, body, body_length);
    return (size_t)header_length + body_length;
}

static int check_post_request(const qweb_http_parser_t *parser,
                              const uint8_t *body,
                              size_t body_length)
{
    CHECK(parser->status == QWEB_HTTP_PARSE_COMPLETE);
    CHECK(strcmp(parser->request.method, "POST") == 0);
    CHECK(parser->request.method_length == 4u);
    CHECK(strcmp(parser->request.target, "/api/generate") == 0);
    CHECK(parser->request.content_length == body_length);
    CHECK(parser->request.body_length == body_length);
    CHECK(memcmp(parser->request.body, body, body_length) == 0);
    CHECK(parser->request.has_content_type == 1u);
    CHECK(strcmp(parser->request.content_type, "application/json") == 0);
    CHECK(parser->request.connection_close_requested == 1u);
    return 0;
}

static int test_http_every_two_way_split(void)
{
    static const uint8_t body[] =
        "{\"tokens\":[374],\"max_new_tokens\":2}";
    uint8_t request[1024];
    size_t request_length = make_post_request(body,
                                              sizeof(body) - 1u,
                                              request,
                                              sizeof(request));
    size_t split;

    CHECK(request_length != 0u);
    for (split = 0u; split <= request_length; ++split) {
        qweb_http_parser_t parser;
        qweb_http_feed_result_t first;
        qweb_http_feed_result_t second;

        qweb_http_parser_init(&parser);
        first = qweb_http_parser_feed(&parser, request, split);
        CHECK(first.consumed == split);
        if (split < request_length) {
            CHECK(first.status == QWEB_HTTP_PARSE_IN_PROGRESS);
            second = qweb_http_parser_feed(&parser,
                                           request + split,
                                           request_length - split);
            CHECK(second.status == QWEB_HTTP_PARSE_COMPLETE);
            CHECK(second.consumed == request_length - split);
        } else {
            CHECK(first.status == QWEB_HTTP_PARSE_COMPLETE);
        }
        CHECK(check_post_request(&parser, body, sizeof(body) - 1u) == 0);
    }
    return 0;
}

static int test_http_one_byte_fragments(void)
{
    static const uint8_t body[] =
        "{\"prompt\":\"FPGA\",\"max_new_tokens\":1}";
    uint8_t request[1024];
    size_t request_length = make_post_request(body,
                                              sizeof(body) - 1u,
                                              request,
                                              sizeof(request));
    qweb_http_parser_t parser;
    size_t offset;

    CHECK(request_length != 0u);
    qweb_http_parser_init(&parser);
    for (offset = 0u; offset < request_length; ++offset) {
        qweb_http_feed_result_t result = qweb_http_parser_feed(
            &parser, request + offset, 1u);

        CHECK(result.consumed == 1u);
        CHECK(result.status == (offset + 1u == request_length
                                   ? QWEB_HTTP_PARSE_COMPLETE
                                   : QWEB_HTTP_PARSE_IN_PROGRESS));
    }
    return check_post_request(&parser, body, sizeof(body) - 1u);
}

static int test_http_stops_before_pipeline_tail(void)
{
    static const uint8_t body[] = "{}";
    static const uint8_t tail[] = "GET /next HTTP/1.1\r\n";
    uint8_t request[1024];
    size_t request_length = make_post_request(body,
                                              sizeof(body) - 1u,
                                              request,
                                              sizeof(request));
    qweb_http_parser_t parser;
    qweb_http_feed_result_t result;
    qweb_http_feed_result_t repeated;

    CHECK(request_length + sizeof(tail) - 1u <= sizeof(request));
    memcpy(request + request_length, tail, sizeof(tail) - 1u);
    qweb_http_parser_init(&parser);
    result = qweb_http_parser_feed(&parser,
                                   request,
                                   request_length + sizeof(tail) - 1u);
    CHECK(result.status == QWEB_HTTP_PARSE_COMPLETE);
    CHECK(result.consumed == request_length);
    repeated = qweb_http_parser_feed(&parser, tail, sizeof(tail) - 1u);
    CHECK(repeated.status == QWEB_HTTP_PARSE_COMPLETE);
    CHECK(repeated.consumed == 0u);
    return 0;
}

static int test_http_get_and_case_insensitive_headers(void)
{
    static const uint8_t request[] =
        "GET /api/health?full=1 HTTP/1.1\r\n"
        "hOsT:\tboard\r\n"
        "CoNnEcTiOn: keep-alive, CLOSE\r\n"
        "cOnTeNt-TyPe:  application/json; charset=utf-8 \t\r\n"
        "\r\n";
    qweb_http_parser_t parser;
    qweb_http_feed_result_t result;

    qweb_http_parser_init(&parser);
    result = qweb_http_parser_feed(&parser, request, sizeof(request) - 1u);
    CHECK(result.status == QWEB_HTTP_PARSE_COMPLETE);
    CHECK(result.consumed == sizeof(request) - 1u);
    CHECK(strcmp(parser.request.method, "GET") == 0);
    CHECK(strcmp(parser.request.target, "/api/health?full=1") == 0);
    CHECK(parser.request.content_length == 0u);
    CHECK(parser.request.body_length == 0u);
    CHECK(parser.request.connection_close_requested == 1u);
    CHECK(strcmp(parser.request.content_type,
                 "application/json; charset=utf-8") == 0);
    return 0;
}

static qweb_http_parse_status_t parse_http_status(const uint8_t *request,
                                                  size_t length)
{
    qweb_http_parser_t parser;
    qweb_http_feed_result_t result;

    qweb_http_parser_init(&parser);
    result = qweb_http_parser_feed(&parser, request, length);
    return result.status;
}

static int test_http_rejects_ambiguous_or_invalid_headers(void)
{
    struct invalid_case {
        const char *request;
        qweb_http_parse_status_t expected;
    } cases[] = {
        {"GET / HTTP/1.0\r\nHost: board\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_VERSION},
        {"GET / HTTP/1.1\r\n\r\n", QWEB_HTTP_PARSE_ERR_HOST},
        {"GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_HOST},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\n"
         "Content-Length: 0\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_CONTENT_LENGTH},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: x\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_CONTENT_LENGTH},
        {"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 8193\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_BODY_TOO_LARGE},
        {"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_TRANSFER_ENCODING},
        {"GET / HTTP/1.1\nHost: a\n\n", QWEB_HTTP_PARSE_ERR_SYNTAX},
        {"GET http://board/ HTTP/1.1\r\nHost: a\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_SYNTAX},
        {"GET /path#fragment HTTP/1.1\r\nHost: a\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_SYNTAX},
        {"GET  / HTTP/1.1\r\nHost: a\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_SYNTAX},
        {"GET / HTTP/1.1\r\n folded: no\r\nHost: a\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_SYNTAX},
        {"GET / HTTP/1.1\r\nHost : a\r\n\r\n",
         QWEB_HTTP_PARSE_ERR_SYNTAX},
        {"GET / HTTP/1.1\r\nHost:\r\n\r\n", QWEB_HTTP_PARSE_ERR_HOST},
    };
    size_t index;

    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        qweb_http_parse_status_t actual = parse_http_status(
            (const uint8_t *)cases[index].request,
            strlen(cases[index].request));

        if (actual != cases[index].expected) {
            fprintf(stderr,
                    "invalid HTTP case %zu: expected %s, got %s\n",
                    index,
                    qweb_http_parse_status_name(cases[index].expected),
                    qweb_http_parse_status_name(actual));
            return 1;
        }
    }
    return 0;
}

static int test_http_fixed_limits_and_error_state(void)
{
    uint8_t oversized[QWEB_HTTP_MAX_HEADER_BYTES + 16u];
    const char *prefix = "GET / HTTP/1.1\r\nHost: board\r\nX-Fill: ";
    size_t prefix_length = strlen(prefix);
    qweb_http_parser_t parser;
    qweb_http_feed_result_t result;
    qweb_http_feed_result_t repeated;

    memcpy(oversized, prefix, prefix_length);
    memset(oversized + prefix_length,
           'a',
           sizeof(oversized) - prefix_length);
    qweb_http_parser_init(&parser);
    result = qweb_http_parser_feed(&parser, oversized, sizeof(oversized));
    CHECK(result.status == QWEB_HTTP_PARSE_ERR_HEADERS_TOO_LARGE);
    CHECK(result.consumed == QWEB_HTTP_MAX_HEADER_BYTES);
    repeated = qweb_http_parser_feed(&parser,
                                     (const uint8_t *)"x",
                                     1u);
    CHECK(repeated.status == QWEB_HTTP_PARSE_ERR_STATE);
    CHECK(repeated.consumed == 0u);

    CHECK(qweb_http_parser_feed(NULL, oversized, 1u).status ==
          QWEB_HTTP_PARSE_ERR_NULL);
    qweb_http_parser_init(&parser);
    CHECK(qweb_http_parser_feed(&parser, NULL, 1u).status ==
          QWEB_HTTP_PARSE_ERR_NULL);
    CHECK(qweb_http_parser_feed(&parser, NULL, 0u).status ==
          QWEB_HTTP_PARSE_IN_PROGRESS);
    return 0;
}

static int test_http_rejects_long_method_target_and_content_type(void)
{
    uint8_t request[1024];
    int length;
    char target[QWEB_HTTP_MAX_TARGET_BYTES + 2u];
    char content_type[QWEB_HTTP_MAX_CONTENT_TYPE_BYTES + 2u];

    length = snprintf((char *)request,
                      sizeof(request),
                      "VERYLONGM / HTTP/1.1\r\nHost: a\r\n\r\n");
    CHECK(length > 0);
    CHECK(parse_http_status(request, (size_t)length) ==
          QWEB_HTTP_PARSE_ERR_FIELD_TOO_LARGE);

    target[0] = '/';
    memset(target + 1u, 'x', sizeof(target) - 2u);
    target[sizeof(target) - 1u] = '\0';
    length = snprintf((char *)request,
                      sizeof(request),
                      "GET %s HTTP/1.1\r\nHost: a\r\n\r\n",
                      target);
    CHECK(length > 0);
    CHECK(parse_http_status(request, (size_t)length) ==
          QWEB_HTTP_PARSE_ERR_FIELD_TOO_LARGE);

    memset(content_type, 'x', sizeof(content_type) - 1u);
    content_type[sizeof(content_type) - 1u] = '\0';
    length = snprintf((char *)request,
                      sizeof(request),
                      "GET / HTTP/1.1\r\nHost: a\r\nContent-Type: %s\r\n\r\n",
                      content_type);
    CHECK(length > 0);
    CHECK(parse_http_status(request, (size_t)length) ==
          QWEB_HTTP_PARSE_ERR_FIELD_TOO_LARGE);
    return 0;
}

static int test_json_prompt_unicode_and_escapes(void)
{
    static const uint8_t json[] =
        " { \"max_new_tokens\" : 2, "
        "\"prompt\" : \"A\\n\\u4f60\\ud83d\\ude00\\u0000\" } ";
    static const uint8_t expected[] = {
        'A', '\n', 0xe4u, 0xbdu, 0xa0u, 0xf0u, 0x9fu, 0x98u, 0x80u, 0x00u
    };
    qweb_generate_request_t request;
    qweb_api_parse_result_t result = qweb_api_parse_generate_request(
        json, sizeof(json) - 1u, &request);

    CHECK(result.status == QWEB_API_PARSE_OK);
    CHECK(request.kind == QWEB_GENERATE_INPUT_PROMPT);
    CHECK(request.max_new_tokens == 2u);
    CHECK(request.prompt_length == sizeof(expected));
    CHECK(memcmp(request.prompt, expected, sizeof(expected)) == 0);
    CHECK(request.token_count == 0u);
    return 0;
}

static int test_json_direct_utf8_and_escaped_key(void)
{
    static const uint8_t json[] = {
        '{', '"', 'p', 'r', 'o', '\\', 'u', '0', '0', '6', 'd', 'p', 't', '"',
        ':', '"', 0xe4u, 0xbdu, 0xa0u, '"', ',',
        '"', 'm', 'a', 'x', '_', 'n', 'e', 'w', '_', 't', 'o', 'k', 'e', 'n',
        's', '"', ':', '1', '}'
    };
    static const uint8_t expected[] = {0xe4u, 0xbdu, 0xa0u};
    qweb_generate_request_t request;
    qweb_api_parse_result_t result = qweb_api_parse_generate_request(
        json, sizeof(json), &request);

    CHECK(result.status == QWEB_API_PARSE_OK);
    CHECK(request.kind == QWEB_GENERATE_INPUT_PROMPT);
    CHECK(request.prompt_length == sizeof(expected));
    CHECK(memcmp(request.prompt, expected, sizeof(expected)) == 0);
    return 0;
}

static int test_json_tokens_boundaries_and_member_order(void)
{
    static const uint8_t json[] =
        "{\"tokens\":[0,374,151935],\"max_new_tokens\":256}";
    qweb_generate_request_t request;
    qweb_api_parse_result_t result = qweb_api_parse_generate_request(
        json, sizeof(json) - 1u, &request);

    CHECK(result.status == QWEB_API_PARSE_OK);
    CHECK(request.kind == QWEB_GENERATE_INPUT_TOKENS);
    CHECK(request.max_new_tokens == 256u);
    CHECK(request.token_count == 3u);
    CHECK(request.token_ids[0] == 0u);
    CHECK(request.token_ids[1] == 374u);
    CHECK(request.token_ids[2] == 151935u);
    CHECK(request.prompt_length == 0u);
    return 0;
}

static int build_token_request(char *json,
                               size_t capacity,
                               unsigned token_count)
{
    size_t used = 0u;
    unsigned index;
    int count;

    count = snprintf(json + used, capacity - used, "{\"tokens\":[");
    if (count < 0 || (size_t)count >= capacity - used) return 0;
    used += (size_t)count;
    for (index = 0u; index < token_count; ++index) {
        count = snprintf(json + used,
                         capacity - used,
                         "%s%u",
                         index == 0u ? "" : ",",
                         index);
        if (count < 0 || (size_t)count >= capacity - used) return 0;
        used += (size_t)count;
    }
    count = snprintf(json + used,
                     capacity - used,
                     "],\"max_new_tokens\":1}");
    if (count < 0 || (size_t)count >= capacity - used) return 0;
    return 1;
}

static int test_json_token_capacity_exact_and_overflow(void)
{
    char json[4096];
    qweb_generate_request_t request;
    qweb_api_parse_result_t result;

    CHECK(build_token_request(json, sizeof(json), 256u));
    result = qweb_api_parse_generate_request((const uint8_t *)json,
                                             strlen(json),
                                             &request);
    CHECK(result.status == QWEB_API_PARSE_OK);
    CHECK(request.token_count == 256u);
    CHECK(request.token_ids[255] == 255u);

    CHECK(build_token_request(json, sizeof(json), 257u));
    result = qweb_api_parse_generate_request((const uint8_t *)json,
                                             strlen(json),
                                             &request);
    CHECK(result.status == QWEB_API_PARSE_ERR_CAPACITY);
    return 0;
}

static int expect_json_status(const uint8_t *json,
                              size_t length,
                              qweb_api_parse_status_t expected)
{
    qweb_generate_request_t request;
    qweb_api_parse_result_t result;

    memset(&request, 0xa5, sizeof(request));
    result = qweb_api_parse_generate_request(json, length, &request);
    if (result.status != expected) {
        fprintf(stderr,
                "JSON expected %s, got %s at %zu: %.*s\n",
                qweb_api_parse_status_name(expected),
                qweb_api_parse_status_name(result.status),
                result.error_offset,
                (int)(length > 120u ? 120u : length),
                (const char *)json);
        return 1;
    }
    CHECK(request.kind == QWEB_GENERATE_INPUT_INVALID);
    return 0;
}

static int test_json_rejects_schema_and_number_errors(void)
{
    struct invalid_case {
        const char *json;
        qweb_api_parse_status_t expected;
    } cases[] = {
        {" \r\n\t", QWEB_API_PARSE_ERR_EMPTY},
        {"[]", QWEB_API_PARSE_ERR_SYNTAX},
        {"{}", QWEB_API_PARSE_ERR_MISSING_MEMBER},
        {"{\"prompt\":\"x\"}", QWEB_API_PARSE_ERR_MISSING_MEMBER},
        {"{\"max_new_tokens\":1}", QWEB_API_PARSE_ERR_INPUT_CHOICE},
        {"{\"prompt\":\"x\",\"tokens\":[1],\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_INPUT_CHOICE},
        {"{\"prompt\":\"x\",\"prompt\":\"y\",\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_DUPLICATE_MEMBER},
        {"{\"prompt\":\"x\",\"pro\\u006dpt\":\"y\","
         "\"max_new_tokens\":1}", QWEB_API_PARSE_ERR_DUPLICATE_MEMBER},
        {"{\"prompt\":\"x\",\"max_new_tokens\":1,\"extra\":0}",
         QWEB_API_PARSE_ERR_UNKNOWN_MEMBER},
        {"{\"prompt\":\"\",\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_RANGE},
        {"{\"tokens\":[],\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_RANGE},
        {"{\"tokens\":[151936],\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_RANGE},
        {"{\"tokens\":[1],\"max_new_tokens\":0}",
         QWEB_API_PARSE_ERR_RANGE},
        {"{\"tokens\":[1],\"max_new_tokens\":257}",
         QWEB_API_PARSE_ERR_RANGE},
        {"{\"tokens\":[-1],\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_NUMBER},
        {"{\"tokens\":[01],\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_NUMBER},
        {"{\"tokens\":[1.0],\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_NUMBER},
        {"{\"tokens\":[4294967296],\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_NUMBER},
        {"{\"tokens\":[1,],\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_SYNTAX},
        {"{\"tokens\":[1],\"max_new_tokens\":1,}",
         QWEB_API_PARSE_ERR_SYNTAX},
        {"{\"tokens\":[1],\"max_new_tokens\":1} x",
         QWEB_API_PARSE_ERR_TRAILING},
    };
    size_t index;

    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        if (expect_json_status((const uint8_t *)cases[index].json,
                               strlen(cases[index].json),
                               cases[index].expected) != 0) {
            return 1;
        }
    }
    return 0;
}

static int test_json_rejects_bad_unicode_and_escapes(void)
{
    static const uint8_t invalid_utf8[] = {
        '{', '"', 'p', 'r', 'o', 'm', 'p', 't', '"', ':', '"',
        0xc0u, 0x80u, '"', ',', '"', 'm', 'a', 'x', '_', 'n', 'e', 'w', '_',
        't', 'o', 'k', 'e', 'n', 's', '"', ':', '1', '}'
    };
    static const uint8_t truncated_utf8[] = {
        '{', '"', 'p', 'r', 'o', 'm', 'p', 't', '"', ':', '"', 0xe4u,
        '"', ',', '"', 'm', 'a', 'x', '_', 'n', 'e', 'w', '_', 't', 'o', 'k',
        'e', 'n', 's', '"', ':', '1', '}'
    };
    struct invalid_case {
        const char *json;
        qweb_api_parse_status_t expected;
    } cases[] = {
        {"{\"prompt\":\"\\x\",\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_ESCAPE},
        {"{\"prompt\":\"\\u12xz\",\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_ESCAPE},
        {"{\"prompt\":\"\\ud800\",\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_ESCAPE},
        {"{\"prompt\":\"\\udc00\",\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_ESCAPE},
        {"{\"prompt\":\"\\ud800\\u0041\",\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_ESCAPE},
        {"{\"prompt\":\"line\nraw\",\"max_new_tokens\":1}",
         QWEB_API_PARSE_ERR_SYNTAX},
    };
    size_t index;

    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        if (expect_json_status((const uint8_t *)cases[index].json,
                               strlen(cases[index].json),
                               cases[index].expected) != 0) {
            return 1;
        }
    }
    CHECK(expect_json_status(invalid_utf8,
                             sizeof(invalid_utf8),
                             QWEB_API_PARSE_ERR_UTF8) == 0);
    CHECK(expect_json_status(truncated_utf8,
                             sizeof(truncated_utf8),
                             QWEB_API_PARSE_ERR_UTF8) == 0);
    return 0;
}

static int test_json_prompt_capacity(void)
{
    size_t prompt_length = QWEB_API_MAX_PROMPT_BYTES + 1u;
    size_t prefix_length = strlen("{\"prompt\":\"");
    size_t suffix_length = strlen("\",\"max_new_tokens\":1}");
    size_t json_length = prefix_length + prompt_length + suffix_length;
    uint8_t *json = (uint8_t *)malloc(json_length);
    int result;

    CHECK(json != NULL);
    memcpy(json, "{\"prompt\":\"", prefix_length);
    memset(json + prefix_length, 'a', prompt_length);
    memcpy(json + prefix_length + prompt_length,
           "\",\"max_new_tokens\":1}",
           suffix_length);
    result = expect_json_status(json,
                                json_length,
                                QWEB_API_PARSE_ERR_CAPACITY);
    free(json);
    return result;
}

static int test_json_null_arguments(void)
{
    static const uint8_t json[] =
        "{\"tokens\":[1],\"max_new_tokens\":1}";
    qweb_generate_request_t request;

    CHECK(qweb_api_parse_generate_request(NULL, 0u, &request).status ==
          QWEB_API_PARSE_ERR_NULL);
    CHECK(qweb_api_parse_generate_request(json,
                                          sizeof(json) - 1u,
                                          NULL).status ==
          QWEB_API_PARSE_ERR_NULL);
    return 0;
}

static int test_json_writer_scalars_and_escaping(void)
{
    static const uint8_t string_value[] = {
        'A', '"', '\\', '\n', 0x00u, 0xe4u, 0xbdu, 0xa0u
    };
    static const char expected[] =
        "{\"u\":4294967295,\"i\":-9223372036854775808,"
        "\"s\":\"A\\\"\\\\\\u000a\\u0000\xE4\xBD\xA0\"}";
    uint8_t output[256];
    qweb_json_writer_t writer;

    qweb_json_writer_init(&writer, output, sizeof(output));
    CHECK(qweb_json_writer_append_cstr(&writer, "{\"u\":") ==
          QWEB_JSON_WRITE_OK);
    CHECK(qweb_json_writer_append_u32(&writer, UINT32_MAX) ==
          QWEB_JSON_WRITE_OK);
    CHECK(qweb_json_writer_append_cstr(&writer, ",\"i\":") ==
          QWEB_JSON_WRITE_OK);
    CHECK(qweb_json_writer_append_i64(&writer, INT64_MIN) ==
          QWEB_JSON_WRITE_OK);
    CHECK(qweb_json_writer_append_cstr(&writer, ",\"s\":") ==
          QWEB_JSON_WRITE_OK);
    CHECK(qweb_json_writer_append_string(&writer,
                                         string_value,
                                         sizeof(string_value)) ==
          QWEB_JSON_WRITE_OK);
    CHECK(qweb_json_writer_append_cstr(&writer, "}") == QWEB_JSON_WRITE_OK);
    CHECK(writer.length == sizeof(expected) - 1u);
    CHECK(memcmp(writer.buffer, expected, writer.length) == 0);
    return 0;
}

static int test_json_writer_errors_and_error_body(void)
{
    static const uint8_t invalid_utf8[] = {0xc0u, 0x80u};
    static const char expected[] =
        "{\"error\":{\"code\":\"BAD_REQUEST\","
        "\"message\":\"bad \\\"json\\\"\"}}";
    uint8_t output[128];
    uint8_t small[1];
    qweb_json_writer_t writer;
    size_t output_length = 99u;

    qweb_json_writer_init(&writer, output, sizeof(output));
    CHECK(qweb_json_writer_append_string(&writer,
                                         invalid_utf8,
                                         sizeof(invalid_utf8)) ==
          QWEB_JSON_WRITE_ERR_UTF8);

    qweb_json_writer_init(&writer, small, sizeof(small));
    CHECK(qweb_json_writer_append_cstr(&writer, "ab") ==
          QWEB_JSON_WRITE_ERR_CAPACITY);
    CHECK(writer.length == 0u);
    CHECK(qweb_json_writer_append_cstr(&writer, "x") ==
          QWEB_JSON_WRITE_ERR_CAPACITY);

    qweb_json_writer_init(&writer, NULL, 0u);
    CHECK(writer.status == QWEB_JSON_WRITE_ERR_NULL);
    CHECK(qweb_json_writer_append_bytes(NULL, output, 1u) ==
          QWEB_JSON_WRITE_ERR_NULL);

    CHECK(qweb_api_format_error_json(output,
                                     sizeof(output),
                                     "BAD_REQUEST",
                                     "bad \"json\"",
                                     &output_length) == QWEB_JSON_WRITE_OK);
    CHECK(output_length == sizeof(expected) - 1u);
    CHECK(memcmp(output, expected, output_length) == 0);

    output_length = 99u;
    CHECK(qweb_api_format_error_json(small,
                                     sizeof(small),
                                     "X",
                                     "Y",
                                     &output_length) ==
          QWEB_JSON_WRITE_ERR_CAPACITY);
    CHECK(output_length == 0u);
    return 0;
}

static int test_http_response_formatting(void)
{
    static const uint8_t body[] = "{\"ok\":true}";
    uint8_t response[512];
    size_t response_length;
    qweb_http_format_status_t status;

    status = qweb_http_format_response(202u,
                                       QWEB_HTTP_CONTENT_JSON,
                                       body,
                                       sizeof(body) - 1u,
                                       response,
                                       sizeof(response),
                                       &response_length);
    CHECK(status == QWEB_HTTP_FORMAT_OK);
    CHECK(response_length < sizeof(response));
    response[response_length] = '\0';
    CHECK(strstr((const char *)response,
                 "HTTP/1.1 202 Accepted\r\n") != NULL);
    CHECK(strstr((const char *)response,
                 "Content-Type: application/json; charset=utf-8\r\n") != NULL);
    CHECK(strstr((const char *)response,
                 "Content-Length: 11\r\n") != NULL);
    CHECK(strstr((const char *)response,
                 "Cache-Control: no-store\r\n") != NULL);
    CHECK(strstr((const char *)response,
                 "Connection: close\r\n\r\n{\"ok\":true}") != NULL);

    CHECK(qweb_http_format_response(201u,
                                    QWEB_HTTP_CONTENT_JSON,
                                    body,
                                    sizeof(body) - 1u,
                                    response,
                                    sizeof(response),
                                    &response_length) ==
          QWEB_HTTP_FORMAT_ERR_STATUS);
    CHECK(qweb_http_format_response(200u,
                                    QWEB_HTTP_CONTENT_JSON,
                                    NULL,
                                    1u,
                                    response,
                                    sizeof(response),
                                    &response_length) ==
          QWEB_HTTP_FORMAT_ERR_NULL);
    CHECK(qweb_http_format_response(200u,
                                    QWEB_HTTP_CONTENT_JSON,
                                    body,
                                    sizeof(body) - 1u,
                                    response,
                                    8u,
                                    &response_length) ==
          QWEB_HTTP_FORMAT_ERR_CAPACITY);
    return 0;
}

static int test_end_to_end_fragmented_http_json(void)
{
    static const uint8_t body[] =
        "{\"prompt\":\"The future of FPGA is\",\"max_new_tokens\":2}";
    static const size_t chunks[] = {1u, 2u, 7u, 3u, 19u, 5u, 31u};
    uint8_t request_bytes[1024];
    size_t request_length = make_post_request(body,
                                              sizeof(body) - 1u,
                                              request_bytes,
                                              sizeof(request_bytes));
    qweb_http_parser_t parser;
    qweb_generate_request_t generate;
    qweb_api_parse_result_t api_result;
    size_t offset = 0u;
    size_t chunk_index = 0u;

    CHECK(request_length != 0u);
    qweb_http_parser_init(&parser);
    while (offset < request_length) {
        size_t chunk = chunks[chunk_index %
                              (sizeof(chunks) / sizeof(chunks[0]))];
        qweb_http_feed_result_t result;

        if (chunk > request_length - offset) chunk = request_length - offset;
        result = qweb_http_parser_feed(&parser,
                                       request_bytes + offset,
                                       chunk);
        CHECK(result.consumed == chunk);
        offset += result.consumed;
        ++chunk_index;
    }
    CHECK(parser.status == QWEB_HTTP_PARSE_COMPLETE);
    api_result = qweb_api_parse_generate_request(parser.request.body,
                                                 parser.request.body_length,
                                                 &generate);
    CHECK(api_result.status == QWEB_API_PARSE_OK);
    CHECK(generate.kind == QWEB_GENERATE_INPUT_PROMPT);
    CHECK(generate.max_new_tokens == 2u);
    CHECK(generate.prompt_length == strlen("The future of FPGA is"));
    CHECK(memcmp(generate.prompt,
                 "The future of FPGA is",
                 generate.prompt_length) == 0);
    return 0;
}

int main(void)
{
    run_test("http every two-way split", test_http_every_two_way_split);
    run_test("http one-byte fragments", test_http_one_byte_fragments);
    run_test("http pipeline boundary", test_http_stops_before_pipeline_tail);
    run_test("http GET and header case", test_http_get_and_case_insensitive_headers);
    run_test("http ambiguous headers rejected",
             test_http_rejects_ambiguous_or_invalid_headers);
    run_test("http fixed limits and error state",
             test_http_fixed_limits_and_error_state);
    run_test("http field limits",
             test_http_rejects_long_method_target_and_content_type);
    run_test("json prompt unicode", test_json_prompt_unicode_and_escapes);
    run_test("json direct utf8 and escaped key",
             test_json_direct_utf8_and_escaped_key);
    run_test("json token boundaries",
             test_json_tokens_boundaries_and_member_order);
    run_test("json token capacity", test_json_token_capacity_exact_and_overflow);
    run_test("json schema and number errors",
             test_json_rejects_schema_and_number_errors);
    run_test("json unicode errors", test_json_rejects_bad_unicode_and_escapes);
    run_test("json prompt capacity", test_json_prompt_capacity);
    run_test("json null arguments", test_json_null_arguments);
    run_test("json writer", test_json_writer_scalars_and_escaping);
    run_test("json writer errors", test_json_writer_errors_and_error_body);
    run_test("http response", test_http_response_formatting);
    run_test("fragmented HTTP to JSON", test_end_to_end_fragmented_http_json);

    if (g_tests_failed != 0u) {
        fprintf(stderr,
                "FAIL qweb core host tests: %u/%u failed\n",
                g_tests_failed,
                g_tests_run);
        return 1;
    }
    printf("PASS qweb core host tests: %u tests\n", g_tests_run);
    return 0;
}
