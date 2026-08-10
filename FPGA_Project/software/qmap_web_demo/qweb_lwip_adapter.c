#include "qweb_lwip_adapter.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "lwip/err.h"
#include "lwip/pbuf.h"

#include "qweb_api.h"

static err_t qweb_lwip_accept_callback(
    void *argument,
    struct tcp_pcb *new_pcb,
    err_t error);
static err_t qweb_lwip_recv_callback(
    void *argument,
    struct tcp_pcb *pcb,
    struct pbuf *packet,
    err_t error);
static err_t qweb_lwip_sent_callback(
    void *argument,
    struct tcp_pcb *pcb,
    u16_t acknowledged);
static err_t qweb_lwip_poll_callback(void *argument, struct tcp_pcb *pcb);
static void qweb_lwip_error_callback(void *argument, err_t error);

static unsigned char qweb_lwip_ascii_lower(unsigned char value)
{
    if (value >= (unsigned char)'A' && value <= (unsigned char)'Z') {
        return (unsigned char)(value +
                               ((unsigned char)'a' - (unsigned char)'A'));
    }
    return value;
}

static int qweb_lwip_ascii_equal_casefold(
    const char *left,
    size_t left_length,
    const char *right,
    size_t right_length)
{
    size_t index;

    if (left_length != right_length) return 0;
    for (index = 0u; index < left_length; ++index) {
        if (qweb_lwip_ascii_lower((unsigned char)left[index]) !=
            qweb_lwip_ascii_lower((unsigned char)right[index])) {
            return 0;
        }
    }
    return 1;
}

static int qweb_lwip_host_entry_valid(const char *host, size_t length)
{
    size_t index;

    if (host == NULL || length == 0u ||
        length > QWEB_LWIP_MAX_HOST_BYTES) {
        return 0;
    }
    for (index = 0u; index < length; ++index) {
        unsigned char value = (unsigned char)host[index];
        if (value <= 0x20u || value >= 0x7fu || value == (unsigned char)'/' ||
            value == (unsigned char)'@') {
            return 0;
        }
    }
    return 1;
}

static uint32_t qweb_lwip_now(const qweb_lwip_adapter_t *adapter)
{
    return adapter->now_ms(adapter->now_context);
}

static int qweb_lwip_elapsed(
    uint32_t now,
    uint32_t started,
    uint32_t interval)
{
    return (uint32_t)(now - started) >= interval;
}

static void qweb_lwip_clear_connection_storage(
    qweb_lwip_adapter_t *adapter)
{
    adapter->connection = NULL;
    adapter->connection_state = QWEB_LWIP_CONNECTION_IDLE;
    adapter->accepted_at_ms = 0u;
    adapter->request_started_at_ms = 0u;
    qweb_http_parser_init(&adapter->parser);
    memset(&adapter->route_response, 0, sizeof(adapter->route_response));
    adapter->tx_header_length = 0u;
    adapter->tx_header_offset = 0u;
    adapter->tx_body = NULL;
    adapter->tx_body_length = 0u;
    adapter->tx_body_offset = 0u;
    adapter->tx_queued = 0u;
    adapter->tx_acked = 0u;
    adapter->tx_total = 0u;
}

static void qweb_lwip_abort_connection(qweb_lwip_adapter_t *adapter)
{
    struct tcp_pcb *pcb = adapter->connection;

    if (pcb == NULL) {
        qweb_lwip_clear_connection_storage(adapter);
        return;
    }
    tcp_arg(pcb, NULL);
    tcp_recv(pcb, NULL);
    tcp_sent(pcb, NULL);
    tcp_err(pcb, NULL);
    tcp_poll(pcb, NULL, 0u);
    qweb_lwip_clear_connection_storage(adapter);
    tcp_abort(pcb);
}

static int qweb_lwip_try_close(qweb_lwip_adapter_t *adapter)
{
    err_t error;

    if (adapter->connection == NULL) {
        qweb_lwip_clear_connection_storage(adapter);
        return QWEB_LWIP_OK;
    }
    adapter->connection_state = QWEB_LWIP_CONNECTION_CLOSING;
    error = tcp_close(adapter->connection);
    if (error == ERR_OK) {
        qweb_lwip_clear_connection_storage(adapter);
        return QWEB_LWIP_OK;
    }
    if (error == ERR_MEM) return QWEB_LWIP_OK;
    qweb_lwip_abort_connection(adapter);
    return QWEB_LWIP_ERR_LWIP;
}

static const char *qweb_lwip_reason_phrase(uint16_t status_code)
{
    switch (status_code) {
    case 200u: return "OK";
    case 202u: return "Accepted";
    case 400u: return "Bad Request";
    case 403u: return "Forbidden";
    case 404u: return "Not Found";
    case 405u: return "Method Not Allowed";
    case 408u: return "Request Timeout";
    case 409u: return "Conflict";
    case 411u: return "Length Required";
    case 413u: return "Content Too Large";
    case 415u: return "Unsupported Media Type";
    case 422u: return "Unprocessable Content";
    case 431u: return "Request Header Fields Too Large";
    case 500u: return "Internal Server Error";
    case 503u: return "Service Unavailable";
    default: return NULL;
    }
}

static const char *qweb_lwip_content_type_name(
    qweb_http_content_type_t content_type)
{
    switch (content_type) {
    case QWEB_HTTP_CONTENT_JSON: return "application/json; charset=utf-8";
    case QWEB_HTTP_CONTENT_HTML: return "text/html; charset=utf-8";
    case QWEB_HTTP_CONTENT_JAVASCRIPT:
        return "text/javascript; charset=utf-8";
    case QWEB_HTTP_CONTENT_CSS: return "text/css; charset=utf-8";
    case QWEB_HTTP_CONTENT_TEXT: return "text/plain; charset=utf-8";
    case QWEB_HTTP_CONTENT_OCTETS: return "application/octet-stream";
    default: return NULL;
    }
}

static int qweb_lwip_etag_valid(const char *etag, size_t etag_length)
{
    size_t first_quote;
    size_t index;

    if (etag == NULL || etag_length == 0u ||
        etag_length > QWEB_LWIP_MAX_ETAG_BYTES) {
        return 0;
    }
    first_quote = 0u;
    if (etag_length >= 4u && etag[0] == 'W' && etag[1] == '/') {
        first_quote = 2u;
    }
    if (etag_length < first_quote + 2u || etag[first_quote] != '"' ||
        etag[etag_length - 1u] != '"') {
        return 0;
    }
    for (index = first_quote + 1u; index + 1u < etag_length; ++index) {
        unsigned char value = (unsigned char)etag[index];
        if (value < 0x21u || value > 0x7eu || value == (unsigned char)'"') {
            return 0;
        }
    }
    return 1;
}

static int qweb_lwip_prepare_response_extended(
    qweb_lwip_adapter_t *adapter,
    const qweb_route_response_t *response,
    const char *etag,
    size_t etag_length,
    uint32_t cache_max_age_seconds)
{
    const char *reason;
    const char *content_type;
    int header_length;

    if (response == NULL ||
        (response->body == NULL && response->body_length != 0u)) {
        return QWEB_LWIP_ERR_NULL;
    }
    reason = qweb_lwip_reason_phrase(response->status_code);
    content_type = qweb_lwip_content_type_name(response->content_type);
    if (reason == NULL || content_type == NULL ||
        response->body_length > SIZE_MAX - QWEB_LWIP_TX_HEADER_BYTES) {
        return QWEB_LWIP_ERR_FORMAT;
    }
    if (etag != NULL || etag_length != 0u) {
        if (!qweb_lwip_etag_valid(etag, etag_length) ||
            etag_length > (size_t)INT_MAX) {
            return QWEB_LWIP_ERR_FORMAT;
        }
        header_length = snprintf(
            (char *)adapter->tx_header,
            sizeof(adapter->tx_header),
            "HTTP/1.1 %u %s\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %zu\r\n"
            "ETag: %.*s\r\n"
            "Cache-Control: public, max-age=%u\r\n"
            "Connection: close\r\n"
            "\r\n",
            (unsigned)response->status_code,
            reason,
            content_type,
            response->body_length,
            (int)etag_length,
            etag,
            (unsigned)cache_max_age_seconds);
    } else {
        header_length = snprintf(
            (char *)adapter->tx_header,
            sizeof(adapter->tx_header),
            "HTTP/1.1 %u %s\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %zu\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "\r\n",
            (unsigned)response->status_code,
            reason,
            content_type,
            response->body_length);
    }
    if (header_length < 0 ||
        (size_t)header_length >= sizeof(adapter->tx_header)) {
        return QWEB_LWIP_ERR_FORMAT;
    }

    adapter->route_response = *response;
    adapter->tx_header_length = (size_t)header_length;
    adapter->tx_header_offset = 0u;
    adapter->tx_body = response->body;
    adapter->tx_body_length = response->body_length;
    adapter->tx_body_offset = 0u;
    adapter->tx_queued = 0u;
    adapter->tx_acked = 0u;
    adapter->tx_total = adapter->tx_header_length + response->body_length;
    adapter->connection_state = QWEB_LWIP_CONNECTION_SENDING;
    return QWEB_LWIP_OK;
}

static int qweb_lwip_prepare_response(
    qweb_lwip_adapter_t *adapter,
    const qweb_route_response_t *response)
{
    return qweb_lwip_prepare_response_extended(adapter,
                                               response,
                                               NULL,
                                               0u,
                                               0u);
}

static int qweb_lwip_prepare_error(
    qweb_lwip_adapter_t *adapter,
    uint16_t status_code,
    const char *code,
    const char *message)
{
    qweb_route_response_t response;
    size_t body_length = 0u;
    qweb_json_write_status_t write_status;

    write_status = qweb_api_format_error_json(
        adapter->json_scratch,
        sizeof(adapter->json_scratch),
        code,
        message,
        &body_length);
    if (write_status != QWEB_JSON_WRITE_OK) return QWEB_LWIP_ERR_FORMAT;
    response.status_code = status_code;
    response.content_type = QWEB_HTTP_CONTENT_JSON;
    response.body = adapter->json_scratch;
    response.body_length = body_length;
    return qweb_lwip_prepare_response(adapter, &response);
}

static int qweb_lwip_host_value(
    const qweb_http_parser_t *parser,
    const char **host,
    size_t *host_length)
{
    size_t offset = 0u;
    int first_line = 1;

    if (parser == NULL || host == NULL || host_length == NULL) return 0;
    *host = NULL;
    *host_length = 0u;
    while (offset + 1u < parser->header_length) {
        size_t line_start = offset;
        size_t line_end = offset;
        size_t colon;
        size_t value_start;
        size_t value_end;

        while (line_end + 1u < parser->header_length &&
               !(parser->header_bytes[line_end] == '\r' &&
                 parser->header_bytes[line_end + 1u] == '\n')) {
            ++line_end;
        }
        if (line_end + 1u >= parser->header_length) return 0;
        offset = line_end + 2u;
        if (first_line != 0) {
            first_line = 0;
            continue;
        }
        if (line_end == line_start) break;
        colon = line_start;
        while (colon < line_end && parser->header_bytes[colon] != ':') {
            ++colon;
        }
        if (colon == line_end) return 0;
        if (!qweb_lwip_ascii_equal_casefold(parser->header_bytes + line_start,
                                            colon - line_start,
                                            "Host",
                                            4u)) {
            continue;
        }
        value_start = colon + 1u;
        while (value_start < line_end &&
               (parser->header_bytes[value_start] == ' ' ||
                parser->header_bytes[value_start] == '\t')) {
            ++value_start;
        }
        value_end = line_end;
        while (value_end > value_start &&
               (parser->header_bytes[value_end - 1u] == ' ' ||
                parser->header_bytes[value_end - 1u] == '\t')) {
            --value_end;
        }
        if (*host != NULL || value_end == value_start) return 0;
        *host = parser->header_bytes + value_start;
        *host_length = value_end - value_start;
    }
    return *host != NULL;
}

static int qweb_lwip_host_allowed(const qweb_lwip_adapter_t *adapter)
{
    const char *host;
    size_t host_length;
    size_t index;

    if (!qweb_lwip_host_value(&adapter->parser, &host, &host_length)) {
        return 0;
    }
    for (index = 0u; index < adapter->allowed_host_count; ++index) {
        if (qweb_lwip_ascii_equal_casefold(
                host,
                host_length,
                adapter->allowed_hosts[index],
                adapter->allowed_host_lengths[index])) {
            return 1;
        }
    }
    return 0;
}

static uint16_t qweb_lwip_parse_error_status(
    qweb_http_parse_status_t status)
{
    if (status == QWEB_HTTP_PARSE_ERR_HEADERS_TOO_LARGE ||
        status == QWEB_HTTP_PARSE_ERR_FIELD_TOO_LARGE) {
        return 431u;
    }
    if (status == QWEB_HTTP_PARSE_ERR_BODY_TOO_LARGE) return 413u;
    if (status == QWEB_HTTP_PARSE_ERR_CONTENT_LENGTH) return 411u;
    return 400u;
}

static int qweb_lwip_write_fragment(
    qweb_lwip_adapter_t *adapter,
    const uint8_t *bytes,
    size_t remaining,
    size_t *offset)
{
    u16_t available;
    size_t chunk;
    err_t error;

    available = tcp_sndbuf(adapter->connection);
    if (available == 0u || remaining == 0u) return QWEB_LWIP_OK;
    chunk = remaining;
    if (chunk > (size_t)available) chunk = (size_t)available;
    if (chunk > (size_t)UINT16_MAX) chunk = (size_t)UINT16_MAX;
    error = tcp_write(adapter->connection,
                      bytes + *offset,
                      (u16_t)chunk,
                      TCP_WRITE_FLAG_COPY);
    if (error == ERR_MEM) return QWEB_LWIP_OK;
    if (error != ERR_OK) return QWEB_LWIP_ERR_LWIP;
    *offset += chunk;
    adapter->tx_queued += chunk;
    return QWEB_LWIP_OK;
}

static int qweb_lwip_flush(qweb_lwip_adapter_t *adapter)
{
    size_t before;
    err_t output_error;
    int rc;

    if (adapter->connection == NULL ||
        adapter->connection_state != QWEB_LWIP_CONNECTION_SENDING) {
        return QWEB_LWIP_OK;
    }

    do {
        before = adapter->tx_queued;
        if (adapter->tx_header_offset < adapter->tx_header_length) {
            rc = qweb_lwip_write_fragment(
                adapter,
                adapter->tx_header,
                adapter->tx_header_length - adapter->tx_header_offset,
                &adapter->tx_header_offset);
            if (rc != QWEB_LWIP_OK) return rc;
        } else if (adapter->tx_body_offset < adapter->tx_body_length) {
            rc = qweb_lwip_write_fragment(
                adapter,
                adapter->tx_body,
                adapter->tx_body_length - adapter->tx_body_offset,
                &adapter->tx_body_offset);
            if (rc != QWEB_LWIP_OK) return rc;
        }
    } while (adapter->tx_queued != before &&
             tcp_sndbuf(adapter->connection) != 0u);

    output_error = tcp_output(adapter->connection);
    if (output_error != ERR_OK && output_error != ERR_MEM) {
        return QWEB_LWIP_ERR_LWIP;
    }
    if (adapter->tx_total == 0u || adapter->tx_acked == adapter->tx_total) {
        return qweb_lwip_try_close(adapter);
    }
    return QWEB_LWIP_OK;
}

static int qweb_lwip_route_complete_request(qweb_lwip_adapter_t *adapter)
{
    qweb_lwip_static_asset_t asset;
    int asset_result = QWEB_LWIP_ASSET_MISS;
    int rc;

    if (!qweb_lwip_host_allowed(adapter)) {
        return qweb_lwip_prepare_error(adapter,
                                       403u,
                                       "host_not_allowed",
                                       "HTTP Host is not allowed");
    }
    if (adapter->resolve_asset != NULL) {
        memset(&asset, 0, sizeof(asset));
        asset_result = adapter->resolve_asset(
            adapter->asset_context,
            adapter->parser.request.target,
            adapter->parser.request.target_length,
            &asset);
        if (asset_result == QWEB_LWIP_ASSET_HIT) {
            qweb_route_response_t response;

            if (asset.body == NULL && asset.body_length != 0u) {
                return qweb_lwip_prepare_error(adapter,
                                               500u,
                                               "asset_failure",
                                               "static asset is invalid");
            }
            if (qweb_lwip_content_type_name(asset.content_type) == NULL) {
                return qweb_lwip_prepare_error(
                    adapter,
                    500u,
                    "asset_failure",
                    "static asset content type is invalid");
            }
            if (!qweb_lwip_etag_valid(asset.etag, asset.etag_length)) {
                return qweb_lwip_prepare_error(adapter,
                                               500u,
                                               "asset_failure",
                                               "static asset ETag is invalid");
            }
            if (adapter->parser.request.method_length != 3u ||
                memcmp(adapter->parser.request.method, "GET", 3u) != 0 ||
                adapter->parser.request.content_length != 0u ||
                adapter->parser.request.body_length != 0u) {
                return qweb_lwip_prepare_error(
                    adapter,
                    405u,
                    "static_method_not_allowed",
                    "static assets require GET with an empty body");
            }
            response.status_code = 200u;
            response.content_type = asset.content_type;
            response.body = asset.body;
            response.body_length = asset.body_length;
            return qweb_lwip_prepare_response_extended(
                adapter,
                &response,
                asset.etag,
                asset.etag_length,
                asset.cache_max_age_seconds);
        }
        if (asset_result != QWEB_LWIP_ASSET_MISS) {
            return qweb_lwip_prepare_error(adapter,
                                           500u,
                                           "asset_failure",
                                           "static asset lookup failed");
        }
    }
    rc = qweb_router_handle(adapter->router,
                            &adapter->parser.request,
                            adapter->json_scratch,
                            sizeof(adapter->json_scratch),
                            &adapter->route_response);
    if (rc != QWEB_ROUTER_OK) {
        return qweb_lwip_prepare_error(adapter,
                                       500u,
                                       "router_failure",
                                       "request routing failed");
    }
    return qweb_lwip_prepare_response(adapter, &adapter->route_response);
}

static err_t qweb_lwip_accept_callback(
    void *argument,
    struct tcp_pcb *new_pcb,
    err_t error)
{
    qweb_lwip_adapter_t *adapter = (qweb_lwip_adapter_t *)argument;
    uint32_t now;

    if (adapter == NULL || new_pcb == NULL || error != ERR_OK ||
        adapter->initialized == 0u) {
        if (new_pcb != NULL) tcp_abort(new_pcb);
        return ERR_ABRT;
    }
    if (adapter->connection != NULL) {
        tcp_abort(new_pcb);
        return ERR_ABRT;
    }

    qweb_lwip_clear_connection_storage(adapter);
    now = qweb_lwip_now(adapter);
    adapter->connection = new_pcb;
    adapter->connection_state = QWEB_LWIP_CONNECTION_RECEIVING;
    adapter->accepted_at_ms = now;
    adapter->request_started_at_ms = now;
    tcp_arg(new_pcb, adapter);
    tcp_recv(new_pcb, qweb_lwip_recv_callback);
    tcp_sent(new_pcb, qweb_lwip_sent_callback);
    tcp_err(new_pcb, qweb_lwip_error_callback);
    tcp_poll(new_pcb,
             qweb_lwip_poll_callback,
             adapter->tcp_poll_interval);
    return ERR_OK;
}

static err_t qweb_lwip_recv_callback(
    void *argument,
    struct tcp_pcb *pcb,
    struct pbuf *packet,
    err_t error)
{
    qweb_lwip_adapter_t *adapter = (qweb_lwip_adapter_t *)argument;
    struct pbuf *fragment;
    u16_t received_length;
    qweb_http_parse_status_t parse_status = QWEB_HTTP_PARSE_IN_PROGRESS;
    int trailing_bytes = 0;
    int rc;

    if (adapter == NULL || pcb == NULL || adapter->connection != pcb) {
        if (packet != NULL) (void)pbuf_free(packet);
        if (pcb != NULL) tcp_abort(pcb);
        return ERR_ABRT;
    }
    if (error != ERR_OK) {
        if (packet != NULL) (void)pbuf_free(packet);
        qweb_lwip_abort_connection(adapter);
        return ERR_ABRT;
    }
    if (packet == NULL) {
        rc = qweb_lwip_try_close(adapter);
        return rc == QWEB_LWIP_OK ? ERR_OK : ERR_ABRT;
    }
    received_length = packet->tot_len;
    if (adapter->connection_state != QWEB_LWIP_CONNECTION_RECEIVING) {
        tcp_recved(pcb, received_length);
        (void)pbuf_free(packet);
        qweb_lwip_abort_connection(adapter);
        return ERR_ABRT;
    }

    for (fragment = packet; fragment != NULL; fragment = fragment->next) {
        qweb_http_feed_result_t feed;

        if (fragment->len == 0u) continue;
        if (parse_status == QWEB_HTTP_PARSE_COMPLETE) {
            trailing_bytes = 1;
            break;
        }
        feed = qweb_http_parser_feed(&adapter->parser,
                                     (const uint8_t *)fragment->payload,
                                     (size_t)fragment->len);
        parse_status = feed.status;
        if (feed.status < 0) break;
        if (feed.status == QWEB_HTTP_PARSE_COMPLETE &&
            feed.consumed != (size_t)fragment->len) {
            trailing_bytes = 1;
            break;
        }
    }
    tcp_recved(pcb, received_length);
    (void)pbuf_free(packet);

    if (parse_status < 0) {
        rc = qweb_lwip_prepare_error(
            adapter,
            qweb_lwip_parse_error_status(parse_status),
            "invalid_http_request",
            qweb_http_parse_status_name(parse_status));
    } else if (trailing_bytes != 0) {
        rc = qweb_lwip_prepare_error(adapter,
                                     400u,
                                     "http_pipelining_not_supported",
                                     "only one request per connection is allowed");
    } else if (parse_status == QWEB_HTTP_PARSE_COMPLETE) {
        rc = qweb_lwip_route_complete_request(adapter);
    } else {
        return ERR_OK;
    }

    if (rc != QWEB_LWIP_OK || qweb_lwip_flush(adapter) != QWEB_LWIP_OK) {
        qweb_lwip_abort_connection(adapter);
        return ERR_ABRT;
    }
    return ERR_OK;
}

static err_t qweb_lwip_sent_callback(
    void *argument,
    struct tcp_pcb *pcb,
    u16_t acknowledged)
{
    qweb_lwip_adapter_t *adapter = (qweb_lwip_adapter_t *)argument;

    if (adapter == NULL || pcb == NULL || adapter->connection != pcb ||
        adapter->connection_state != QWEB_LWIP_CONNECTION_SENDING ||
        adapter->tx_acked > adapter->tx_queued ||
        (size_t)acknowledged > adapter->tx_queued - adapter->tx_acked) {
        if (adapter != NULL && adapter->connection == pcb) {
            qweb_lwip_abort_connection(adapter);
        } else if (pcb != NULL) {
            tcp_abort(pcb);
        }
        return ERR_ABRT;
    }
    adapter->tx_acked += (size_t)acknowledged;
    if (adapter->tx_acked == adapter->tx_total &&
        adapter->tx_queued == adapter->tx_total) {
        return qweb_lwip_try_close(adapter) == QWEB_LWIP_OK ? ERR_OK
                                                            : ERR_ABRT;
    }
    if (qweb_lwip_flush(adapter) != QWEB_LWIP_OK) {
        qweb_lwip_abort_connection(adapter);
        return ERR_ABRT;
    }
    return ERR_OK;
}

static err_t qweb_lwip_poll_callback(void *argument, struct tcp_pcb *pcb)
{
    qweb_lwip_adapter_t *adapter = (qweb_lwip_adapter_t *)argument;

    if (adapter == NULL || pcb == NULL || adapter->connection != pcb) {
        if (pcb != NULL) tcp_abort(pcb);
        return ERR_ABRT;
    }
    if (qweb_lwip_adapter_service(adapter) != QWEB_LWIP_OK) {
        if (adapter->connection == pcb) qweb_lwip_abort_connection(adapter);
        return ERR_ABRT;
    }
    return ERR_OK;
}

static void qweb_lwip_error_callback(void *argument, err_t error)
{
    qweb_lwip_adapter_t *adapter = (qweb_lwip_adapter_t *)argument;

    (void)error;
    if (adapter != NULL) qweb_lwip_clear_connection_storage(adapter);
}

int qweb_lwip_adapter_init(
    qweb_lwip_adapter_t *adapter,
    qweb_router_t *router,
    const qweb_lwip_config_t *config)
{
    size_t index;

    if (adapter == NULL || router == NULL || config == NULL) {
        return QWEB_LWIP_ERR_NULL;
    }
    if (config->listen_port == 0u || config->now_ms == NULL ||
        config->request_timeout_ms == 0u ||
        config->connection_timeout_ms <= config->request_timeout_ms ||
        config->request_timeout_ms > (uint32_t)INT32_MAX ||
        config->connection_timeout_ms > (uint32_t)INT32_MAX ||
        config->allowed_host_count == 0u ||
        config->allowed_host_count > QWEB_LWIP_MAX_ALLOWED_HOSTS) {
        return QWEB_LWIP_ERR_CONFIG;
    }

    memset(adapter, 0, sizeof(*adapter));
    adapter->router = router;
    adapter->now_ms = config->now_ms;
    adapter->now_context = config->now_context;
    adapter->resolve_asset = config->resolve_asset;
    adapter->asset_context = config->asset_context;
    adapter->listen_port = config->listen_port;
    adapter->tcp_poll_interval = config->tcp_poll_interval != 0u
                                     ? config->tcp_poll_interval
                                     : QWEB_LWIP_DEFAULT_POLL_INTERVAL;
    adapter->request_timeout_ms = config->request_timeout_ms;
    adapter->connection_timeout_ms = config->connection_timeout_ms;
    adapter->allowed_host_count = config->allowed_host_count;
    for (index = 0u; index < config->allowed_host_count; ++index) {
        size_t length;

        if (config->allowed_hosts[index] == NULL) {
            memset(adapter, 0, sizeof(*adapter));
            return QWEB_LWIP_ERR_CONFIG;
        }
        length = strlen(config->allowed_hosts[index]);
        if (!qweb_lwip_host_entry_valid(config->allowed_hosts[index], length)) {
            memset(adapter, 0, sizeof(*adapter));
            return QWEB_LWIP_ERR_CONFIG;
        }
        memcpy(adapter->allowed_hosts[index],
               config->allowed_hosts[index],
               length + 1u);
        adapter->allowed_host_lengths[index] = length;
    }
    qweb_http_parser_init(&adapter->parser);
    adapter->initialized = 1u;
    return QWEB_LWIP_OK;
}

int qweb_lwip_adapter_start(qweb_lwip_adapter_t *adapter)
{
    struct tcp_pcb *listener;
    struct tcp_pcb *listening;
    err_t error;

    if (adapter == NULL) return QWEB_LWIP_ERR_NULL;
    if (adapter->initialized == 0u || adapter->router == NULL ||
        adapter->listener != NULL) {
        return QWEB_LWIP_ERR_STATE;
    }
    listener = tcp_new_ip_type(IPADDR_TYPE_ANY);
    if (listener == NULL) return QWEB_LWIP_ERR_LWIP;
    error = tcp_bind(listener, IP_ANY_TYPE, adapter->listen_port);
    if (error != ERR_OK) {
        if (tcp_close(listener) != ERR_OK) tcp_abort(listener);
        return QWEB_LWIP_ERR_LWIP;
    }
    listening = tcp_listen_with_backlog_and_err(listener, 1u, &error);
    if (listening == NULL || error != ERR_OK) {
        if (listening != NULL) {
            if (tcp_close(listening) != ERR_OK) tcp_abort(listening);
        } else {
            /* lwIP retains the original PCB on a failed listen conversion. */
            if (tcp_close(listener) != ERR_OK) tcp_abort(listener);
        }
        return QWEB_LWIP_ERR_LWIP;
    }
    adapter->listener = listening;
    tcp_arg(listening, adapter);
    tcp_accept(listening, qweb_lwip_accept_callback);
    return QWEB_LWIP_OK;
}

void qweb_lwip_adapter_stop(qweb_lwip_adapter_t *adapter)
{
    struct tcp_pcb *listener;

    if (adapter == NULL) return;
    if (adapter->connection != NULL) qweb_lwip_abort_connection(adapter);
    listener = adapter->listener;
    adapter->listener = NULL;
    if (listener != NULL) {
        tcp_arg(listener, NULL);
        tcp_accept(listener, NULL);
        if (tcp_close(listener) != ERR_OK) tcp_abort(listener);
    }
}

int qweb_lwip_adapter_service(qweb_lwip_adapter_t *adapter)
{
    uint32_t now;
    int rc;

    if (adapter == NULL) return QWEB_LWIP_ERR_NULL;
    if (adapter->initialized == 0u) return QWEB_LWIP_ERR_STATE;
    if (adapter->connection == NULL) return QWEB_LWIP_OK;

    now = qweb_lwip_now(adapter);
    if (adapter->connection_state == QWEB_LWIP_CONNECTION_RECEIVING &&
        qweb_lwip_elapsed(now,
                          adapter->request_started_at_ms,
                          adapter->request_timeout_ms)) {
        rc = qweb_lwip_prepare_error(adapter,
                                     408u,
                                     "request_timeout",
                                     "HTTP request deadline expired");
        if (rc != QWEB_LWIP_OK) {
            qweb_lwip_abort_connection(adapter);
            return rc;
        }
    }
    if (adapter->connection != NULL &&
        qweb_lwip_elapsed(now,
                          adapter->accepted_at_ms,
                          adapter->connection_timeout_ms)) {
        qweb_lwip_abort_connection(adapter);
        return QWEB_LWIP_OK;
    }
    if (adapter->connection_state == QWEB_LWIP_CONNECTION_SENDING) {
        rc = qweb_lwip_flush(adapter);
        if (rc != QWEB_LWIP_OK) {
            qweb_lwip_abort_connection(adapter);
            return rc;
        }
    } else if (adapter->connection_state == QWEB_LWIP_CONNECTION_CLOSING) {
        return qweb_lwip_try_close(adapter);
    }
    return QWEB_LWIP_OK;
}

int qweb_lwip_adapter_connection_active(
    const qweb_lwip_adapter_t *adapter)
{
    return adapter != NULL && adapter->connection != NULL;
}
