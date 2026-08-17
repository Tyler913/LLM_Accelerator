#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "netif/xadapter.h"

#include "qweb_board_app.h"

#define TEST_TX_CAPACITY 65536u
#define TEST_REQUEST_CAPACITY 16384u
#define TEST_MMIO_WORDS 256u

static qweb_lwip_adapter_t test_adapter;
static qweb_router_t test_router;
static qweb_job_t test_job;
static qweb_board_app_t test_board_app;
static struct tcp_pcb test_listener_pcb;
static struct tcp_pcb test_connection_pcb;
static struct tcp_pcb test_second_pcb;
static uint8_t test_tx_bytes[TEST_TX_CAPACITY];
static size_t test_tx_length;
static uint8_t test_request_bytes[TEST_REQUEST_CAPACITY];
static uint32_t test_now_ms;
static unsigned test_route_calls;
static unsigned test_job_step_calls;
static unsigned test_network_pump_calls;
static unsigned test_pbuf_free_calls;
static unsigned test_recved_bytes;
static unsigned test_tcp_output_calls;
static unsigned test_fast_timer_calls;
static unsigned test_slow_timer_calls;
static unsigned test_xemac_input_calls;
static int test_job_active;
static err_t test_forced_write_error;
static err_t test_forced_output_error;
static err_t test_forced_bind_error;
static err_t test_forced_listen_error;
static err_t test_forced_close_error;
static unsigned test_close_calls;
static struct tcp_pcb *test_last_close_pcb;
static size_t test_close_tx_queued;
static size_t test_close_tx_acked;
static size_t test_close_tx_total;
static qweb_lwip_connection_state_t test_close_connection_state;
static u8_t test_close_priority;
static int test_close_callbacks_detached;
static uint32_t test_mmio[TEST_MMIO_WORDS];
static qot_layer_qmap_bases_t test_layer_tables[QOT_MAX_LAYERS];
static unsigned test_runner_pumps_after_start;
static uint32_t test_runner_done_mask;

volatile int TcpFastTmrFlag;
volatile int TcpSlowTmrFlag;

void tcp_fasttmr(void);
void tcp_slowtmr(void);

static void test_fail(const char *expression, const char *file, int line)
{
    (void)fprintf(stderr,
                  "FAIL %s:%d: %s\n",
                  file,
                  line,
                  expression);
    exit(EXIT_FAILURE);
}

#define TEST_CHECK(expression)                                               \
    do {                                                                     \
        if (!(expression)) test_fail(#expression, __FILE__, __LINE__);       \
    } while (0)

static int test_bytes_contain(
    const uint8_t *haystack,
    size_t haystack_length,
    const char *needle)
{
    size_t needle_length = strlen(needle);
    size_t offset;

    if (needle_length == 0u) return 1;
    if (needle_length > haystack_length) return 0;
    for (offset = 0u; offset <= haystack_length - needle_length; ++offset) {
        if (memcmp(haystack + offset, needle, needle_length) == 0) return 1;
    }
    return 0;
}

static size_t test_http_body_offset(void)
{
    size_t offset;

    for (offset = 0u; offset + 3u < test_tx_length; ++offset) {
        if (memcmp(test_tx_bytes + offset, "\r\n\r\n", 4u) == 0) {
            return offset + 4u;
        }
    }
    return SIZE_MAX;
}

static uint32_t test_clock(void *context)
{
    TEST_CHECK(context == &test_now_ms);
    return test_now_ms;
}

static int test_asset_resolver(
    void *context,
    const char *target,
    size_t target_length,
    qweb_lwip_static_asset_t *asset)
{
    static const uint8_t index_body[] = "<html>qweb</html>";
    static const char index_etag[] = "\"qweb-index-test\"";

    TEST_CHECK(context == &test_router);
    TEST_CHECK(target != NULL);
    TEST_CHECK(asset != NULL);
    if (target_length == 1u && memcmp(target, "/", 1u) == 0) {
        asset->content_type = QWEB_HTTP_CONTENT_HTML;
        asset->body = index_body;
        asset->body_length = sizeof(index_body) - 1u;
        asset->etag = index_etag;
        asset->etag_length = sizeof(index_etag) - 1u;
        asset->cache_max_age_seconds = 60u;
        return QWEB_LWIP_ASSET_HIT;
    }
    if (target_length == 10u && memcmp(target, "/asset-err", 10u) == 0) {
        return QWEB_LWIP_ASSET_ERROR;
    }
    return QWEB_LWIP_ASSET_MISS;
}

int qweb_router_handle(
    qweb_router_t *router,
    const qweb_http_request_t *request,
    uint8_t *json_scratch,
    size_t json_scratch_capacity,
    qweb_route_response_t *response)
{
    static const uint8_t accepted[] = "{\"job_id\":1,\"state\":\"queued\"}";
    static const uint8_t health[] = "{\"ready\":true}";
    static const uint8_t missing[] = "{\"error\":\"missing\"}";
    static const uint8_t raw[] = {0x00u, 0x41u, 0xffu, 0x0au};

    (void)json_scratch;
    (void)json_scratch_capacity;
    TEST_CHECK(router == &test_router);
    TEST_CHECK(request != NULL);
    TEST_CHECK(response != NULL);
    ++test_route_calls;
    if (request->target_length == 13u &&
        memcmp(request->target, "/router-error", 13u) == 0) {
        return QWEB_ROUTER_ERR_STATE;
    }
    if (request->target_length == 13u &&
        memcmp(request->target, "/api/generate", 13u) == 0) {
        test_job_active = 1;
        response->status_code = 202u;
        response->content_type = QWEB_HTTP_CONTENT_JSON;
        response->body = accepted;
        response->body_length = sizeof(accepted) - 1u;
    } else if (request->target_length == 22u &&
               memcmp(request->target,
                      "/api/generate/1/output",
                      22u) == 0) {
        response->status_code = 200u;
        response->content_type = QWEB_HTTP_CONTENT_OCTETS;
        response->body = raw;
        response->body_length = sizeof(raw);
    } else if (request->target_length == 11u &&
               memcmp(request->target, "/api/health", 11u) == 0) {
        response->status_code = 200u;
        response->content_type = QWEB_HTTP_CONTENT_JSON;
        response->body = health;
        response->body_length = sizeof(health) - 1u;
    } else {
        response->status_code = 404u;
        response->content_type = QWEB_HTTP_CONTENT_JSON;
        response->body = missing;
        response->body_length = sizeof(missing) - 1u;
    }
    return QWEB_ROUTER_OK;
}

int qweb_job_is_active(const qweb_job_t *job)
{
    TEST_CHECK(job == &test_job);
    return test_job_active;
}

int qweb_job_step(qweb_job_t *job, qot_session_event_t *event)
{
    TEST_CHECK(job == &test_job);
    TEST_CHECK(event != NULL);
    ++test_job_step_calls;
    test_job_active = 0;
    memset(event, 0, sizeof(*event));
    event->kind = QOT_SESSION_EVENT_DONE;
    return QWEB_JOB_OK;
}

struct tcp_pcb *tcp_new_ip_type(u8_t type)
{
    TEST_CHECK(type == IPADDR_TYPE_ANY);
    memset(&test_listener_pcb, 0, sizeof(test_listener_pcb));
    test_listener_pcb.send_buffer = UINT16_MAX;
    test_listener_pcb.prio = TCP_PRIO_NORMAL;
    return &test_listener_pcb;
}

void tcp_arg(struct tcp_pcb *pcb, void *argument)
{
    TEST_CHECK(pcb != NULL);
    pcb->callback_argument = argument;
}

void tcp_recv(struct tcp_pcb *pcb, tcp_recv_fn callback)
{
    TEST_CHECK(pcb != NULL);
    pcb->recv_callback = callback;
}

void tcp_sent(struct tcp_pcb *pcb, tcp_sent_fn callback)
{
    TEST_CHECK(pcb != NULL);
    pcb->sent_callback = callback;
}

void tcp_err(struct tcp_pcb *pcb, tcp_err_fn callback)
{
    TEST_CHECK(pcb != NULL);
    pcb->error_callback = callback;
}

void tcp_accept(struct tcp_pcb *pcb, tcp_accept_fn callback)
{
    TEST_CHECK(pcb != NULL);
    pcb->accept_callback = callback;
}

void tcp_poll(struct tcp_pcb *pcb, tcp_poll_fn callback, u8_t interval)
{
    TEST_CHECK(pcb != NULL);
    pcb->poll_callback = callback;
    pcb->poll_interval = interval;
}

void tcp_setprio(struct tcp_pcb *pcb, u8_t priority)
{
    TEST_CHECK(pcb != NULL);
    pcb->prio = priority;
}

u16_t test_tcp_sndbuf(struct tcp_pcb *pcb)
{
    TEST_CHECK(pcb != NULL);
    return pcb->send_buffer;
}

void tcp_recved(struct tcp_pcb *pcb, u16_t length)
{
    TEST_CHECK(pcb != NULL);
    test_recved_bytes += (unsigned)length;
}

err_t tcp_bind(struct tcp_pcb *pcb, const void *address, u16_t port)
{
    TEST_CHECK(pcb != NULL);
    TEST_CHECK(address == IP_ANY_TYPE);
    pcb->bound_port = port;
    return test_forced_bind_error;
}

struct tcp_pcb *tcp_listen_with_backlog_and_err(
    struct tcp_pcb *pcb,
    u8_t backlog,
    err_t *error)
{
    TEST_CHECK(pcb != NULL);
    TEST_CHECK(backlog == 1u);
    TEST_CHECK(error != NULL);
    *error = test_forced_listen_error;
    if (*error != ERR_OK) return NULL;
    pcb->listening = 1u;
    return pcb;
}

void tcp_abort(struct tcp_pcb *pcb)
{
    TEST_CHECK(pcb != NULL);
    pcb->aborted = 1u;
}

err_t tcp_close(struct tcp_pcb *pcb)
{
    TEST_CHECK(pcb != NULL);
    ++test_close_calls;
    test_last_close_pcb = pcb;
    test_close_tx_queued = test_adapter.tx_queued;
    test_close_tx_acked = test_adapter.tx_acked;
    test_close_tx_total = test_adapter.tx_total;
    test_close_connection_state = test_adapter.connection_state;
    test_close_priority = pcb->prio;
    test_close_callbacks_detached =
        pcb->callback_argument == NULL &&
        pcb->recv_callback == NULL &&
        pcb->sent_callback == NULL &&
        pcb->error_callback == NULL &&
        pcb->poll_callback == NULL &&
        pcb->poll_interval == 0u;
    if (test_forced_close_error != ERR_OK) {
        err_t error = test_forced_close_error;
        test_forced_close_error = ERR_OK;
        return error;
    }
    pcb->closed = 1u;
    return ERR_OK;
}

err_t tcp_write(
    struct tcp_pcb *pcb,
    const void *bytes,
    u16_t length,
    u8_t flags)
{
    TEST_CHECK(pcb != NULL);
    TEST_CHECK(bytes != NULL || length == 0u);
    TEST_CHECK(flags == TCP_WRITE_FLAG_COPY);
    if (test_forced_write_error != ERR_OK) {
        err_t error = test_forced_write_error;
        test_forced_write_error = ERR_OK;
        return error;
    }
    TEST_CHECK(length <= pcb->send_buffer);
    TEST_CHECK(test_tx_length + (size_t)length <= TEST_TX_CAPACITY);
    memcpy(test_tx_bytes + test_tx_length, bytes, (size_t)length);
    test_tx_length += (size_t)length;
    pcb->send_buffer = (u16_t)(pcb->send_buffer - length);
    return ERR_OK;
}

err_t tcp_output(struct tcp_pcb *pcb)
{
    TEST_CHECK(pcb != NULL);
    ++test_tcp_output_calls;
    if (test_forced_output_error != ERR_OK) {
        err_t error = test_forced_output_error;
        test_forced_output_error = ERR_OK;
        return error;
    }
    return ERR_OK;
}

u8_t pbuf_free(struct pbuf *packet)
{
    TEST_CHECK(packet != NULL);
    ++test_pbuf_free_calls;
    return 1u;
}

void tcp_fasttmr(void)
{
    ++test_fast_timer_calls;
}

void tcp_slowtmr(void)
{
    ++test_slow_timer_calls;
}

int xemacif_input(struct netif *network_interface)
{
    TEST_CHECK(network_interface != NULL);
    ++test_xemac_input_calls;
    return 0;
}

static void test_generic_network_pump(void *context)
{
    uint32_t control;

    TEST_CHECK(context == &test_board_app);
    ++test_network_pump_calls;
    control = test_mmio[QOT_REG_CTRL / sizeof(test_mmio[0])];
    if ((control & QOT_CTRL_START_MASK) != 0u) {
        ++test_runner_pumps_after_start;
        if (test_runner_pumps_after_start >= 1u) {
            test_mmio[QOT_REG_STATUS / sizeof(test_mmio[0])] =
                QOT_STATUS_DONE_STICKY_MASK;
            test_mmio[QOT_REG_OUT_TOKEN / sizeof(test_mmio[0])] = 264u;
            test_mmio[QOT_REG_LAYERS / sizeof(test_mmio[0])] =
                28u | (28u << 16);
            test_mmio[QOT_REG_LAYER_DONE_MASK / sizeof(test_mmio[0])] =
                test_runner_done_mask;
            test_mmio[QOT_REG_LAYER_ERROR_MASK / sizeof(test_mmio[0])] = 0u;
        }
    }
}

static void test_reset_fakes(void)
{
    memset(&test_adapter, 0, sizeof(test_adapter));
    memset(&test_router, 0, sizeof(test_router));
    memset(&test_job, 0, sizeof(test_job));
    memset(&test_board_app, 0, sizeof(test_board_app));
    memset(&test_listener_pcb, 0, sizeof(test_listener_pcb));
    memset(&test_connection_pcb, 0, sizeof(test_connection_pcb));
    memset(&test_second_pcb, 0, sizeof(test_second_pcb));
    memset(test_tx_bytes, 0, sizeof(test_tx_bytes));
    memset(test_request_bytes, 0, sizeof(test_request_bytes));
    memset(test_mmio, 0, sizeof(test_mmio));
    memset(test_layer_tables, 0, sizeof(test_layer_tables));
    test_tx_length = 0u;
    test_now_ms = 100u;
    test_route_calls = 0u;
    test_job_step_calls = 0u;
    test_network_pump_calls = 0u;
    test_pbuf_free_calls = 0u;
    test_recved_bytes = 0u;
    test_tcp_output_calls = 0u;
    test_fast_timer_calls = 0u;
    test_slow_timer_calls = 0u;
    test_xemac_input_calls = 0u;
    test_job_active = 0;
    test_forced_write_error = ERR_OK;
    test_forced_output_error = ERR_OK;
    test_forced_bind_error = ERR_OK;
    test_forced_listen_error = ERR_OK;
    test_forced_close_error = ERR_OK;
    test_close_calls = 0u;
    test_last_close_pcb = NULL;
    test_close_tx_queued = 0u;
    test_close_tx_acked = 0u;
    test_close_tx_total = 0u;
    test_close_connection_state = QWEB_LWIP_CONNECTION_IDLE;
    test_close_priority = 0u;
    test_close_callbacks_detached = 0;
    test_runner_pumps_after_start = 0u;
    test_runner_done_mask = UINT32_C(0x0fffffff);
    TcpFastTmrFlag = 0;
    TcpSlowTmrFlag = 0;
}

static void test_setup_adapter(int with_assets)
{
    qweb_lwip_config_t config;

    memset(&config, 0, sizeof(config));
    config.listen_port = 8080u;
    config.tcp_poll_interval = 1u;
    config.request_timeout_ms = 1000u;
    config.connection_timeout_ms = 5000u;
    config.now_ms = test_clock;
    config.now_context = &test_now_ms;
    config.allowed_host_count = 2u;
    config.allowed_hosts[0] = "board.local:8080";
    config.allowed_hosts[1] = "192.168.1.10:8080";
    if (with_assets != 0) {
        config.resolve_asset = test_asset_resolver;
        config.asset_context = &test_router;
    }
    TEST_CHECK(qweb_lwip_adapter_init(&test_adapter,
                                      &test_router,
                                      &config) == QWEB_LWIP_OK);
    TEST_CHECK(qweb_lwip_adapter_start(&test_adapter) == QWEB_LWIP_OK);
    TEST_CHECK(test_adapter.listener == &test_listener_pcb);
    TEST_CHECK(test_listener_pcb.accept_callback != NULL);
}

static void test_accept_pcb(struct tcp_pcb *pcb, u16_t send_buffer)
{
    err_t error;

    TEST_CHECK(pcb != NULL);
    memset(pcb, 0, sizeof(*pcb));
    pcb->send_buffer = send_buffer;
    pcb->prio = TCP_PRIO_NORMAL;
    error = test_listener_pcb.accept_callback(
        test_listener_pcb.callback_argument,
        pcb,
        ERR_OK);
    TEST_CHECK(error == ERR_OK);
    TEST_CHECK(test_adapter.connection == pcb);
    TEST_CHECK(pcb->recv_callback != NULL);
}

static void test_accept_connection(u16_t send_buffer)
{
    test_accept_pcb(&test_connection_pcb, send_buffer);
}

static void test_send_bytes(const uint8_t *bytes, size_t length)
{
    struct pbuf packet;
    err_t error;

    TEST_CHECK(length <= (size_t)UINT16_MAX);
    memset(&packet, 0, sizeof(packet));
    packet.payload = (void *)bytes;
    packet.len = (u16_t)length;
    packet.tot_len = (u16_t)length;
    error = test_connection_pcb.recv_callback(
        test_connection_pcb.callback_argument,
        &test_connection_pcb,
        &packet,
        ERR_OK);
    TEST_CHECK(error == ERR_OK || error == ERR_ABRT);
}

static void test_send_chain(
    const uint8_t *first,
    size_t first_length,
    const uint8_t *second,
    size_t second_length)
{
    struct pbuf first_packet;
    struct pbuf second_packet;
    err_t error;

    TEST_CHECK(first_length + second_length <= (size_t)UINT16_MAX);
    memset(&first_packet, 0, sizeof(first_packet));
    memset(&second_packet, 0, sizeof(second_packet));
    first_packet.next = &second_packet;
    first_packet.payload = (void *)first;
    first_packet.len = (u16_t)first_length;
    first_packet.tot_len = (u16_t)(first_length + second_length);
    second_packet.payload = (void *)second;
    second_packet.len = (u16_t)second_length;
    second_packet.tot_len = (u16_t)second_length;
    error = test_connection_pcb.recv_callback(
        test_connection_pcb.callback_argument,
        &test_connection_pcb,
        &first_packet,
        ERR_OK);
    TEST_CHECK(error == ERR_OK || error == ERR_ABRT);
}

static size_t test_build_request(
    const char *method,
    const char *target,
    const char *host,
    const uint8_t *body,
    size_t body_length,
    int include_json_type)
{
    int header_length;

    header_length = snprintf(
        (char *)test_request_bytes,
        sizeof(test_request_bytes),
        "%s %s HTTP/1.1\r\n"
        "Host: %s\r\n"
        "%s"
        "Content-Length: %zu\r\n"
        "\r\n",
        method,
        target,
        host,
        include_json_type != 0 ? "Content-Type: application/json\r\n" : "",
        body_length);
    TEST_CHECK(header_length >= 0);
    TEST_CHECK((size_t)header_length + body_length < sizeof(test_request_bytes));
    if (body_length != 0u) {
        TEST_CHECK(body != NULL);
        memcpy(test_request_bytes + (size_t)header_length, body, body_length);
    }
    return (size_t)header_length + body_length;
}

static void test_ack_until_closed(void)
{
    unsigned guard = 0u;

    while (test_adapter.connection != NULL && guard < 100u) {
        ++guard;
        if (test_adapter.connection_state == QWEB_LWIP_CONNECTION_SENDING) {
            size_t outstanding = test_adapter.tx_queued - test_adapter.tx_acked;

            if (outstanding != 0u) {
                u16_t acknowledged = outstanding > (size_t)UINT16_MAX
                                         ? UINT16_MAX
                                         : (u16_t)outstanding;
                test_connection_pcb.send_buffer = (u16_t)(
                    UINT16_MAX - test_connection_pcb.send_buffer < acknowledged
                        ? UINT16_MAX
                        : test_connection_pcb.send_buffer + acknowledged);
                TEST_CHECK(test_connection_pcb.sent_callback != NULL);
                TEST_CHECK(test_connection_pcb.sent_callback(
                               test_connection_pcb.callback_argument,
                               &test_connection_pcb,
                               acknowledged) == ERR_OK);
            } else {
                TEST_CHECK(qweb_lwip_adapter_service(&test_adapter) ==
                           QWEB_LWIP_OK);
            }
        } else {
            TEST_CHECK(qweb_lwip_adapter_service(&test_adapter) ==
                       QWEB_LWIP_OK);
        }
    }
    TEST_CHECK(guard < 100u);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_connection_pcb.closed != 0u);
    TEST_CHECK(test_connection_pcb.callback_argument == NULL);
    TEST_CHECK(test_connection_pcb.recv_callback == NULL);
    TEST_CHECK(test_connection_pcb.sent_callback == NULL);
    TEST_CHECK(test_connection_pcb.error_callback == NULL);
    TEST_CHECK(test_connection_pcb.poll_callback == NULL);
    TEST_CHECK(test_connection_pcb.poll_interval == 0u);
    TEST_CHECK(test_connection_pcb.prio == TCP_PRIO_MIN);
}

static void test_consecutive_connections_detach_callbacks(void)
{
    size_t request_length;
    unsigned iteration;

    test_reset_fakes();
    test_setup_adapter(0);
    request_length = test_build_request("GET",
                                        "/api/health",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);

    for (iteration = 0u; iteration < 128u; ++iteration) {
        test_accept_connection(UINT16_MAX);
        test_send_bytes(test_request_bytes, request_length);
        TEST_CHECK(test_route_calls == iteration + 1u);
        TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                      test_tx_length,
                                      "HTTP/1.1 200 OK\r\n"));
        test_ack_until_closed();
    }
}

static void test_zero_ack_close_releases_connection_slot(void)
{
    size_t request_length;

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    request_length = test_build_request("GET",
                                        "/api/health",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_close_calls == 1u);
    TEST_CHECK(test_last_close_pcb == &test_connection_pcb);
    TEST_CHECK(test_connection_pcb.closed != 0u);
    TEST_CHECK(test_close_tx_total != 0u);
    TEST_CHECK(test_close_tx_queued == test_close_tx_total);
    TEST_CHECK(test_close_tx_acked == 0u);
    TEST_CHECK(test_close_connection_state ==
               QWEB_LWIP_CONNECTION_CLOSING);
    TEST_CHECK(test_close_priority == TCP_PRIO_MIN);
    TEST_CHECK(test_close_callbacks_detached != 0);

    test_accept_pcb(&test_second_pcb, UINT16_MAX);
    TEST_CHECK(test_adapter.connection == &test_second_pcb);
    TEST_CHECK(test_second_pcb.prio == TCP_PRIO_NORMAL);
    TEST_CHECK(test_second_pcb.recv_callback != NULL);
}

static void test_partial_queue_closes_after_final_write(void)
{
    size_t request_length;
    unsigned guard = 0u;

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(23u);
    request_length = test_build_request("GET",
                                        "/api/health",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_adapter.connection == &test_connection_pcb);
    TEST_CHECK(test_adapter.connection_state == QWEB_LWIP_CONNECTION_SENDING);
    TEST_CHECK(test_adapter.tx_queued == 23u);
    TEST_CHECK(test_adapter.tx_queued < test_adapter.tx_total);
    TEST_CHECK(test_adapter.tx_acked == 0u);
    TEST_CHECK(test_close_calls == 0u);

    while (test_adapter.connection != NULL && guard < 100u) {
        size_t outstanding = test_adapter.tx_queued - test_adapter.tx_acked;
        size_t queued_before = test_adapter.tx_queued;
        u16_t acknowledged;
        tcp_sent_fn sent_callback = test_connection_pcb.sent_callback;
        void *callback_argument = test_connection_pcb.callback_argument;

        ++guard;
        TEST_CHECK(outstanding != 0u);
        TEST_CHECK(outstanding <= (size_t)UINT16_MAX);
        TEST_CHECK(sent_callback != NULL);
        acknowledged = (u16_t)outstanding;
        test_connection_pcb.send_buffer = (u16_t)(
            UINT16_MAX - test_connection_pcb.send_buffer < acknowledged
                ? UINT16_MAX
                : test_connection_pcb.send_buffer + acknowledged);
        TEST_CHECK(sent_callback(callback_argument,
                                 &test_connection_pcb,
                                 acknowledged) == ERR_OK);
        if (test_adapter.connection != NULL) {
            TEST_CHECK(test_adapter.tx_queued > queued_before);
            TEST_CHECK(test_adapter.tx_queued < test_adapter.tx_total);
            TEST_CHECK(test_close_calls == 0u);
        }
    }
    TEST_CHECK(guard < 100u);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_close_calls == 1u);
    TEST_CHECK(test_last_close_pcb == &test_connection_pcb);
    TEST_CHECK(test_close_tx_queued == test_close_tx_total);
    TEST_CHECK(test_close_tx_acked < test_close_tx_total);
    TEST_CHECK(test_close_connection_state ==
               QWEB_LWIP_CONNECTION_CLOSING);
    TEST_CHECK(test_close_priority == TCP_PRIO_MIN);
    TEST_CHECK(test_close_callbacks_detached != 0);
    TEST_CHECK(test_connection_pcb.closed != 0u);
}

static void test_close_err_mem_restores_connection_contract(void)
{
    const u8_t original_priority = 37u;
    size_t request_length;

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    test_connection_pcb.prio = original_priority;
    test_forced_close_error = ERR_MEM;
    request_length = test_build_request("GET",
                                        "/api/health",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);

    TEST_CHECK(test_adapter.connection == &test_connection_pcb);
    TEST_CHECK(test_adapter.connection_state == QWEB_LWIP_CONNECTION_SENDING);
    TEST_CHECK(test_adapter.tx_queued == test_adapter.tx_total);
    TEST_CHECK(test_adapter.tx_acked == 0u);
    TEST_CHECK(test_close_calls == 1u);
    TEST_CHECK(test_last_close_pcb == &test_connection_pcb);
    TEST_CHECK(test_close_tx_queued == test_close_tx_total);
    TEST_CHECK(test_close_tx_acked == 0u);
    TEST_CHECK(test_close_connection_state ==
               QWEB_LWIP_CONNECTION_CLOSING);
    TEST_CHECK(test_close_priority == TCP_PRIO_MIN);
    TEST_CHECK(test_close_callbacks_detached != 0);
    TEST_CHECK(test_connection_pcb.closed == 0u);
    TEST_CHECK(test_connection_pcb.callback_argument == &test_adapter);
    TEST_CHECK(test_connection_pcb.recv_callback != NULL);
    TEST_CHECK(test_connection_pcb.sent_callback != NULL);
    TEST_CHECK(test_connection_pcb.error_callback != NULL);
    TEST_CHECK(test_connection_pcb.poll_callback != NULL);
    TEST_CHECK(test_connection_pcb.poll_interval == 1u);
    TEST_CHECK(test_connection_pcb.prio == original_priority);

    TEST_CHECK(qweb_lwip_adapter_service(&test_adapter) == QWEB_LWIP_OK);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_close_calls == 2u);
    TEST_CHECK(test_last_close_pcb == &test_connection_pcb);
    TEST_CHECK(test_close_tx_queued == test_close_tx_total);
    TEST_CHECK(test_close_tx_acked == 0u);
    TEST_CHECK(test_close_connection_state ==
               QWEB_LWIP_CONNECTION_CLOSING);
    TEST_CHECK(test_close_priority == TCP_PRIO_MIN);
    TEST_CHECK(test_close_callbacks_detached != 0);
    TEST_CHECK(test_connection_pcb.closed != 0u);
    TEST_CHECK(test_connection_pcb.callback_argument == NULL);
    TEST_CHECK(test_connection_pcb.recv_callback == NULL);
    TEST_CHECK(test_connection_pcb.sent_callback == NULL);
    TEST_CHECK(test_connection_pcb.error_callback == NULL);
    TEST_CHECK(test_connection_pcb.poll_callback == NULL);
    TEST_CHECK(test_connection_pcb.prio == TCP_PRIO_MIN);
}

static void test_config_and_start_failures(void)
{
    qweb_lwip_config_t config;

    test_reset_fakes();
    memset(&config, 0, sizeof(config));
    config.listen_port = 80u;
    config.request_timeout_ms = 100u;
    config.connection_timeout_ms = 200u;
    config.now_ms = test_clock;
    config.now_context = &test_now_ms;
    TEST_CHECK(qweb_lwip_adapter_init(&test_adapter,
                                      &test_router,
                                      &config) == QWEB_LWIP_ERR_CONFIG);
    config.allowed_host_count = 1u;
    config.allowed_hosts[0] = "bad host";
    TEST_CHECK(qweb_lwip_adapter_init(&test_adapter,
                                      &test_router,
                                      &config) == QWEB_LWIP_ERR_CONFIG);
    config.allowed_hosts[0] = "board.local";
    TEST_CHECK(qweb_lwip_adapter_init(&test_adapter,
                                      &test_router,
                                      &config) == QWEB_LWIP_OK);
    test_forced_bind_error = ERR_ARG;
    TEST_CHECK(qweb_lwip_adapter_start(&test_adapter) == QWEB_LWIP_ERR_LWIP);
}

static void test_fragmented_submit_is_async(void)
{
    static const uint8_t body[] =
        "{\"tokens\":[374],\"max_new_tokens\":2}";
    size_t request_length;
    err_t second_error;

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    memset(&test_second_pcb, 0, sizeof(test_second_pcb));
    second_error = test_listener_pcb.accept_callback(
        test_listener_pcb.callback_argument,
        &test_second_pcb,
        ERR_OK);
    TEST_CHECK(second_error == ERR_ABRT);
    TEST_CHECK(test_second_pcb.aborted != 0u);

    request_length = test_build_request("POST",
                                        "/api/generate",
                                        "BOARD.LOCAL:8080",
                                        body,
                                        sizeof(body) - 1u,
                                        1);
    test_send_bytes(test_request_bytes, 17u);
    TEST_CHECK(test_route_calls == 0u);
    test_send_chain(test_request_bytes + 17u,
                    31u,
                    test_request_bytes + 48u,
                    request_length - 48u);
    TEST_CHECK(test_route_calls == 1u);
    TEST_CHECK(test_job_active != 0);
    TEST_CHECK(test_job_step_calls == 0u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "HTTP/1.1 202 Accepted\r\n"));
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "Cache-Control: no-store\r\n"));
    test_ack_until_closed();

    TEST_CHECK(qweb_board_app_init(&test_board_app,
                                   &test_job,
                                   &test_adapter,
                                   test_generic_network_pump,
                                   &test_board_app,
                                   2u) == QWEB_BOARD_OK);
    TEST_CHECK(qweb_board_app_step(&test_board_app) == QWEB_BOARD_OK);
    TEST_CHECK(test_job_step_calls == 1u);
    TEST_CHECK(test_network_pump_calls == 1u);
}

static void test_static_asset_hit_miss_and_method(void)
{
    size_t request_length;
    size_t body_offset;

    test_reset_fakes();
    test_setup_adapter(1);
    test_accept_connection(UINT16_MAX);
    request_length = test_build_request("GET",
                                        "/",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_route_calls == 0u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "ETag: \"qweb-index-test\"\r\n"));
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "Cache-Control: public, max-age=60\r\n"));
    body_offset = test_http_body_offset();
    TEST_CHECK(body_offset != SIZE_MAX);
    TEST_CHECK(test_tx_length - body_offset == strlen("<html>qweb</html>"));
    TEST_CHECK(memcmp(test_tx_bytes + body_offset,
                      "<html>qweb</html>",
                      strlen("<html>qweb</html>")) == 0);
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(1);
    test_accept_connection(UINT16_MAX);
    request_length = test_build_request("GET",
                                        "/unknown",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_route_calls == 1u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "HTTP/1.1 404 Not Found\r\n"));
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(1);
    test_accept_connection(UINT16_MAX);
    request_length = test_build_request("POST",
                                        "/",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_route_calls == 0u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "HTTP/1.1 405 Method Not Allowed\r\n"));
    test_ack_until_closed();
}

static void test_host_pipeline_timeout_and_close_paths(void)
{
    size_t request_length;
    size_t first_length;
    tcp_err_fn error_callback;
    void *error_argument;

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    request_length = test_build_request("GET",
                                        "/api/health",
                                        "attacker.example:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_route_calls == 0u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "HTTP/1.1 403 Forbidden\r\n"));
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    first_length = test_build_request("GET",
                                      "/api/health",
                                      "board.local:8080",
                                      NULL,
                                      0u,
                                      0);
    TEST_CHECK(first_length + first_length < sizeof(test_request_bytes));
    memcpy(test_request_bytes + first_length,
           test_request_bytes,
           first_length);
    test_send_bytes(test_request_bytes, first_length * 2u);
    TEST_CHECK(test_route_calls == 0u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "http_pipelining_not_supported"));
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    test_now_ms += 1000u;
    TEST_CHECK(qweb_lwip_adapter_service(&test_adapter) == QWEB_LWIP_OK);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "HTTP/1.1 408 Request Timeout\r\n"));
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(0u);
    test_now_ms += 5000u;
    TEST_CHECK(qweb_lwip_adapter_service(&test_adapter) == QWEB_LWIP_OK);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_connection_pcb.aborted != 0u);

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    TEST_CHECK(test_connection_pcb.recv_callback(
                   test_connection_pcb.callback_argument,
                   &test_connection_pcb,
                   NULL,
                   ERR_OK) == ERR_OK);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_connection_pcb.closed != 0u);

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    error_callback = test_connection_pcb.error_callback;
    error_argument = test_connection_pcb.callback_argument;
    TEST_CHECK(error_callback != NULL);
    error_callback(error_argument, ERR_RST);
    TEST_CHECK(test_adapter.connection == NULL);
}

static void test_binary_output_is_length_delimited(void)
{
    size_t request_length;
    size_t body_offset;
    static const uint8_t expected[] = {0x00u, 0x41u, 0xffu, 0x0au};

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(64u);
    request_length = test_build_request("GET",
                                        "/api/generate/1/output",
                                        "192.168.1.10:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "Content-Type: application/octet-stream\r\n"));
    test_ack_until_closed();
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "Content-Length: 4\r\n"));
    body_offset = test_http_body_offset();
    TEST_CHECK(body_offset != SIZE_MAX);
    TEST_CHECK(test_tx_length - body_offset == sizeof(expected));
    TEST_CHECK(memcmp(test_tx_bytes + body_offset,
                      expected,
                      sizeof(expected)) == 0);
}

static void test_error_and_retry_paths(void)
{
    static const uint8_t malformed[] =
        "GET /api/health HTTP/1.1\nHost: board.local:8080\n\n";
    struct pbuf error_packet;
    size_t request_length;
    err_t receive_error;

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    test_send_bytes(malformed, sizeof(malformed) - 1u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "HTTP/1.1 400 Bad Request\r\n"));
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(1);
    test_accept_connection(UINT16_MAX);
    request_length = test_build_request("GET",
                                        "/asset-err",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_route_calls == 0u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "HTTP/1.1 500 Internal Server Error\r\n"));
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    request_length = test_build_request("GET",
                                        "/router-error",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_route_calls == 1u);
    TEST_CHECK(test_bytes_contain(test_tx_bytes,
                                  test_tx_length,
                                  "router_failure"));
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    test_forced_write_error = ERR_MEM;
    request_length = test_build_request("GET",
                                        "/api/health",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_tx_length == 0u);
    TEST_CHECK(qweb_lwip_adapter_service(&test_adapter) == QWEB_LWIP_OK);
    TEST_CHECK(test_tx_length != 0u);
    test_ack_until_closed();

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    test_forced_write_error = ERR_RST;
    request_length = test_build_request("GET",
                                        "/api/health",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_connection_pcb.aborted != 0u);

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    test_forced_output_error = ERR_RST;
    request_length = test_build_request("GET",
                                        "/api/health",
                                        "board.local:8080",
                                        NULL,
                                        0u,
                                        0);
    test_send_bytes(test_request_bytes, request_length);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_connection_pcb.aborted != 0u);

    test_reset_fakes();
    test_setup_adapter(0);
    test_accept_connection(UINT16_MAX);
    memset(&error_packet, 0, sizeof(error_packet));
    error_packet.payload = test_request_bytes;
    error_packet.len = 1u;
    error_packet.tot_len = 1u;
    receive_error = test_connection_pcb.recv_callback(
        test_connection_pcb.callback_argument,
        &test_connection_pcb,
        &error_packet,
        ERR_RST);
    TEST_CHECK(receive_error == ERR_ABRT);
    TEST_CHECK(test_adapter.connection == NULL);
    TEST_CHECK(test_connection_pcb.aborted != 0u);
}

static void test_xilinx_pump_and_full28_runner(void)
{
    qweb_board_xilinx_network_t network;
    struct netif network_interface;
    qot_run_config_t config;
    qot_result_t result;
    uint32_t layer;
    int rc;

    test_reset_fakes();
    memset(&network, 0, sizeof(network));
    memset(&network_interface, 0, sizeof(network_interface));
    network.network_interface = &network_interface;
    TcpFastTmrFlag = 1;
    TcpSlowTmrFlag = 1;
    qweb_board_xilinx_lwip_pump(&network);
    TEST_CHECK(test_fast_timer_calls == 1u);
    TEST_CHECK(test_slow_timer_calls == 1u);
    TEST_CHECK(test_xemac_input_calls == 1u);
    TEST_CHECK(TcpFastTmrFlag == 0);
    TEST_CHECK(TcpSlowTmrFlag == 0);

    test_setup_adapter(0);
    TEST_CHECK(qweb_board_app_init(&test_board_app,
                                   &test_job,
                                   &test_adapter,
                                   test_generic_network_pump,
                                   &test_board_app,
                                   3u) == QWEB_BOARD_OK);
    memset(&config, 0, sizeof(config));
    for (layer = 0u; layer < QOT_MAX_LAYERS; ++layer) {
        test_layer_tables[layer].qkv = UINT64_C(0x40000000);
        test_layer_tables[layer].input_norm = UINT64_C(0x40001000);
        test_layer_tables[layer].attn_frontend = UINT64_C(0x40002000);
        test_layer_tables[layer].attn_score_value = UINT64_C(0x40003000);
        test_layer_tables[layer].o_proj = UINT64_C(0x40004000);
        test_layer_tables[layer].post_attn_norm = UINT64_C(0x40005000);
        test_layer_tables[layer].mlp_gate_up = UINT64_C(0x40006000);
        test_layer_tables[layer].mlp_silu_mul = UINT64_C(0x40007000);
        test_layer_tables[layer].mlp_down = UINT64_C(0x40008000);
        test_layer_tables[layer].mlp_residual_add = UINT64_C(0x40009000);
    }
    config.layer_start = 0u;
    config.layer_count = QOT_MAX_LAYERS;
    config.position = 0u;
    config.runtime_context_enable = 1u;
    config.embedding_enable = 1u;
    config.embedding_weight_base = UINT64_C(0x40010000);
    config.embedding_scale_base = UINT64_C(0x40020000);
    config.input_hidden_base = UINT64_C(0x40030000);
    config.output_hidden_base = UINT64_C(0x40040000);
    config.kv_cache_base = UINT64_C(0x40050000);
    config.final_tail_qmap_base = UINT64_C(0x40060000);
    config.layer_qmap_bases = test_layer_tables;
    config.layer_qmap_base_count = QOT_MAX_LAYERS;
    memset(&result, 0, sizeof(result));
    rc = qweb_board_qot_runner(&test_board_app,
                               (uintptr_t)test_mmio,
                               &config,
                               374u,
                               0u,
                               100u,
                               &result);
    TEST_CHECK(rc == QOT_OK);
    TEST_CHECK(result.token_id == 264u);
    TEST_CHECK(result.layers_started == 28u);
    TEST_CHECK(result.layers_completed == 28u);
    TEST_CHECK(result.layer_done_mask == UINT32_C(0x0fffffff));
    TEST_CHECK(test_runner_pumps_after_start >= 1u);

    memset(test_mmio, 0, sizeof(test_mmio));
    memset(&result, 0, sizeof(result));
    test_runner_pumps_after_start = 0u;
    test_runner_done_mask = UINT32_C(0x07ffffff);
    rc = qweb_board_qot_runner(&test_board_app,
                               (uintptr_t)test_mmio,
                               &config,
                               374u,
                               0u,
                               100u,
                               &result);
    TEST_CHECK(rc == QOT_ERR_STATUS);
}

int main(void)
{
    test_config_and_start_failures();
    test_consecutive_connections_detach_callbacks();
    test_zero_ack_close_releases_connection_slot();
    test_partial_queue_closes_after_final_write();
    test_close_err_mem_restores_connection_contract();
    test_fragmented_submit_is_async();
    test_static_asset_hit_miss_and_method();
    test_host_pipeline_timeout_and_close_paths();
    test_binary_output_is_length_delimited();
    test_error_and_retry_paths();
    test_xilinx_pump_and_full28_runner();
    (void)printf("PASS qweb raw-lwIP adapter host mock tests\n");
    return EXIT_SUCCESS;
}
