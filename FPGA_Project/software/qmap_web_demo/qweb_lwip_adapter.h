#ifndef QWEB_LWIP_ADAPTER_H
#define QWEB_LWIP_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#include "lwip/tcp.h"

#include "qweb_router.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Application storage is fixed. lwIP may still use its configured memp pools,
 * but this adapter never calls malloc/calloc/realloc/free and never owns more
 * than one accepted connection.
 */
#define QWEB_LWIP_MAX_ALLOWED_HOSTS 4u
#define QWEB_LWIP_MAX_HOST_BYTES 127u
#define QWEB_LWIP_MAX_ETAG_BYTES 95u
#define QWEB_LWIP_TX_HEADER_BYTES 512u
#define QWEB_LWIP_DEFAULT_POLL_INTERVAL 2u

enum {
    QWEB_LWIP_OK = 0,
    QWEB_LWIP_ERR_NULL = -1,
    QWEB_LWIP_ERR_CONFIG = -2,
    QWEB_LWIP_ERR_STATE = -3,
    QWEB_LWIP_ERR_LWIP = -4,
    QWEB_LWIP_ERR_FORMAT = -5
};

typedef uint32_t (*qweb_lwip_now_ms_fn)(void *context);

enum {
    QWEB_LWIP_ASSET_ERROR = -1,
    QWEB_LWIP_ASSET_MISS = 0,
    QWEB_LWIP_ASSET_HIT = 1
};

typedef struct qweb_lwip_static_asset {
    qweb_http_content_type_t content_type;
    const uint8_t *body;
    size_t body_length;
    const char *etag;
    size_t etag_length;
    uint32_t cache_max_age_seconds;
} qweb_lwip_static_asset_t;

/*
 * Resolve an exact request target such as "/" or "/app.js". The resolver
 * returns HIT only for a stable, length-delimited ROM/static body; the adapter
 * enforces GET with an empty body before serving it. MISS falls through to the
 * API router, while ERROR becomes a bounded HTTP 500 response.
 */
typedef int (*qweb_lwip_asset_resolver_fn)(
    void *context,
    const char *target,
    size_t target_length,
    qweb_lwip_static_asset_t *asset);

typedef struct qweb_lwip_config {
    uint16_t listen_port;
    uint8_t tcp_poll_interval;
    uint32_t request_timeout_ms;
    uint32_t connection_timeout_ms;
    qweb_lwip_now_ms_fn now_ms;
    void *now_context;
    qweb_lwip_asset_resolver_fn resolve_asset;
    void *asset_context;
    size_t allowed_host_count;
    const char *allowed_hosts[QWEB_LWIP_MAX_ALLOWED_HOSTS];
} qweb_lwip_config_t;

typedef enum qweb_lwip_connection_state {
    QWEB_LWIP_CONNECTION_IDLE = 0,
    QWEB_LWIP_CONNECTION_RECEIVING,
    QWEB_LWIP_CONNECTION_SENDING,
    QWEB_LWIP_CONNECTION_CLOSING
} qweb_lwip_connection_state_t;

/*
 * This object is much larger than a bare-metal BSP stack. Place it in static
 * or global storage. Host authorities are copied during init and compared
 * case-insensitively as exact, full HTTP Host values (including a port when
 * one is present). There are no wildcard or suffix matches.
 */
typedef struct qweb_lwip_adapter {
    qweb_router_t *router;
    qweb_lwip_now_ms_fn now_ms;
    void *now_context;
    qweb_lwip_asset_resolver_fn resolve_asset;
    void *asset_context;
    struct tcp_pcb *listener;
    struct tcp_pcb *connection;
    qweb_lwip_connection_state_t connection_state;
    uint16_t listen_port;
    uint8_t tcp_poll_interval;
    uint8_t initialized;
    uint32_t request_timeout_ms;
    uint32_t connection_timeout_ms;
    uint32_t accepted_at_ms;
    uint32_t request_started_at_ms;
    size_t allowed_host_count;
    char allowed_hosts[QWEB_LWIP_MAX_ALLOWED_HOSTS]
                      [QWEB_LWIP_MAX_HOST_BYTES + 1u];
    size_t allowed_host_lengths[QWEB_LWIP_MAX_ALLOWED_HOSTS];
    qweb_http_parser_t parser;
    uint8_t json_scratch[QWEB_ROUTER_MAX_JSON_BODY_BYTES];
    qweb_route_response_t route_response;
    uint8_t tx_header[QWEB_LWIP_TX_HEADER_BYTES];
    size_t tx_header_length;
    size_t tx_header_offset;
    const uint8_t *tx_body;
    size_t tx_body_length;
    size_t tx_body_offset;
    size_t tx_queued;
    size_t tx_acked;
    size_t tx_total;
} qweb_lwip_adapter_t;

int qweb_lwip_adapter_init(
    qweb_lwip_adapter_t *adapter,
    qweb_router_t *router,
    const qweb_lwip_config_t *config);

/* Create, bind, and listen using the lwIP raw callback API. */
int qweb_lwip_adapter_start(qweb_lwip_adapter_t *adapter);

/* Abort an accepted connection and close the listener. Safe after partial init. */
void qweb_lwip_adapter_stop(qweb_lwip_adapter_t *adapter);

/*
 * Service TX retry and both request and absolute connection deadlines. Call
 * this from the board main loop and from the PL runner's network pump.
 */
int qweb_lwip_adapter_service(qweb_lwip_adapter_t *adapter);

int qweb_lwip_adapter_connection_active(
    const qweb_lwip_adapter_t *adapter);

#ifdef __cplusplus
}
#endif

#endif /* QWEB_LWIP_ADAPTER_H */
