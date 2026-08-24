#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lwip/ip4_addr.h"
#include "lwip/netif.h"
#include "xiltimer.h"
#include "xparameters.h"

#include "qmap_model_config_generated.h"
#include "qtk_tokenizer_runtime.h"
#include "qweb_board_app.h"
#include "qweb_board_entry.h"
#include "qweb_router.h"
#include "web_assets.h"

#ifndef QWEB_QOT_BASEADDR
#  if defined(XPAR_QMAP_ONE_TOKEN_AXI_BD_0_BASEADDR)
#    define QWEB_QOT_BASEADDR \
        ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXI_BD_0_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXI_BD_0_S_AXI_BASEADDR)
#    define QWEB_QOT_BASEADDR \
        ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXI_BD_0_S_AXI_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXI_BD_0_S_AXI_CTRL_BASEADDR)
#    define QWEB_QOT_BASEADDR \
        ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXI_BD_0_S_AXI_CTRL_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_BASEADDR)
#    define QWEB_QOT_BASEADDR \
        ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_S_AXI_BASEADDR)
#    define QWEB_QOT_BASEADDR \
        ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_S_AXI_BASEADDR)
#  elif defined(XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_S_AXI_CTRL_BASEADDR)
#    define QWEB_QOT_BASEADDR \
        ((uintptr_t)XPAR_QMAP_ONE_TOKEN_AXIL_TOP_0_S_AXI_CTRL_BASEADDR)
#  else
#    define QWEB_QOT_BASEADDR ((uintptr_t)0u)
#  endif
#endif

#ifndef QWEB_DDR4_STATUS_BASE
#define QWEB_DDR4_STATUS_BASE UINT64_C(0x00000000A0010000)
#endif

#ifndef QWEB_MODEL_POLL_ATTEMPTS
#define QWEB_MODEL_POLL_ATTEMPTS UINT32_C(200000000)
#endif

#ifndef QWEB_POLLS_BETWEEN_NETWORK_PUMPS
#define QWEB_POLLS_BETWEEN_NETWORK_PUMPS UINT32_C(1024)
#endif

#ifndef QWEB_HTTP_PORT
#define QWEB_HTTP_PORT UINT16_C(80)
#endif

#ifndef QWEB_HTTP_REQUEST_TIMEOUT_MS
#define QWEB_HTTP_REQUEST_TIMEOUT_MS UINT32_C(5000)
#endif

#ifndef QWEB_HTTP_CONNECTION_TIMEOUT_MS
#define QWEB_HTTP_CONNECTION_TIMEOUT_MS UINT32_C(30000)
#endif

#ifndef QWEB_HTTP_CACHE_MAX_AGE_SECONDS
#define QWEB_HTTP_CACHE_MAX_AGE_SECONDS UINT32_C(60)
#endif

#define QWEB_DDR4_CALIB_COMPLETE_MASK UINT32_C(0x00000001)
#define QWEB_DDR4_UI_RESET_MASK       UINT32_C(0x00000002)
#define QWEB_DDR4_AXI_RESETN_MASK     UINT32_C(0x00000004)
#define QWEB_QMAP_MAGIC               UINT32_C(0x50414D51)

extern struct netif *echo_netif;
#if defined(QWEB_BOARD_ENTRY_HOST_TEST)
extern const uint8_t *qweb_board_entry_host_tokenizer(size_t *size);
#else
extern const uint8_t qot_tokenizer_asset_start[];
extern const uint8_t qot_tokenizer_asset_end[];
#endif

static qtk_asset_t qweb_tokenizer;
static qweb_job_t qweb_job;
static qweb_router_t qweb_router;
static qweb_lwip_adapter_t qweb_adapter;
static qweb_board_app_t qweb_app;
static qweb_board_xilinx_network_t qweb_network;
static uint8_t qweb_started;
static char qweb_board_host[IP4ADDR_STRLEN_MAX];
static char qweb_board_authority[IP4ADDR_STRLEN_MAX + 6u];

#if defined(QWEB_BOARD_ENTRY_HOST_TEST)
extern uint32_t qweb_board_entry_host_read32(uint64_t address);

static uint32_t qweb_board_read32(uint64_t address)
{
    return qweb_board_entry_host_read32(address);
}
#else
static uint32_t qweb_board_read32(uint64_t address)
{
    return *(volatile const uint32_t *)(uintptr_t)address;
}
#endif

static int qweb_model_memory_ready(void)
{
    uint32_t ddr_status = qweb_board_read32(QWEB_DDR4_STATUS_BASE);
    uint32_t qkv_magic =
        qweb_board_read32(qot_model_layer_qmap_bases[0].qkv);
    uint32_t input_norm_magic =
        qweb_board_read32(qot_model_layer_qmap_bases[0].input_norm);
    uint32_t final_tail_magic =
        qweb_board_read32(QOT_MODEL_FINAL_TAIL_QMAP_BASE);

    (void)printf("DDR4 status=0x%08lx\r\n", (unsigned long)ddr_status);
    if ((ddr_status & QWEB_DDR4_CALIB_COMPLETE_MASK) == 0u ||
        (ddr_status & QWEB_DDR4_UI_RESET_MASK) != 0u ||
        (ddr_status & QWEB_DDR4_AXI_RESETN_MASK) == 0u) {
        (void)printf("ERROR DDR4_NOT_READY\r\n");
        return 0;
    }
    if (qkv_magic != QWEB_QMAP_MAGIC ||
        input_norm_magic != QWEB_QMAP_MAGIC ||
        final_tail_magic != QWEB_QMAP_MAGIC) {
        (void)printf("ERROR RUNTIME_SENTINEL qkv=0x%08lx"
                     " input_norm=0x%08lx final_tail=0x%08lx\r\n",
                     (unsigned long)qkv_magic,
                     (unsigned long)input_norm_magic,
                     (unsigned long)final_tail_magic);
        return 0;
    }
    return 1;
}

static uint32_t qweb_now_ms(void *context)
{
    XTime ticks;
    uint64_t ticks_per_millisecond;

    (void)context;
    XTime_GetTime(&ticks);
    ticks_per_millisecond = (uint64_t)COUNTS_PER_SECOND / UINT64_C(1000);
    if (ticks_per_millisecond == 0u) return 0u;
    return (uint32_t)((uint64_t)ticks / ticks_per_millisecond);
}

static int qweb_capture_board_authority(void)
{
    const ip4_addr_t *address;
    int written;

    if (echo_netif == NULL) return 0;
    address = netif_ip4_addr(echo_netif);
    if (address == NULL ||
        (ip4_addr1(address) == 0u && ip4_addr2(address) == 0u &&
         ip4_addr3(address) == 0u && ip4_addr4(address) == 0u)) {
        return 0;
    }
    written = snprintf(qweb_board_host,
                       sizeof(qweb_board_host),
                       "%u.%u.%u.%u",
                       (unsigned)ip4_addr1(address),
                       (unsigned)ip4_addr2(address),
                       (unsigned)ip4_addr3(address),
                       (unsigned)ip4_addr4(address));
    if (written <= 0 || (size_t)written >= sizeof(qweb_board_host)) return 0;
    written = snprintf(qweb_board_authority,
                       sizeof(qweb_board_authority),
                       "%s:%u",
                       qweb_board_host,
                       (unsigned)QWEB_HTTP_PORT);
    return written > 0 && (size_t)written < sizeof(qweb_board_authority);
}

static int qweb_content_type_from_mime(
    const char *mime,
    qweb_http_content_type_t *content_type)
{
    if (mime == NULL || content_type == NULL) return 0;
    if (strcmp(mime, "text/html; charset=utf-8") == 0) {
        *content_type = QWEB_HTTP_CONTENT_HTML;
        return 1;
    }
    if (strcmp(mime, "text/css; charset=utf-8") == 0) {
        *content_type = QWEB_HTTP_CONTENT_CSS;
        return 1;
    }
    if (strcmp(mime, "text/javascript; charset=utf-8") == 0) {
        *content_type = QWEB_HTTP_CONTENT_JAVASCRIPT;
        return 1;
    }
    return 0;
}

static int qweb_resolve_static_asset(
    void *context,
    const char *target,
    size_t target_length,
    qweb_lwip_static_asset_t *resolved)
{
    const qweb_web_asset_t *asset;

    (void)context;
    if (target == NULL || resolved == NULL) return QWEB_LWIP_ASSET_ERROR;
    asset = qweb_web_asset_find(target, target_length);
    if (asset == NULL) return QWEB_LWIP_ASSET_MISS;
    if (asset->body == NULL || asset->etag == NULL ||
        !qweb_content_type_from_mime(asset->mime_type,
                                    &resolved->content_type)) {
        return QWEB_LWIP_ASSET_ERROR;
    }
    resolved->body = asset->body;
    resolved->body_length = asset->body_length;
    resolved->etag = asset->etag;
    resolved->etag_length = strlen(asset->etag);
    resolved->cache_max_age_seconds = QWEB_HTTP_CACHE_MAX_AGE_SECONDS;
    return QWEB_LWIP_ASSET_HIT;
}

static int qweb_initialize_runtime(void)
{
    const char *allowed_hosts[] = {
        qweb_board_host,
        qweb_board_authority,
    };
    qweb_lwip_config_t network_config;
    qot_run_config_t model_config;
    qtk_status_t tokenizer_status;
    size_t tokenizer_size;
    size_t host_index;
    int rc;

    if ((uintptr_t)QWEB_QOT_BASEADDR == (uintptr_t)0u ||
        echo_netif == NULL || !qweb_model_memory_ready()) {
        return 0;
    }
    if (!qweb_capture_board_authority()) {
        (void)printf("ERROR NETWORK_ADDRESS\r\n");
        return 0;
    }
#if defined(QWEB_BOARD_ENTRY_HOST_TEST)
    {
        const uint8_t *tokenizer_data =
            qweb_board_entry_host_tokenizer(&tokenizer_size);
        tokenizer_status = qtk_asset_init(&qweb_tokenizer,
                                          tokenizer_data,
                                          tokenizer_size);
    }
#else
    tokenizer_size =
        (size_t)(qot_tokenizer_asset_end - qot_tokenizer_asset_start);
    tokenizer_status = qtk_asset_init(&qweb_tokenizer,
                                      qot_tokenizer_asset_start,
                                      tokenizer_size);
#endif
    if (tokenizer_status != QTK_OK) {
        (void)printf("ERROR TOKENIZER_ASSET rc=%d bytes=%lu\r\n",
                     (int)tokenizer_status,
                     (unsigned long)tokenizer_size);
        return 0;
    }

    memset(&network_config, 0, sizeof(network_config));
    qweb_network.network_interface = echo_netif;
    rc = qweb_board_app_init(&qweb_app,
                             &qweb_job,
                             &qweb_adapter,
                             qweb_board_xilinx_lwip_pump,
                             &qweb_network,
                             QWEB_POLLS_BETWEEN_NETWORK_PUMPS);
    if (rc != QWEB_BOARD_OK) return 0;

    model_config = qot_model_default_run_config();
    rc = qweb_job_init(&qweb_job,
                       (uintptr_t)QWEB_QOT_BASEADDR,
                       &model_config,
                       &qweb_tokenizer,
                       qweb_board_qot_runner,
                       &qweb_app,
                       QWEB_MODEL_POLL_ATTEMPTS);
    if (rc != QWEB_JOB_OK) return 0;
    rc = qweb_router_init(&qweb_router, &qweb_job);
    if (rc != QWEB_ROUTER_OK) return 0;

    network_config.listen_port = QWEB_HTTP_PORT;
    network_config.request_timeout_ms = QWEB_HTTP_REQUEST_TIMEOUT_MS;
    network_config.connection_timeout_ms =
        QWEB_HTTP_CONNECTION_TIMEOUT_MS;
    network_config.now_ms = qweb_now_ms;
    network_config.resolve_asset = qweb_resolve_static_asset;
    network_config.allowed_host_count =
        sizeof(allowed_hosts) / sizeof(allowed_hosts[0]);
    for (host_index = 0u;
         host_index < network_config.allowed_host_count;
         ++host_index) {
        network_config.allowed_hosts[host_index] = allowed_hosts[host_index];
    }
    rc = qweb_lwip_adapter_init(&qweb_adapter,
                                &qweb_router,
                                &network_config);
    if (rc != QWEB_LWIP_OK) return 0;
    rc = qweb_lwip_adapter_start(&qweb_adapter);
    if (rc != QWEB_LWIP_OK) {
        qweb_lwip_adapter_stop(&qweb_adapter);
        return 0;
    }

    (void)printf("TOKENIZER tokens=%lu model_vocab=%lu eos=%lu bytes=%lu\r\n",
                 (unsigned long)qweb_tokenizer.token_count,
                 (unsigned long)qweb_tokenizer.model_vocab_size,
                 (unsigned long)qweb_tokenizer.eos_token_id,
                 (unsigned long)tokenizer_size);
    (void)printf("QWEB READY http://%s/ context=%u vocab=%u\r\n",
                 qweb_board_authority,
                 (unsigned)QOT_MAX_CONTEXT,
                 (unsigned)QOT_VOCAB_SIZE);
    return 1;
}

void print_app_header(void)
{
    (void)printf("\r\nQwen3-0.6B full28 PS Web demo\r\n");
    (void)printf("QOT_BASEADDR=0x%016llx\r\n",
                 (unsigned long long)(uintptr_t)QWEB_QOT_BASEADDR);
}

int start_application(void)
{
    if (qweb_started != 0u) return -1;
    qweb_started = qweb_initialize_runtime() ? 1u : 0u;
    if (qweb_started == 0u) {
        (void)printf("ERROR QWEB_INITIALIZATION\r\n");
        return -1;
    }
    return 0;
}

int transfer_data(void)
{
    int rc;

    if (qweb_started == 0u) return -1;
    rc = qweb_board_app_step(&qweb_app);
    if (rc == QWEB_BOARD_ERR_JOB) {
        (void)printf("ERROR QWEB_JOB job=%lu code=%d session=%d\r\n",
                     (unsigned long)qweb_job.job_id,
                     qweb_job.error_code,
                     qweb_job.session_error_code);
        return 0;
    }
    return rc == QWEB_BOARD_OK ? 0 : -1;
}
