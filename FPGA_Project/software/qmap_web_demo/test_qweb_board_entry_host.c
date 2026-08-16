#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lwip/netif.h"
#include "xtime_l.h"

#include "qmap_model_config_generated.h"
#include "qweb_board_app.h"
#include "qweb_board_entry.h"
#include "qweb_job.h"
#include "qweb_lwip_adapter.h"
#include "qweb_router.h"
#include "web_assets.h"

#define TEST_QMAP_MAGIC UINT32_C(0x50414D51)
#define TEST_DDR_READY  UINT32_C(0x00000005)

static struct netif test_netif;
struct netif *echo_netif = &test_netif;

static uint32_t test_ddr_status;
static uint32_t test_qmap_magic = TEST_QMAP_MAGIC;
static XTime test_ticks;
static int test_adapter_start_result;
static int test_board_step_result = QWEB_BOARD_OK;
static unsigned test_tokenizer_init_calls;
static unsigned test_adapter_init_calls;
static unsigned test_adapter_start_calls;
static unsigned test_adapter_stop_calls;
static unsigned test_job_init_calls;
static unsigned test_router_init_calls;
static unsigned test_board_init_calls;
static qweb_lwip_config_t test_network_config;
static qweb_job_t *test_job_pointer;

static const uint8_t test_tokenizer_bytes[] = { 0x51u, 0x54u, 0x4bu, 0x31u };
static const uint8_t test_index_body[] = "<html>board</html>";
static const qweb_web_asset_t test_index_asset = {
    "/",
    1u,
    "text/html; charset=utf-8",
    "\"entry-etag\"",
    test_index_body,
    sizeof(test_index_body) - 1u,
};

static void test_set_ipv4(uint8_t first,
                          uint8_t second,
                          uint8_t third,
                          uint8_t fourth)
{
    uint8_t *bytes = (uint8_t *)&test_netif.ip_addr.addr;

    bytes[0] = first;
    bytes[1] = second;
    bytes[2] = third;
    bytes[3] = fourth;
}

uint32_t qweb_board_entry_host_read32(uint64_t address);
const uint8_t *qweb_board_entry_host_tokenizer(size_t *size);

static void test_fail(const char *expression, const char *file, int line)
{
    (void)fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expression);
    exit(EXIT_FAILURE);
}

#define TEST_CHECK(expression)                                               \
    do {                                                                     \
        if (!(expression)) test_fail(#expression, __FILE__, __LINE__);       \
    } while (0)

uint32_t qweb_board_entry_host_read32(uint64_t address)
{
    if (address == UINT64_C(0x00000000A0010000)) return test_ddr_status;
    if (address == qot_model_layer_qmap_bases[0].qkv ||
        address == qot_model_layer_qmap_bases[0].input_norm ||
        address == QOT_MODEL_FINAL_TAIL_QMAP_BASE) {
        return test_qmap_magic;
    }
    return 0u;
}

const uint8_t *qweb_board_entry_host_tokenizer(size_t *size)
{
    TEST_CHECK(size != NULL);
    *size = sizeof(test_tokenizer_bytes);
    return test_tokenizer_bytes;
}

void XTime_GetTime(XTime *ticks)
{
    TEST_CHECK(ticks != NULL);
    *ticks = test_ticks;
}

qtk_status_t qtk_asset_init(
    qtk_asset_t *asset,
    const void *bytes,
    size_t length)
{
    TEST_CHECK(asset != NULL);
    TEST_CHECK(bytes == test_tokenizer_bytes);
    TEST_CHECK(length == sizeof(test_tokenizer_bytes));
    memset(asset, 0, sizeof(*asset));
    asset->token_count = 151669u;
    asset->model_vocab_size = QOT_VOCAB_SIZE;
    asset->eos_token_id = 151643u;
    ++test_tokenizer_init_calls;
    return QTK_OK;
}

const qweb_web_asset_t *qweb_web_asset_find(
    const char *path,
    size_t path_length)
{
    if (path != NULL && path_length == 1u && path[0] == '/') {
        return &test_index_asset;
    }
    return NULL;
}

int qweb_board_app_init(
    qweb_board_app_t *app,
    qweb_job_t *job,
    qweb_lwip_adapter_t *adapter,
    qweb_board_network_pump_fn network_pump,
    void *network_context,
    uint32_t polls_between_pumps)
{
    TEST_CHECK(app != NULL);
    TEST_CHECK(job != NULL);
    TEST_CHECK(adapter != NULL);
    TEST_CHECK(network_pump == qweb_board_xilinx_lwip_pump);
    TEST_CHECK(network_context != NULL);
    TEST_CHECK(polls_between_pumps == 1024u);
    ++test_board_init_calls;
    return QWEB_BOARD_OK;
}

int qweb_board_app_step(qweb_board_app_t *app)
{
    TEST_CHECK(app != NULL);
    return test_board_step_result;
}

int qweb_board_qot_runner(
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
    (void)result;
    return QOT_OK;
}

void qweb_board_xilinx_lwip_pump(void *context)
{
    (void)context;
}

int qweb_job_init(
    qweb_job_t *job,
    uintptr_t qot_base,
    const qot_run_config_t *config,
    const qtk_asset_t *tokenizer,
    qot_session_runner_fn runner,
    void *runner_context,
    uint32_t max_polls_per_token)
{
    TEST_CHECK(job != NULL);
    TEST_CHECK(qot_base == (uintptr_t)UINT64_C(0xA0040000));
    TEST_CHECK(config != NULL);
    TEST_CHECK(config->layer_count == QOT_MODEL_LAYER_COUNT);
    TEST_CHECK(tokenizer != NULL);
    TEST_CHECK(runner == qweb_board_qot_runner);
    TEST_CHECK(runner_context != NULL);
    TEST_CHECK(max_polls_per_token == UINT32_C(200000000));
    test_job_pointer = job;
    ++test_job_init_calls;
    return QWEB_JOB_OK;
}

int qweb_router_init(qweb_router_t *router, qweb_job_t *job)
{
    TEST_CHECK(router != NULL);
    TEST_CHECK(job == test_job_pointer);
    ++test_router_init_calls;
    return QWEB_ROUTER_OK;
}

int qweb_lwip_adapter_init(
    qweb_lwip_adapter_t *adapter,
    qweb_router_t *router,
    const qweb_lwip_config_t *config)
{
    TEST_CHECK(adapter != NULL);
    TEST_CHECK(router != NULL);
    TEST_CHECK(config != NULL);
    test_network_config = *config;
    ++test_adapter_init_calls;
    return QWEB_LWIP_OK;
}

int qweb_lwip_adapter_start(qweb_lwip_adapter_t *adapter)
{
    TEST_CHECK(adapter != NULL);
    ++test_adapter_start_calls;
    return test_adapter_start_result;
}

void qweb_lwip_adapter_stop(qweb_lwip_adapter_t *adapter)
{
    TEST_CHECK(adapter != NULL);
    ++test_adapter_stop_calls;
}

static void test_static_resolver_and_clock(void)
{
    qweb_lwip_static_asset_t asset;
    int rc;

    TEST_CHECK(test_network_config.listen_port == 80u);
    TEST_CHECK(test_network_config.request_timeout_ms == 5000u);
    TEST_CHECK(test_network_config.connection_timeout_ms == 30000u);
    TEST_CHECK(test_network_config.allowed_host_count == 2u);
    TEST_CHECK(strcmp(test_network_config.allowed_hosts[0],
                      "10.20.30.40") == 0);
    TEST_CHECK(strcmp(test_network_config.allowed_hosts[1],
                      "10.20.30.40:80") == 0);
    TEST_CHECK(test_network_config.now_ms != NULL);
    test_ticks = UINT64_C(123400000);
    TEST_CHECK(test_network_config.now_ms(
                   test_network_config.now_context) == 1234u);

    memset(&asset, 0, sizeof(asset));
    rc = test_network_config.resolve_asset(
        test_network_config.asset_context, "/", 1u, &asset);
    TEST_CHECK(rc == QWEB_LWIP_ASSET_HIT);
    TEST_CHECK(asset.content_type == QWEB_HTTP_CONTENT_HTML);
    TEST_CHECK(asset.body == test_index_body);
    TEST_CHECK(asset.body_length == sizeof(test_index_body) - 1u);
    TEST_CHECK(asset.etag_length == strlen(test_index_asset.etag));
    TEST_CHECK(asset.cache_max_age_seconds == 60u);
    rc = test_network_config.resolve_asset(
        test_network_config.asset_context, "/missing", 8u, &asset);
    TEST_CHECK(rc == QWEB_LWIP_ASSET_MISS);
}

int main(void)
{
    print_app_header();

    test_ddr_status = 0u;
    TEST_CHECK(start_application() == -1);
    TEST_CHECK(test_tokenizer_init_calls == 0u);

    test_ddr_status = TEST_DDR_READY;
    TEST_CHECK(start_application() == -1);
    TEST_CHECK(test_tokenizer_init_calls == 0u);

    test_set_ipv4(10u, 20u, 30u, 40u);
    test_adapter_start_result = QWEB_LWIP_ERR_LWIP;
    TEST_CHECK(start_application() == -1);
    TEST_CHECK(test_adapter_stop_calls == 1u);

    test_adapter_start_result = QWEB_LWIP_OK;
    TEST_CHECK(start_application() == 0);
    TEST_CHECK(start_application() == -1);
    TEST_CHECK(test_tokenizer_init_calls == 2u);
    TEST_CHECK(test_board_init_calls == 2u);
    TEST_CHECK(test_job_init_calls == 2u);
    TEST_CHECK(test_router_init_calls == 2u);
    TEST_CHECK(test_adapter_init_calls == 2u);
    TEST_CHECK(test_adapter_start_calls == 2u);
    test_static_resolver_and_clock();

    test_board_step_result = QWEB_BOARD_OK;
    TEST_CHECK(transfer_data() == 0);
    TEST_CHECK(test_job_pointer != NULL);
    test_job_pointer->job_id = 7u;
    test_job_pointer->error_code = -11;
    test_job_pointer->session_error_code = -12;
    test_board_step_result = QWEB_BOARD_ERR_JOB;
    TEST_CHECK(transfer_data() == 0);
    test_board_step_result = QWEB_BOARD_ERR_STATE;
    TEST_CHECK(transfer_data() == -1);

    (void)printf("PASS qweb board entry host assembly test\n");
    return EXIT_SUCCESS;
}
