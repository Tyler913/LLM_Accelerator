#include "qweb_board_app.h"

#include <string.h>

#include "netif/xadapter.h"

extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;
void tcp_fasttmr(void);
void tcp_slowtmr(void);

static void qweb_board_pump_once(qweb_board_app_t *app)
{
    app->network_pump(app->network_context);
    (void)qweb_lwip_adapter_service(app->adapter);
}

static int qweb_board_read_validated_result(
    uintptr_t base,
    const qot_run_config_t *config,
    qot_result_t *result)
{
    uint32_t expected_mask;
    int rc = qot_read_result(base, result);

    if (rc != QOT_OK) return rc;
    if (config->layer_count >= 32u) {
        expected_mask = UINT32_MAX;
    } else {
        expected_mask = ((UINT32_C(1) << config->layer_count) - UINT32_C(1))
                        << config->layer_start;
    }
    if ((result->status & QOT_STATUS_DONE_STICKY_MASK) == 0u ||
        QOT_STATUS_HAS_ANY_ERROR(result->status) ||
        result->layers_started != config->layer_count ||
        result->layers_completed != config->layer_count ||
        result->layer_done_mask != expected_mask ||
        result->layer_error_mask != 0u) {
        return QOT_ERR_STATUS;
    }
    return QOT_OK;
}

int qweb_board_app_init(
    qweb_board_app_t *app,
    qweb_job_t *job,
    qweb_lwip_adapter_t *adapter,
    qweb_board_network_pump_fn network_pump,
    void *network_context,
    uint32_t polls_between_pumps)
{
    if (app == NULL || job == NULL || adapter == NULL ||
        network_pump == NULL) {
        return QWEB_BOARD_ERR_NULL;
    }
    if (polls_between_pumps == 0u) {
        polls_between_pumps = QWEB_BOARD_DEFAULT_POLLS_BETWEEN_PUMPS;
    }
    memset(app, 0, sizeof(*app));
    app->job = job;
    app->adapter = adapter;
    app->network_pump = network_pump;
    app->network_context = network_context;
    app->polls_between_pumps = polls_between_pumps;
    app->initialized = 1u;
    return QWEB_BOARD_OK;
}

int qweb_board_app_step(qweb_board_app_t *app)
{
    qot_session_event_t event;
    int rc;

    if (app == NULL) return QWEB_BOARD_ERR_NULL;
    if (app->initialized == 0u || app->job == NULL ||
        app->adapter == NULL || app->network_pump == NULL ||
        app->polls_between_pumps == 0u) {
        return QWEB_BOARD_ERR_STATE;
    }
    qweb_board_pump_once(app);
    if (!qweb_job_is_active(app->job)) return QWEB_BOARD_OK;

    rc = qweb_job_step(app->job, &event);
    return rc == QWEB_JOB_OK ? QWEB_BOARD_OK : QWEB_BOARD_ERR_JOB;
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
    qweb_board_app_t *app = (qweb_board_app_t *)runner_context;
    qot_run_config_t token_config;
    uint32_t polls = 0u;
    uint32_t polls_since_pump = 0u;
    int rc;

    if (app == NULL || config == NULL || result == NULL) return QOT_ERR_NULL;
    if (app->initialized == 0u || app->network_pump == NULL ||
        app->adapter == NULL || app->polls_between_pumps == 0u) {
        return QOT_SESSION_ERR_STATE;
    }

    qweb_board_pump_once(app);
    token_config = *config;
    token_config.input_token_id = input_token_id;
    token_config.position = position;
    rc = qot_configure_run(base, &token_config);
    if (rc != QOT_OK) return rc;
    qweb_board_pump_once(app);
    rc = qot_start(base);
    if (rc != QOT_OK) return rc;

    for (;;) {
        uint32_t status = qot_read32(base, QOT_REG_STATUS);

        if (QOT_STATUS_HAS_ANY_ERROR(status)) {
            (void)qot_read_result(base, result);
            return QOT_ERR_STATUS;
        }
        if ((status & QOT_STATUS_DONE_STICKY_MASK) != 0u) {
            return qweb_board_read_validated_result(base,
                                                    &token_config,
                                                    result);
        }

        ++polls;
        ++polls_since_pump;
        if (polls_since_pump >= app->polls_between_pumps) {
            qweb_board_pump_once(app);
            polls_since_pump = 0u;
        }
        if (max_polls != 0u && polls >= max_polls) {
            (void)qot_read_result(base, result);
            return QOT_ERR_TIMEOUT;
        }
    }
}

void qweb_board_xilinx_lwip_pump(void *context)
{
    qweb_board_xilinx_network_t *network =
        (qweb_board_xilinx_network_t *)context;

    if (TcpFastTmrFlag != 0) {
        tcp_fasttmr();
        TcpFastTmrFlag = 0;
    }
    if (TcpSlowTmrFlag != 0) {
        tcp_slowtmr();
        TcpSlowTmrFlag = 0;
    }
    if (network != NULL && network->network_interface != NULL) {
        (void)xemacif_input(network->network_interface);
    }
}
