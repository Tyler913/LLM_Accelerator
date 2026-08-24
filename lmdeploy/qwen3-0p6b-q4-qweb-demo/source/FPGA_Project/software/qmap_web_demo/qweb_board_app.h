#ifndef QWEB_BOARD_APP_H

#define QWEB_BOARD_APP_H

#include <stdint.h>

#include "qweb_job.h"
#include "qweb_lwip_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QWEB_BOARD_DEFAULT_POLLS_BETWEEN_PUMPS 1024u

enum {
    QWEB_BOARD_OK = 0,
    QWEB_BOARD_ERR_NULL = -1,
    QWEB_BOARD_ERR_CONFIG = -2,
    QWEB_BOARD_ERR_STATE = -3,
    QWEB_BOARD_ERR_JOB = -4
};

typedef void (*qweb_board_network_pump_fn)(void *context);

/*
 * Initialize this small coordinator before passing qweb_board_qot_runner and
 * its address to qweb_job_init(). The pointed-to job and adapter remain owned
 * by caller-provided static storage.
 */
typedef struct qweb_board_app {
    qweb_job_t *job;
    qweb_lwip_adapter_t *adapter;
    qweb_board_network_pump_fn network_pump;
    void *network_context;
    uint32_t polls_between_pumps;
    uint8_t initialized;
} qweb_board_app_t;

int qweb_board_app_init(
    qweb_board_app_t *app,
    qweb_job_t *job,
    qweb_lwip_adapter_t *adapter,
    qweb_board_network_pump_fn network_pump,
    void *network_context,
    uint32_t polls_between_pumps);

/*
 * One bare-metal main-loop turn: pump lwIP, service deadlines/TX, then advance
 * at most one complete model-token job step. This function, never a raw recv
 * callback, owns calls to qweb_job_step().
 */
int qweb_board_app_step(qweb_board_app_t *app);

/*
 * qot_session runner that keeps the network alive while polling PL. It pumps
 * at a configured bounded poll interval and also before and after MMIO setup.
 */
int qweb_board_qot_runner(
    void *runner_context,
    uintptr_t base,
    const qot_run_config_t *config,
    uint32_t input_token_id,
    uint32_t position,
    uint32_t max_polls,
    qot_result_t *result);

struct netif;

typedef struct qweb_board_xilinx_network {
    struct netif *network_interface;
} qweb_board_xilinx_network_t;

/*
 * Concrete pump for the Vitis standalone lwip_echo_server platform. It services
 * TcpFastTmrFlag, TcpSlowTmrFlag, and xemacif_input(). Use this as the app's
 * network_pump; qweb_board_qot_runner then performs the same work during every
 * long PL token poll. The template main/entry file remains app-owned.
 */
void qweb_board_xilinx_lwip_pump(void *context);

#ifdef __cplusplus
}
#endif

#endif /* QWEB_BOARD_APP_H */
