#ifndef QWEB_BOARD_ENTRY_H
#define QWEB_BOARD_ENTRY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Entry points required by AMD's standalone lwip_echo_server template main. */
void print_app_header(void);
int start_application(void);
int transfer_data(void);

#ifdef __cplusplus
}
#endif

#endif /* QWEB_BOARD_ENTRY_H */
