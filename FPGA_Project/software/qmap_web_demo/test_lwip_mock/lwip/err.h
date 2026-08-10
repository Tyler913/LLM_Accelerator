#ifndef TEST_LWIP_ERR_H
#define TEST_LWIP_ERR_H

#include <stdint.h>

typedef int err_t;
typedef uint8_t u8_t;
typedef uint16_t u16_t;
typedef uint32_t u32_t;

#define ERR_OK 0
#define ERR_MEM -1
#define ERR_ABRT -2
#define ERR_RST -3
#define ERR_ARG -4

#endif /* TEST_LWIP_ERR_H */
