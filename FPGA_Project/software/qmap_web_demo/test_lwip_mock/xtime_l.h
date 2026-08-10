#ifndef TEST_XTIME_L_H
#define TEST_XTIME_L_H

#include <stdint.h>

typedef uint64_t XTime;

#define COUNTS_PER_SECOND UINT64_C(100000000)

void XTime_GetTime(XTime *ticks);

#endif /* TEST_XTIME_L_H */
