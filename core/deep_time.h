// Header protection
#ifndef DEEP_TIME_H
#define DEEP_TIME_H

#include <stdint.h>
#include <time.h>

#ifdef DEEPC_WINDOWS
#include <windows.h>
#endif //DEEPC_WINDOWS 

uint64_t nanos();
char *get_time();

#endif // DEEP_TIME_H
