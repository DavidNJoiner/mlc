#include "deep_time.h"

/*  -------------------------------------------------------*/
/*  Monotonic Chrono                                       */
/*  -------------------------------------------------------*/
uint64_t nanos()
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

char *get_time()
{
    static char buffer[80]; // Static so it retains its value after the function returns
    time_t rawtime;
    struct tm *timeinfo;

    time(&rawtime);                 // Get the current time
    timeinfo = localtime(&rawtime); // Convert to local time format

    // Format the time into the buffer. Here, we're using the format "HH:MM:SS"
    strftime(buffer, sizeof(buffer), "%H:%M:%S", timeinfo);

    return buffer;
}