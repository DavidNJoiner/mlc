#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include <stdio.h>

/**
 * This module provides an interface to manage and monitor various metrics across the application.
 */

typedef struct StateManager
{
    int total_bytes_allocated;
} StateManager;

// total_bytes_allocated abstract interface for users:
void increase_total_bytes_allocated(int bytes);
void decrease_total_bytes_allocated(int bytes);
int get_total_bytes_allocated(void);

#endif // STATE_MANAGER_H
