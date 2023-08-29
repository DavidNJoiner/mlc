#include "state_manager.h"

// Private function to get a singleton instance of the StateManager.
static StateManager *StateManager_getInstance()
{
    static StateManager instance = {0}; // Initialize all members to zero
    return &instance;
}

void increase_total_bytes_allocated(int bytes)
{
    StateManager *mm = StateManager_getInstance();
    mm->total_bytes_allocated += bytes;

    // TODO: Add alerting functionality if total_bytes_allocated exceeds a certain threshold to update the memory visualization.
}

void decrease_total_bytes_allocated(int bytes)
{
    printf("\t\t[State Manager] Freeing %d bytes from pool\n", bytes);

    StateManager *mm = StateManager_getInstance();
    if ((get_total_bytes_allocated() - bytes) < 0)
    {
        printf("\t\t[State Manager] Pool is empty\n");
        mm->total_bytes_allocated = 0;
    }
    else
    {
        mm->total_bytes_allocated -= bytes;
    }
}

int get_total_bytes_allocated(void)
{
    StateManager *mm = StateManager_getInstance();
    return mm->total_bytes_allocated;
}
