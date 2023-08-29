#include <math.h>
#include "mempool.h"

void print_list_subblock(MemoryBlock_t *memblock, uint32_t i)
{

    printf("freelist[%d]: \n", i);

    uintptr_t *p = (uintptr_t *)&(*(memblock->freelist[i]));
    while (*p != 0)
    {
        printf("    0x%08lx, 0x%08lx\n", (uintptr_t)*p, (uintptr_t)*p - (uintptr_t)memblock->m_subblock_array);
        p = (uintptr_t *)*p;
    }
}

uint32_t get_subblock_index(const MemoryBlock_t *memblock_array_start, SubBlock_t *subblock_address)
{
    return (uint32_t)((uintptr_t)subblock_address - (uintptr_t)memblock_array_start);
}

SubBlock_t *subblock_alloc(uint32_t size, MemoryBlock_t *MEMBLOCK)
{
    printf("\033[0;37m[Call] subblock_alloc\033[0m\n");
    printf("\t\033[34m[Info]\033[0m MemoryBlock addr = %p\n", (uintptr_t *)MEMBLOCK);

    SubBlock_t *subblock;
    MemoryBlock_t *memblock;
    uint32_t i, order = 0;

    size = ALIGN_SIZE(size);

    // one more byte for storing order
    while (BLOCKSIZE * i < size + 1)
    {
        i++;
    }

    order = (i < MIN_ORDER) ? MIN_ORDER : (i > MAX_ORDER ? MAX_ORDER : i);

    printf("\t\033[0;32m[Debug]\033[0m order = 2^%d = %f\n", order, pow(2, order));
    printf("\t\033[0;32m[Debug]\033[0m Current i value: %d\n", i);
    printf("\t\033[0;32m[Debug]\033[0mfreelist[%d] address: %p\n", i, (void *)MEMBLOCK->freelist[i]);

    // level up until non-null list found
    for (;; i++)
    {
        if (i > MAX_ORDER)
            return NULL;
        if (*(uintptr_t *)MEMBLOCK->freelist[i])
            printf("\t\033[0;32m[Debug]\033[0m MEMBLOCK->freelist[%d] = %p\n", i, (uintptr_t *)MEMBLOCK->freelist[i]);
        break;
    }

    // remove the block out of list
    subblock = (SubBlock_t *)MEMBLOCK->freelist[i];
    MEMBLOCK->freelist[i] = *(SubBlock_t **)MEMBLOCK->freelist[i];

    // split until i == order
    while (i-- > order)
    {
        printf("order = %d\n", i);
        memblock = (MemoryBlock_t *)MEMBLOCKOF(subblock, i, MEMBLOCK);
        memblock->freelist[i] = (SubBlock_t *)memblock;
    }

    // Align the starting address of the block
    subblock = ALIGN_ADDR(subblock);

    // store order in previous byte
    *((uint8_t *)(subblock - 1)) = order;

    subblock->m_size = size + 1;
    subblock->m_ID = get_subblock_index(MEMBLOCK, subblock);

    increase_total_bytes_allocated(sizeof(SubBlock_t));
    // add_entry("subblock_alloc", 2, (double)(sizeof(SubBlock_t)), 0.00);

    printf("\t\033[34m[Info]\033[0m SubBlock allocated at %p with size %zu and ID %d\n", subblock, subblock->m_size, subblock->m_ID);

    return subblock;
}

// Merge two SubBlocks in a common MemoryBlock
void _subblock_merge_(MemoryBlock_t *memblock, SubBlock_t *subblock1, SubBlock_t *subblock2)
{
    // Ensure both SubBlocks are adjacent
    uintptr_t distance = (uintptr_t)subblock2 - (uintptr_t)subblock1;
    if (distance != subblock1->m_size && distance != subblock2->m_size)
    {
        printf("\t\033[0;31m[Error]\033[0m SubBlocks are not adjacent and cannot be merged.\n");
        return;
    }

    // Merge the SubBlocks
    subblock1->m_size *= 2;
    subblock2->m_size = 0; // Mark the second SubBlock as empty
}

// Remove a SubBlock
void _subblock_free_(MemoryBlock_t *memblock, SubBlock_t *subblock)
{
    printf("\033[0;37m[Call] _subblock_free_\033[0m\n");
    // Check if its buddy is also free and merge them
    uint32_t subblock_size = (uint32_t)log2((double)subblock->m_size); // No loss, subblocksize are always round.
    uintptr_t buddy_address = (uintptr_t)MEMBLOCKOF(subblock, subblock_size, memblock);
    SubBlock_t *buddy = (SubBlock_t *)buddy_address;

    printf("\t\033[34m[Info]\033[0m subblock address : %ld\n", (uintptr_t)subblock);
    printf("\t\033[34m[Info]\033[0m buddy_address : %ld\n", (uintptr_t)buddy);

    if (buddy->m_size == subblock->m_size)
    {
        _subblock_merge_(memblock, subblock, buddy);
    }
    // Mark the SubBlock as free
    subblock->m_size = 0;

    decrease_total_bytes_allocated(sizeof(SubBlock_t));
    add_entry("_subblock_free_", 2, 0.00, (double)(sizeof(SubBlock_t)));
}

// Optimize the layout within the MemoryBlock after deletion
void _subblock_coalescing_(MemoryBlock_t *memblock)
{
    // Iterate through the MemoryBlock and try to merge adjacent free SubBlocks
    uintptr_t current_address = (uintptr_t)memblock->m_subblock_array;
    uintptr_t end_address = current_address + BLOCKSIZE;

    while (current_address < end_address)
    {
        SubBlock_t *current_subblock = (SubBlock_t *)current_address;
        if (current_subblock->m_size == 0)
        {
            current_address += sizeof(SubBlock_t *); // Move to the next SubBlock
            continue;
        }

        uintptr_t next_address = current_address + current_subblock->m_size;
        SubBlock_t *next_subblock = (SubBlock_t *)next_address;

        // If the next SubBlock is free and of the same size, merge them
        if (next_subblock->m_size == current_subblock->m_size)
        {
            _subblock_merge_(memblock, current_subblock, next_subblock);
        }
        else
        {
            current_address = next_address;
        }
    }
}

void subblock_free_all(MemoryBlock_t *memblock)
{
    printf("\033[0;37m[Call] subblock_free_all\033[0m\n");

    if (memblock->m_subblock_array != NULL)
    {
        // Check if the MemoryBlock pointer is NULL
        if (memblock == NULL)
        {
            printf("\t\033[0;31m[Error]\033[0m NULL MemoryBlock pointer\n");
            return;
        }

        // Loop through the entire memory block
        uintptr_t current_address = (uintptr_t)memblock->m_subblock_array;
        uintptr_t end_address = current_address + BLOCKSIZE;

        // Fetch order in previous byte
        uint32_t i = *((uint8_t *)(current_address - 1));

        printf("\t\033[34m[Info]\033[0m blocksize from address = %lu\n", (unsigned long)(end_address - current_address));

        while (current_address < end_address)
        {
            // Get the current subblock
            SubBlock_t *current_subblock = (SubBlock_t *)current_address;

            printf("\t\tfreeing subblock %p \n", current_subblock);

            // Remove the current subblock
            _subblock_free_(memblock, current_subblock);

            current_address += (1 << i);
        }

        // Optimize the layout within the MemoryBlock after deletion
        _subblock_coalescing_(memblock);
        printf("\t\033[34m[Info]\033[0m subblock freed\n");
    }
    else
    {
        printf("\t\033[34m[Info]\033[0m no subblock to free\n");
    }
}