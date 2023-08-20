#include <math.h>
#include "mempool.h"

void print_list_subblock(MemoryBlock_ptr memblock, uint32_t i)
{

    printf("freelist[%d]: \n", i);

    uintptr_t *p = (uintptr_t *)&(*(memblock->freelist[i]));
    while (*p != 0)
    {
        printf("    0x%08lx, 0x%08lx\n", (uintptr_t)*p, (uintptr_t)*p - (uintptr_t)memblock->m_subblock_array);
        p = (uintptr_t *)*p;
    }
}

uint32_t get_subblock_index(const MemoryBlock_ptr memblock_array_start, SubBlock_ptr subblock_address)
{
    return (uint32_t)((uintptr_t)subblock_address - (uintptr_t)memblock_array_start);
}

SubBlock_t *subblock_malloc(uint32_t size, MemoryBlock_ptr MEMBLOCK)
{
    printf("\n[call] : subblock_malloc\n");
    printf("\t[info] MemoryBlock addr = %p\n", MEMBLOCK);

    SubBlock_ptr subblock;
    MemoryBlock_ptr memblock;
    uint32_t i, order = 0;

    size = ALIGN_SIZE(size);

    // one more byte for storing order
    while (BLOCKSIZE * i < size + 1)
    {
        i++;
    }

    order = (i < MIN_ORDER) ? MIN_ORDER : (i > MAX_ORDER ? MAX_ORDER : i);

    printf("\ti = %d, order = 2^%d\n", i, order);

    // level up until non-null list found
    for (;; i++)
    {
        if (i > MAX_ORDER)
            return NULL;
        if (MEMBLOCK->freelist[i])
            printf("MEMBLOCK->freelist[%d] = %p\n", i, (uintptr_t *)MEMBLOCK->freelist[i]);
        break;
    }

    // remove the block out of list
    subblock = (SubBlock_ptr)MEMBLOCK->freelist[i];
    MEMBLOCK->freelist[i] = *(SubBlock_ptr *)MEMBLOCK->freelist[i];

    // split until i == order
    while (i-- > order)
    {
        memblock = (MemoryBlock_ptr)MEMBLOCKOF(subblock, i, MEMBLOCK);
        memblock->freelist[i] = (SubBlock_ptr)memblock;
    }

    // Align the starting address of the block
    subblock = ALIGN_ADDR(subblock);

    // store order in previous byte
    *((uint8_t *)(subblock - 1)) = order;

    subblock->m_size = size + 1;
    subblock->m_ID = get_subblock_index(MEMBLOCK, subblock);

    printf("\tSubBlock allocated at %p with size %zu and ID %d\n", subblock, subblock->m_size, subblock->m_ID);

    return subblock;
}

// Merge two SubBlocks in a common MemoryBlock
void merge_subblocks(MemoryBlock_ptr memblock, SubBlock_ptr subblock1, SubBlock_ptr subblock2)
{
    // Ensure both SubBlocks are adjacent
    uintptr_t distance = (uintptr_t)subblock2 - (uintptr_t)subblock1;
    if (distance != subblock1->m_size && distance != subblock2->m_size)
    {
        printf("[Error] SubBlocks are not adjacent and cannot be merged.\n");
        return;
    }

    // Merge the SubBlocks
    subblock1->m_size *= 2;
    subblock2->m_size = 0; // Mark the second SubBlock as empty
}

// Remove a SubBlock
void remove_subblock(MemoryBlock_ptr memblock, SubBlock_ptr subblock)
{
    printf("\t[call] : remove_subblock\n");
    // Check if its buddy is also free and merge them
    uint32_t subblock_size = (uint32_t)log2((double)subblock->m_size); // No loss, subblocksize are always round.
    uintptr_t buddy_address = (uintptr_t)MEMBLOCKOF(subblock, subblock_size, memblock);
    SubBlock_ptr buddy = (SubBlock_ptr)buddy_address;

    printf("subblock address : %ld", (uintptr_t)subblock);
    printf("buddy_address : %ld", (uintptr_t)buddy);

    if (buddy->m_size == subblock->m_size)
    {
        merge_subblocks(memblock, subblock, buddy);
    }
    // Mark the SubBlock as free
    subblock->m_size = 0;
}

// Optimize the layout within the MemoryBlock after deletion
void optimize_layout(MemoryBlock_ptr memblock)
{
    // Iterate through the MemoryBlock and try to merge adjacent free SubBlocks
    uintptr_t current_address = (uintptr_t)memblock->m_subblock_array;
    uintptr_t end_address = current_address + BLOCKSIZE;

    while (current_address < end_address)
    {
        SubBlock_ptr current_subblock = (SubBlock_ptr)current_address;
        if (current_subblock->m_size == 0)
        {
            current_address += sizeof(SubBlock_ptr); // Move to the next SubBlock
            continue;
        }

        uintptr_t next_address = current_address + current_subblock->m_size;
        SubBlock_ptr next_subblock = (SubBlock_ptr)next_address;

        // If the next SubBlock is free and of the same size, merge them
        if (next_subblock->m_size == current_subblock->m_size)
        {
            merge_subblocks(memblock, current_subblock, next_subblock);
        }
        else
        {
            current_address = next_address;
        }
    }
}

void subblock_free_all(MemoryBlock_ptr memblock)
{
    printf("\n[call] : subblock_free_all\n");

    if (memblock->m_subblock_array != NULL)
    {
        // Check if the MemoryBlock pointer is NULL
        if (memblock == NULL)
        {
            printf("[Error] NULL MemoryBlock pointer\n");
            return;
        }

        // Loop through the entire memory block
        uintptr_t current_address = (uintptr_t)memblock->m_subblock_array;
        uintptr_t end_address = current_address + BLOCKSIZE;

        // Fetch order in previous byte
        uint32_t i = *((uint8_t *)(current_address - 1));

        printf("\t[Info] blocksize from address = %lu\n", (unsigned long)(end_address - current_address));

        while (current_address < end_address)
        {
            // Get the current subblock
            SubBlock_ptr current_subblock = (SubBlock_ptr)current_address;

            printf("\t\tfreeing subblock %p \n", current_subblock);

            // Remove the current subblock
            remove_subblock(memblock, current_subblock);

            current_address += (1 << i);
        }

        // Optimize the layout within the MemoryBlock after deletion
        optimize_layout(memblock);
        printf("\t[Info] subblock freed\n");
    }
    else
    {
        printf("\t[Info] no subblock to free\n");
    }
}