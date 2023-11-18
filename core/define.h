#ifndef _DEFINE_H_
#define _DEFINE_H_

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdalign.h>
#include <math.h>

// Properly define static assertions.
#if defined(__clang__) || defined(__GNUC__)
/** @brief Static assertion */
#define STATIC_ASSERT _Static_assert
#else

/** @brief Static assertion */
#define STATIC_ASSERT static_assert
#endif

// Ensure all types are of the correct size.

/** @brief Assert uint8_t to be 1 byte.*/
STATIC_ASSERT(sizeof(uint8_t) == 1, "Expected uint8_t to be 1 byte.");

/** @brief Assert uint16_t to be 2 bytes.*/
STATIC_ASSERT(sizeof(uint16_t) == 2, "Expected uint16_t to be 2 bytes.");

/** @brief Assert uint32_t to be 4 bytes.*/
STATIC_ASSERT(sizeof(uint32_t) == 4, "Expected uint32_t to be 4 bytes.");

/** @brief Assert uint64_t to be 8 bytes.*/
STATIC_ASSERT(sizeof(uint64_t) == 8, "Expected uint64_t to be 8 bytes.");

/** @brief Assert int8_t to be 1 byte.*/
STATIC_ASSERT(sizeof(int8_t) == 1, "Expected int8_t to be 1 byte.");

/** @brief Assert int16_t to be 2 bytes.*/
STATIC_ASSERT(sizeof(int16_t) == 2, "Expected int16_t to be 2 bytes.");

/** @brief Assert int32_t to be 4 bytes.*/
STATIC_ASSERT(sizeof(int32_t) == 4, "Expected int32_t to be 4 bytes.");

/** @brief Assert int64_t to be 8 bytes.*/
STATIC_ASSERT(sizeof(int64_t) == 8, "Expected int64_t to be 8 bytes.");

/** @brief Assert float to be 4 bytes.*/
STATIC_ASSERT(sizeof(float) == 4, "Expected float to be 4 bytes.");

/** @brief Assert double to be 8 bytes.*/
STATIC_ASSERT(sizeof(double) == 8, "Expected double to be 8 bytes.");

/** @brief True.*/
#define true 1

/** @brief False. */
#define false 0

#define INITIAL_POOL_BLOCKS 1
#define MAX_POOL_INSTANCES 1
#define DEEPC_MIN_BLOCK_SIZE sizeof(uint32_t)
#define DEEPC_VOID_POINTER void*
#define DEEPC_SIZE_OF_VOID_POINTER sizeof(DEEPC_VOID_POINTER)

// Pool define
#define MAX_ORDER 10 // 2 ** 10 == 1024 bytes
#define MIN_ORDER 4  // 2 ** 4 == 16 bytes
/* the order ranges 0..MAX_ORDER, the largest subblock is 2**(MAX_ORDER) */

// TODO : avoid hardcoding memblock size
#if defined(_M_X64) || defined(__amd64__)
#define BLOCKSIZE 1216
#else
#define BLOCKSIZE 1072
#endif

/* the address of the memoryblock of a subblock from freelists[i]. */

#define _MEMBASE(MEMBLOCK) ((uintptr_t)(MEMBLOCK)->m_subblock_array)
#define _OFFSET(b, MEMBLOCK) ((uintptr_t)(b)-_MEMBASE(MEMBLOCK))
#define _MEMBLOCKOF(b, i, MEMBLOCK) (_OFFSET(b, MEMBLOCK) ^ ((uint32_t)1 << (i)))
#define MEMBLOCKOF(b, i, MEMBLOCK) ((DEEPC_VOID_POINTER)(_MEMBLOCKOF(b, i, MEMBLOCK) + _MEMBASE(MEMBLOCK)))

// Ensure that each subblock is aligned to a multiple of the machine's word size.
#define ALIGN_SIZE(size) (((size) + DEEPC_SIZE_OF_VOID_POINTER - 1) & ~(DEEPC_SIZE_OF_VOID_POINTER - 1))
#define ALIGN_ADDR(DEEPC_VOID_POINTER) ((void *)((uintptr_t)(DEEPC_VOID_POINTER + DEEPC_SIZE_OF_VOID_POINTER - 1) & ~(DEEPC_SIZE_OF_VOID_POINTER - 1)))

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#define DEEPC_WINDOWS true
#ifndef _WIN64
#error "64-bit is required on Windows!"
#endif
#elif defined(__linux__) || defined(__gnu_linux__)
// Linux OS
#define DEEPC_LINUX true

// Inlining
#if defined(__clang__) || defined(__gcc__)
/** @brief Inline qualifier */
#define DEEPC_INLINE __attribute__((always_inline)) inline

/** @brief No-inline qualifier */
#define DEEPC_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)

/** @brief Inline qualifier */
#define DEEPC_INLINE __forceinline

/** @brief No-inline qualifier */
#define DEEPC_NOINLINE __declspec(noinline)
#else

/** @brief Inline qualifier */
#define DEEPC_INLINE static inline

/** @brief No-inline qualifier */
#define DEEPC_NOINLINE
#endif

DEEPC_INLINE uint64_t get_aligned(uint64_t operand, uint64_t granularity)
{
    return ((operand + (granularity - 1)) & ~(granularity - 1));
}

/** @brief Gets the number of bytes from amount of gibibytes (GiB) (1024*1024*1024) */
#define GIBIBYTES(amount) ((amount) * 1024ULL * 1024ULL * 1024ULL)
/** @brief Gets the number of bytes from amount of mebibytes (MiB) (1024*1024) */
#define MEBIBYTES(amount) ((amount) * 1024ULL * 1024ULL)
/** @brief Gets the number of bytes from amount of kibibytes (KiB) (1024) */
#define KIBIBYTES(amount) ((amount) * 1024ULL)

/** @brief Gets the number of bytes from amount of gigabytes (GB) (1000*1000*1000) */
#define GIGABYTES(amount) ((amount) * 1000ULL * 1000ULL * 1000ULL)
/** @brief Gets the number of bytes from amount of megabytes (MB) (1000*1000) */
#define MEGABYTES(amount) ((amount) * 1000ULL * 1000ULL)
/** @brief Gets the number of bytes from amount of kilobytes (KB) (1000) */
#define KILOBYTES(amount) ((amount) * 1000ULL)

#define DEEPC_MIN(x, y) (x < y ? x : y)
#define DEEPC_MAX(x, y) (x > y ? x : y)
#define LOG2(x) (log(x) / log(2.0))

#endif
#endif // _DEFINE_H_
