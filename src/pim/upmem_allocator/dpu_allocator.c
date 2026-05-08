#include <mram.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common.h"

/*
 * Lightweight MRAM allocator scaffold.
 *
 * Design notes:
 * - Metadata remains in WRAM.
 * - Allocation granularity is 1 KB by default; callers should align model
 *   config to the same slot size before invoking.
 * - `mem_compact()` assumes the host has quiesced normal decode work and will
 *   consume the remap table before issuing any further accesses. In a real
 *   deployment this should be guarded by a host-visible stop-the-world command
 *   and followed by pointer-table refresh on the host side.
 */

#define CLOVER_MRAM_POOL_BYTES (64u * 1024u * 1024u)
#define CLOVER_SLOT_BYTES CLOVER_ALLOCATOR_ALIGNMENT_BYTES
#define CLOVER_TOTAL_SLOTS (CLOVER_MRAM_POOL_BYTES / CLOVER_SLOT_BYTES)
#define CLOVER_BITMAP_WORDS ((CLOVER_TOTAL_SLOTS + 31u) / 32u)

__mram_noinit uint8_t clover_mram_pool[CLOVER_MRAM_POOL_BYTES];

static uint32_t g_bitmap[CLOVER_BITMAP_WORDS];
static clover_alloc_record_t g_records[CLOVER_ALLOCATOR_MAX_ALLOCS];
static uint32_t g_next_fit_slot = 0u;

static uint32_t bytes_to_slots(size_t size_bytes)
{
    uint32_t size_u32 = (uint32_t)size_bytes;
    return (size_u32 + CLOVER_SLOT_BYTES - 1u) / CLOVER_SLOT_BYTES;
}

static int bitmap_test(uint32_t slot_idx)
{
    return (g_bitmap[slot_idx / 32u] >> (slot_idx % 32u)) & 1u;
}

static void bitmap_set_range(uint32_t start_slot, uint32_t slot_count, int value)
{
    for (uint32_t slot = start_slot; slot < start_slot + slot_count; ++slot) {
        uint32_t *word = &g_bitmap[slot / 32u];
        uint32_t mask = 1u << (slot % 32u);
        if (value) {
            *word |= mask;
        } else {
            *word &= ~mask;
        }
    }
}

static clover_alloc_record_t *find_record_by_ptr(void *ptr)
{
    uintptr_t ptr_value = (uintptr_t)ptr;
    for (uint32_t idx = 0; idx < CLOVER_ALLOCATOR_MAX_ALLOCS; ++idx) {
        if (g_records[idx].in_use && g_records[idx].ptr_bytes == (uint32_t)ptr_value) {
            return &g_records[idx];
        }
    }
    return NULL;
}

static clover_alloc_record_t *reserve_record(void)
{
    for (uint32_t idx = 0; idx < CLOVER_ALLOCATOR_MAX_ALLOCS; ++idx) {
        if (!g_records[idx].in_use) {
            g_records[idx].in_use = 1u;
            return &g_records[idx];
        }
    }
    return NULL;
}

void mem_init(void)
{
    memset(g_bitmap, 0, sizeof(g_bitmap));
    memset(g_records, 0, sizeof(g_records));
    g_next_fit_slot = 0u;
}

void *mem_alloc(size_t size)
{
    uint32_t slots_needed = bytes_to_slots(size);
    if (slots_needed == 0u || slots_needed > CLOVER_TOTAL_SLOTS) {
        return NULL;
    }

    uint32_t scanned = 0u;
    uint32_t start = g_next_fit_slot;
    while (scanned + slots_needed <= CLOVER_TOTAL_SLOTS) {
        uint32_t probe = (start + scanned) % CLOVER_TOTAL_SLOTS;
        uint32_t run = 0u;
        while (probe + run < CLOVER_TOTAL_SLOTS && run < slots_needed && !bitmap_test(probe + run)) {
            ++run;
        }
        if (run == slots_needed) {
            clover_alloc_record_t *record = reserve_record();
            if (record == NULL) {
                return NULL;
            }
            bitmap_set_range(probe, slots_needed, 1);
            record->ptr_bytes = probe * CLOVER_SLOT_BYTES;
            record->size_bytes = slots_needed * CLOVER_SLOT_BYTES;
            g_next_fit_slot = probe + slots_needed;
            return (void *)(uintptr_t)(record->ptr_bytes);
        }
        scanned += (run > 0u) ? run + 1u : 1u;
    }
    return NULL;
}

void mem_free(void *ptr)
{
    clover_alloc_record_t *record = find_record_by_ptr(ptr);
    if (record == NULL) {
        return;
    }
    uint32_t start_slot = record->ptr_bytes / CLOVER_SLOT_BYTES;
    uint32_t slot_count = bytes_to_slots(record->size_bytes);
    bitmap_set_range(start_slot, slot_count, 0);
    memset(record, 0, sizeof(*record));
    if (start_slot < g_next_fit_slot) {
        g_next_fit_slot = start_slot;
    }
}

void mem_compact(uint32_t *remap_table, uint32_t max_entries)
{
    uint8_t buffer[CLOVER_SLOT_BYTES];
    uint32_t compact_slot = 0u;
    uint32_t remap_entries = 0u;

    /*
     * Safety contract:
     * 1. Host pauses normal decode and DMA traffic for this DPU.
     * 2. Host triggers compaction and waits for completion.
     * 3. DPU copies each live block downward using an intermediate WRAM buffer.
     * 4. DPU writes `(old_ptr, new_ptr, size_bytes)` triples to `remap_table`.
     * 5. Host refreshes every pointer table that can reference moved blocks.
     * 6. Host resumes decode only after all remaps are applied.
     */
    memset(g_bitmap, 0, sizeof(g_bitmap));
    for (uint32_t idx = 0u; idx < CLOVER_ALLOCATOR_MAX_ALLOCS; ++idx) {
        clover_alloc_record_t *record = &g_records[idx];
        if (!record->in_use) {
            continue;
        }
        uint32_t old_slot = record->ptr_bytes / CLOVER_SLOT_BYTES;
        uint32_t slot_count = bytes_to_slots(record->size_bytes);
        if (old_slot != compact_slot) {
            for (uint32_t slot = 0u; slot < slot_count; ++slot) {
                mram_read(&clover_mram_pool[(old_slot + slot) * CLOVER_SLOT_BYTES], buffer, CLOVER_SLOT_BYTES);
                mram_write(buffer, &clover_mram_pool[(compact_slot + slot) * CLOVER_SLOT_BYTES], CLOVER_SLOT_BYTES);
            }
            if (remap_table != NULL && remap_entries + CLOVER_ALLOCATOR_REMAP_STRIDE <= max_entries) {
                remap_table[remap_entries++] = record->ptr_bytes;
                remap_table[remap_entries++] = compact_slot * CLOVER_SLOT_BYTES;
                remap_table[remap_entries++] = record->size_bytes;
            }
            record->ptr_bytes = compact_slot * CLOVER_SLOT_BYTES;
        }
        bitmap_set_range(compact_slot, slot_count, 1);
        compact_slot += slot_count;
    }
    g_next_fit_slot = compact_slot;
}

/*
 * Unit test ideas:
 * - allocate 3 small blocks, free the middle one, ensure next-fit reuses gap
 * - allocate until full, verify NULL on overflow
 * - free alternating blocks, run mem_compact, verify remap table is populated
 * - allocate after compaction, verify new block lands at compacted tail
 */
