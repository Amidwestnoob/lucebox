// Regression for speculative K/V cleanup under KVFlash paging.
//
// Qwen35DFlashTarget::restore_kv() and rollback_to() zero the attention
// rows a rejected speculative span wrote (the maskless AR decode graph
// relies on unwritten rows being zero). In the dense layout the row index
// equals the logical position. Under KVFlash the pager maps logical
// positions to physical pool slots, so a logical-index memset would both
// MISS every dirty row and CLEAR unrelated live rows whose physical slot
// index happens to equal a dirty logical position.
//
// This test pins dflash_kv_clear_speculative_rows() — the shared cleanup
// restore_kv() and the rollback_to() truncation call — against a real
// KvFlashPager with a deliberately non-identity block order, on CPU and
// CUDA backends:
//   1. paged restore: each dirty logical position's PHYSICAL row is zeroed;
//   2. paged restore: the unrelated live rows at physical indices equal to
//      the dirty logical positions keep their bytes (the historical bug
//      cleared them);
//   3. every other resident row survives byte-identical;
//   4. non-resident positions are skipped without touching the pool;
//   5. dense restore/rollback truncation zeroes exactly [first, end).
#include "dflash_kv_restore.h"
#include "kvflash_pager.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using dflash::common::KvFlashConfig;
using dflash::common::KvFlashPager;
using dflash::common::dflash_kv_clear_speculative_rows;

constexpr int kHeadDim = 4;
constexpr int kHeads = 2;

float pattern(int tensor_id, int64_t head, int64_t row, int64_t col) {
    return (float)((tensor_id * 2 + head) * 100000 + row * 10 + col);
}

void fill_rows(ggml_tensor * t, int tensor_id) {
    const int64_t rows = t->ne[1];
    std::vector<float> buf((size_t)kHeadDim * rows * kHeads);
    for (int64_t h = 0; h < kHeads; ++h)
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t c = 0; c < kHeadDim; ++c)
                buf[(size_t)h * rows * kHeadDim + r * kHeadDim + c] =
                    pattern(tensor_id, h, r, c);
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(float));
}

// Returns the number of mismatching rows; `expect_zero` lists rows that
// must be fully zero, all other rows must hold the fill pattern.
int check_rows(ggml_tensor * t, int tensor_id,
               const std::vector<int64_t> & expect_zero, const char * what) {
    const int64_t rows = t->ne[1];
    std::vector<float> buf((size_t)kHeadDim * rows * kHeads);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    int bad = 0;
    for (int64_t h = 0; h < kHeads; ++h) {
        for (int64_t r = 0; r < rows; ++r) {
            const bool want_zero =
                std::find(expect_zero.begin(), expect_zero.end(), r) !=
                expect_zero.end();
            for (int64_t c = 0; c < kHeadDim; ++c) {
                const float got =
                    buf[(size_t)h * rows * kHeadDim + r * kHeadDim + c];
                const float want =
                    want_zero ? 0.0f : pattern(tensor_id, h, r, c);
                if (got != want) {
                    if (bad < 8) {
                        std::printf(
                            "  FAIL(%s): tensor=%d head=%lld row=%lld col=%lld "
                            "got=%.1f want=%.1f\n",
                            what, tensor_id, (long long)h, (long long)r,
                            (long long)c, got, want);
                    }
                    bad++;
                    break;
                }
            }
        }
    }
    return bad;
}

int run_paged(ggml_backend_t backend, const char * name) {
    // Pool geometry: 7 chunks of 64 = 448 slots (the pager minimum for
    // 1 sink + 4 tail chunks plus an evictable pair).
    KvFlashConfig cfg;
    cfg.chunk_tokens = 64;
    cfg.pool_tokens = 448;
    cfg.sink_chunks = 1;
    cfg.tail_window_chunks = 4;

    ggml_init_params ip{};
    ip.mem_size = 2 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                         kHeadDim, cfg.pool_tokens, kHeads);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                         kHeadDim, cfg.pool_tokens, kHeads);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::printf("[%s] FAIL: tensor alloc\n", name);
        ggml_free(ctx);
        return 1;
    }

    KvFlashPager pager;
    if (!pager.attach(cfg, {k}, {v})) {
        std::printf("[%s] FAIL: pager attach\n", name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }
    // Non-identity placement: chunk 0 gets block 6, chunk 1 gets block 5, …
    pager.set_block_order({6, 5, 4, 3, 2, 1, 0});
    if (!pager.alloc_span(0, cfg.pool_tokens)) {
        std::printf("[%s] FAIL: alloc_span\n", name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    // The rejected speculative span: logical positions [100, 104), written
    // by a verify and then rolled back by restore_kv().
    const int spec_begin = 100, spec_end = 104;
    std::vector<int64_t> dirty_slots;
    for (int pos = spec_begin; pos < spec_end; ++pos) {
        const int slot = pager.slot_of(pos);
        if (slot < 0) {
            std::printf("[%s] FAIL: precondition, pos %d not resident\n",
                        name, pos);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return 1;
        }
        dirty_slots.push_back(slot);
    }
    if (dirty_slots[0] == spec_begin) {
        std::printf("[%s] FAIL: precondition, mapping is identity "
                    "(slot_of(%d)=%d); the paged case would not differ "
                    "from dense\n", name, spec_begin, (int)dirty_slots[0]);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    fill_rows(k, 0);
    fill_rows(v, 1);

    // Restore-path cleanup for the rejected span, paged layout.
    dflash_kv_clear_speculative_rows({k}, {v}, &pager, spec_begin, spec_end);

    int failures = 0;
    failures += check_rows(k, 0, dirty_slots, "paged-restore K");
    failures += check_rows(v, 1, dirty_slots, "paged-restore V");
    // The rows the historical logical-index memset would have cleared are
    // physical slots 100..103 — live rows of OTHER logical positions. The
    // sweep above already proves they kept their bytes; make the canary
    // explicit for review clarity.
    for (int r = spec_begin; r < spec_end; ++r) {
        bool is_dirty = std::find(dirty_slots.begin(), dirty_slots.end(),
                                  (int64_t)r) != dirty_slots.end();
        if (is_dirty) {
            std::printf("[%s] FAIL: precondition, dirty slot aliases the "
                        "logical index %d\n", name, r);
            failures++;
        }
    }

    // Non-resident positions (chunk beyond the allocated pool) are skipped:
    // rollback/restore of a span past the pool must not touch any row.
    dflash_kv_clear_speculative_rows({k}, {v}, &pager,
                                     cfg.pool_tokens, cfg.pool_tokens + 4);
    failures += check_rows(k, 0, dirty_slots, "paged-nonresident K");
    failures += check_rows(v, 1, dirty_slots, "paged-nonresident V");

    if (failures == 0) {
        std::printf("[%s] OK: paged restore zeroed physical slots "
                    "%d..%d and preserved unrelated rows\n",
                    name, (int)dirty_slots.front(), (int)dirty_slots.back());
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return failures;
}

int run_dense(ggml_backend_t backend, const char * name) {
    // Dense layout: row == logical position. Covers both the restore_kv()
    // dirty-span clear and the rollback_to() truncation past the commit.
    const int64_t rows = 64;
    ggml_init_params ip{};
    ip.mem_size = 2 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                         kHeadDim, rows, kHeads);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                         kHeadDim, rows, kHeads);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::printf("[%s] FAIL: dense tensor alloc\n", name);
        ggml_free(ctx);
        return 1;
    }
    fill_rows(k, 0);
    fill_rows(v, 1);

    // Rollback truncation of rows [10, 14): commit ends at 10, verify wrote
    // through 14 before rejection.
    dflash_kv_clear_speculative_rows({k}, {v}, /*pager=*/nullptr, 10, 14);
    // Empty span is a no-op.
    dflash_kv_clear_speculative_rows({k}, {v}, /*pager=*/nullptr, 20, 20);

    int failures = 0;
    failures += check_rows(k, 0, {10, 11, 12, 13}, "dense K");
    failures += check_rows(v, 1, {10, 11, 12, 13}, "dense V");
    if (failures == 0) {
        std::printf("[%s] OK: dense restore/rollback zeroed exactly rows "
                    "10..13\n", name);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return failures;
}

int run_backend(ggml_backend_t backend, const char * name) {
    return run_paged(backend, name) + run_dense(backend, name);
}

}  // namespace

int main(int argc, char ** argv) {
    const bool cpu_only = argc > 1 && std::strcmp(argv[1], "--cpu") == 0;
    int failures = 0;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::printf("FAIL: cpu backend init\n");
        return 1;
    }
    failures += run_backend(cpu, "cpu");
    ggml_backend_free(cpu);

#ifdef GGML_USE_CUDA
    if (!cpu_only) {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (!gpu) {
            std::printf("FAIL: cuda backend init\n");
            return failures + 1;
        }
        failures += run_backend(gpu, "cuda");
        ggml_backend_free(gpu);
    }
#endif

    std::printf(failures == 0 ? "PASS\n" : "FAILURES=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
