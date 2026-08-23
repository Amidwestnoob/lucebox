// Speculative K/V row cleanup shared by Qwen35DFlashTarget::restore_kv()
// and rollback_to().
//
// Rejected draft rows must not stay in the attention cache: the maskless
// AR decode graph pads its flash-attention span past cur_pos and relies
// on unwritten cache rows being zero, so leftover rejected-draft K/V
// would leak into the softmax denominator of any AR step that follows.
//
// Dense layout: cache tensors are [head_dim, n_ctx, n_head_kv] and the
// row index equals the logical position, so a rejected span
// [first_pos, end_pos) is one contiguous per-head memset.
//
// KVFlash paging: cache tensors are pool-sized and a logical position's
// row lives at the pager-assigned physical slot. A logical-index memset
// would both miss the dirty rows and clear unrelated resident rows, so
// each dirty position is resolved through the pager's const slot lookup
// instead; a position whose chunk is no longer resident has no stale row
// to clear. A zero row is the pager's documented freed-slot state
// (exp(-max) ~ 0 under the chunk-granular slot mask).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "kvflash_pager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash::common {

// Zero cache row `row` (every KV head) in each [head_dim, rows, n_head_kv]
// tensor. Rows outside the tensor are ignored (defense in depth).
inline void dflash_kv_zero_row(const std::vector<ggml_tensor *> & tensors,
                               int64_t row) {
    for (ggml_tensor * tensor : tensors) {
        if (!tensor || row < 0 || row >= tensor->ne[1]) continue;
        const int64_t heads = tensor->ne[2];
        for (int64_t head = 0; head < heads; ++head) {
            const size_t offset = (size_t)head * tensor->nb[2] +
                (size_t)row * tensor->nb[1];
            ggml_backend_tensor_memset(tensor, 0, offset, tensor->nb[1]);
        }
    }
}

// Zero the K/V rows a rejected speculative span [first_pos, end_pos) wrote.
// `pager` selects the layout: null means dense (row == logical position),
// non-null means kvflash pooled (row == pager physical slot).
inline void dflash_kv_clear_speculative_rows(
        const std::vector<ggml_tensor *> & attn_k,
        const std::vector<ggml_tensor *> & attn_v,
        const KvFlashPager * pager,
        int first_pos, int end_pos) {
    if (end_pos <= first_pos) return;
    if (pager != nullptr) {
        for (int pos = first_pos; pos < end_pos; ++pos) {
            const int slot = pager->slot_of(pos);
            if (slot < 0) continue;  // chunk paged out: no resident stale row
            dflash_kv_zero_row(attn_k, slot);
            dflash_kv_zero_row(attn_v, slot);
        }
        return;
    }
    const int64_t n_rows = end_pos - first_pos;
    auto clear_range = [&](const std::vector<ggml_tensor *> & tensors) {
        for (ggml_tensor * tensor : tensors) {
            if (!tensor || (int64_t)first_pos >= tensor->ne[1]) continue;
            const int64_t n_clamped =
                (std::min)(n_rows, tensor->ne[1] - first_pos);
            const int64_t heads = tensor->ne[2];
            for (int64_t head = 0; head < heads; ++head) {
                const size_t offset = (size_t)head * tensor->nb[2] +
                    (size_t)first_pos * tensor->nb[1];
                ggml_backend_tensor_memset(
                    tensor, 0, offset, (size_t)n_clamped * tensor->nb[1]);
            }
        }
    };
    clear_range(attn_k);
    clear_range(attn_v);
}

}  // namespace dflash::common
