// Native-commit policy constants and helpers for the Qwen3.8 DFlash2 path.
//
// The native commit derives greedy acceptance, the correction token, and the
// committed recurrent state from batched verify rows that are built to be
// bit-identical to the one-token autoregressive graph (see
// qwen35_dflash_target.cpp, AR-exact rows). Two host-side policies keep that
// guarantee and keep the adaptive controller live; they are pinned here and
// covered by test_dflash2_contract.
#pragma once

#include <algorithm>

namespace dflash::common {

// Verify width cap as a draft-token depth (verify rows = depth + 1 including
// the seed). The CUDA MMVQ GEMV keeps one launch geometry for 1..4 columns,
// so every quantized projection in a <=4-row verify reduces in exactly the
// same order as the one-token AR step and the committed recurrent state stays
// bit-identical to AR decode. Verified on RTX 3090: rollback-committed state
// is bit-equal to an exact AR walk at verify widths 2..4 and diverges at
// width 5+, where the kernel table switches geometry. Raising this past 3
// breaks exact greedy parity of the native commit.
constexpr int kDflash2NativeCommitMaxDepth = 3;

// Depth floor for the native commit. The adaptive controller's contraction
// can reach depth 0, and observe() ignores zero-proposal steps, so depth 0
// is an absorbing state: acceptance would never be observed again and every
// remaining step would degrade to one committed token per full draft+verify
// cycle. The draft forward runs regardless of depth, so one proposal costs
// nothing extra — keep at least one so the controller re-expands the moment
// the content becomes predictable again.
inline int dflash2_native_depth_floor(int chosen_depth, int depth_budget) {
    if (depth_budget <= 0) return chosen_depth;
    return std::max(chosen_depth, 1);
}

}  // namespace dflash::common
