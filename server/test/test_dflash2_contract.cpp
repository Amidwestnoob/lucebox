#include "CppUnitTestFramework.hpp"
#include "common/dflash2_adaptive.h"
#include "common/dflash2_speculation.h"
#include "common/dflash2_tensor_names.h"
#include "qwen35/dflash2_native_commit.h"

#include <string>
#include <vector>

namespace {
struct DFlash2ContractFixture {};
}

using namespace dflash::common;

TEST_CASE(DFlash2ContractFixture, published_flattened_tensor_aliases_are_stable) {
    CHECK(dflash2_tensor_alias("selector.hidden_proj.weight") == "selector_hidden.weight");
    CHECK(dflash2_tensor_alias("selector.pred_codebook") == "selector_predecessor.weight");
    CHECK(dflash2_tensor_alias("selector.succ_codebook") == "selector_successor.weight");
    CHECK(dflash2_tensor_alias("blk.0.attn_conv.base") == "blk.0.attn_conv_base");
    CHECK(dflash2_tensor_alias("blk.4.ffn_conv.proj.weight") == "blk.4.ffn_conv_proj.weight");
    CHECK(dflash2_tensor_alias("blk.0.attn_q.weight").empty());
}

TEST_CASE(DFlash2ContractFixture, adaptive_depth_contracts_and_expands_with_acceptance) {
    DFlash2AdaptiveController controller(/*max_depth=*/7, /*initial_depth=*/2);
    CHECK(controller.choose(7) == 2);
    controller.observe(/*proposed=*/2, /*accepted=*/0);
    CHECK(controller.choose(7) == 1);
    controller.observe(/*proposed=*/1, /*accepted=*/0);
    CHECK(controller.choose(7) == 0);
    controller.observe(/*proposed=*/7, /*accepted=*/7);
    CHECK(controller.choose(7) == 1);
    controller.observe(/*proposed=*/7, /*accepted=*/7);
    CHECK(controller.choose(7) == 2);
    CHECK(controller.choose(7) <= 7);
}

TEST_CASE(DFlash2ContractFixture, native_depth_floor_prevents_absorbing_zero) {
    // Regression: the controller alone can contract to depth 0, and
    // observe(0, 0) is a no-op, so a native-commit request stuck at depth 0
    // could never observe acceptance again — every later step degraded to
    // one committed token per full draft+verify cycle. The native commit
    // floors the chosen depth at one proposal, so observations keep flowing
    // and the controller re-expands when content becomes predictable.
    DFlash2AdaptiveController controller(/*max_depth=*/7, /*initial_depth=*/2);
    for (int step = 0; step < 32; ++step) {
        const int depth =
            dflash2_native_depth_floor(controller.choose(7), 7);
        CHECK(depth >= 1);
        controller.observe(depth, /*accepted=*/0);
    }
    CHECK(dflash2_native_depth_floor(controller.choose(7), 7) == 1);
    controller.observe(/*proposed=*/1, /*accepted=*/1);
    controller.observe(/*proposed=*/2, /*accepted=*/2);
    CHECK(controller.choose(7) >= 2);

    // A zero budget still yields zero (nothing to propose).
    CHECK(dflash2_native_depth_floor(0, 0) == 0);
}

TEST_CASE(DFlash2ContractFixture, native_commit_width_cap_stays_in_mmvq_uniform_class) {
    // The native commit derives acceptance, the correction token, and the
    // committed recurrent state from batched verify rows. Those rows are
    // bit-identical to the one-token AR graph only while every quantized
    // projection keeps the single-geometry MMVQ launch class, which holds
    // for 1..4 columns (seed + up to 3 proposals). Raising the cap breaks
    // exact greedy parity; this pin fails loudly if someone tries.
    CHECK(kDflash2NativeCommitMaxDepth == 3);
    DFlash2AdaptiveController controller(/*max_depth=*/7, /*initial_depth=*/2);
    for (int i = 0; i < 8; ++i) {
        controller.observe(/*proposed=*/7, /*accepted=*/7);
    }
    CHECK(controller.choose(kDflash2NativeCommitMaxDepth) ==
          kDflash2NativeCommitMaxDepth);
    CHECK(controller.choose(kDflash2NativeCommitMaxDepth) + 1 <= 4);
    CHECK(dflash2_native_depth_floor(
              controller.choose(kDflash2NativeCommitMaxDepth),
              kDflash2NativeCommitMaxDepth) == kDflash2NativeCommitMaxDepth);
}

TEST_CASE(DFlash2ContractFixture, adaptive_depth_is_disabled_without_behavior_change) {
    DFlash2AdaptiveController controller(/*max_depth=*/7, /*initial_depth=*/2,
                                         /*enabled=*/false);
    CHECK(controller.choose(7) == 7);
    controller.observe(/*proposed=*/7, /*accepted=*/0);
    CHECK(controller.choose(3) == 3);
    CHECK(controller.telemetry().proposed_tokens == 0);
    CHECK(controller.telemetry().accepted_tokens == 0);
}

TEST_CASE(DFlash2ContractFixture, telemetry_accounting_is_bounded_and_exact) {
    DFlash2AdaptiveController controller(/*max_depth=*/7, /*initial_depth=*/2);
    controller.observe(/*proposed=*/7, /*accepted=*/3);
    controller.observe(/*proposed=*/2, /*accepted=*/2);
    const auto telemetry = controller.telemetry();
    CHECK(telemetry.proposed_tokens == 9);
    CHECK(telemetry.accepted_tokens == 5);
    CHECK(telemetry.last_depth >= 0);
    CHECK(telemetry.last_depth <= 7);
}

TEST_CASE(DFlash2ContractFixture, exact_prefix_and_rollback_replay_are_deterministic) {
    const std::vector<int32_t> draft{41, 99, 21, 30, 40};
    const std::vector<int32_t> target{99, 20, 31, 40, 50};
    const int accepted = dflash2_accept_prefix(draft, target, 4);
    CHECK(accepted == 2);
    const std::vector<int32_t> replay = dflash2_replay_tokens(
        draft, target, accepted, /*bonus=*/target[accepted - 1]);
    CHECK((replay == std::vector<int32_t>{41, 99, 20}));
}

TEST_CASE(DFlash2ContractFixture, fixed_width_verify_preserves_prefix_and_bounds_padding) {
    const std::vector<int32_t> draft{41, 99, 21, 30, 40};
    const std::vector<int32_t> verify = dflash2_fixed_verify_tokens(
        draft, /*verify_len=*/3, /*verify_width=*/5, /*pad_token=*/7);
    CHECK((verify == std::vector<int32_t>{41, 99, 21, 7, 7}));
    CHECK(dflash2_fixed_verify_tokens(draft, /*verify_len=*/0,
                                      /*verify_width=*/5, /*pad_token=*/7).empty());
}
