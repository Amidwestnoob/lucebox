// Qwen35DFlashTarget — DFlashTarget implementation for qwen35 hybrid models.
//
// Wraps the existing qwen35 target infrastructure (TargetWeights, TargetCache,
// StepGraph, DraftFeatureMirror) behind the generic DFlashTarget interface.
// This adapter enables the generic spec-decode loop to drive qwen35 verification.

#pragma once

#include "common/dflash_target.h"
#include "internal.h"         // TargetWeights, TargetCache, DraftWeights
#include "step_graph.h"
#include "graph_builders.h"
#include "kvflash_pager.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace dflash::common {

class Qwen35DFlashTarget : public DFlashTarget {
public:
    // Non-owning references — caller must ensure lifetime.
    Qwen35DFlashTarget(TargetWeights & w,
                       TargetCache & cache,
                       ggml_backend_t backend,
                       StepGraph & sg,
                       int kq_stride_pad,
                       int fa_window);

    ~Qwen35DFlashTarget() override;

    // ── DFlashTarget interface ──────────────────────────────────────

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override;

    // One-token forward through the AR-exact graph shape (set_rows KV write,
    // maskless 256-padded flash-attention span). Bit-identical numerics to
    // the AR decode step; feature capture stays on for the draft mirror.
    // Qwen35-specific extension (not part of the generic DFlashTarget
    // interface); reached through the typed pointer in do_spec_decode.
    bool ar_exact_step(int32_t tok, int pos, int & argmax_out);

    bool read_verify_logits(int n_tokens, std::vector<float> & out) override;

    bool snapshot_kv() override;
    bool restore_kv() override;
    bool supports_fast_rollback() const override;
    bool exact_fast_rollback() const override { return specla_active(); }
    bool rollback_failure_is_recoverable() const override { return !specla_active(); }
    bool rollback_to(int base_pos, int commit_n) override;
    bool finish_speculative_state() override;
    void set_capture_target_features(bool enabled) override {
        capture_target_features_ = enabled;
    }
    void set_force_verify_mask(bool enabled) override {
        force_verify_mask_ = enabled;
    }
    // AR-exact verify rows (qwen35-specific extension): verify_batch builds
    // the interleaved per-row write+attend graph whose per-row logits and
    // state advance are bit-identical to sequential AR decode.
    void set_ar_exact_rows(bool enabled) {
        ar_exact_rows_ = enabled;
    }

    bool supports_tree_verify() const override;
    bool verify_tree(int committed,
                     const DDTree & tree,
                     const std::vector<int32_t> & flat_tokens,
                     int n_alloc,
                     std::vector<int32_t> & posterior_out,
                     std::vector<float> * logits_out = nullptr) override;
    bool rollback_to_tree(int committed,
                          const DDTree & tree,
                          const std::vector<int> & accepted_dfs) override;

    bool is_eos(int token) const override;

    bool embed_tokens(const int32_t * tokens, int n,
                      float * out) const override;

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;

    ggml_tensor * lm_head_tensor() override { return w_.output; }

    bool project_hidden_to_topk(const float * hidden,
                                int n_tokens,
                                int K,
                                float temperature,
                                std::vector<float> & top_log_probs,
                                std::vector<int32_t> & top_token_ids) override;

    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override;
    const std::vector<int> & capture_layer_ids() const override;

    // kvflash mode: verify writes are slot-mapped via the pager and the
    // attention mask carries slot validity (resident committed positions
    // only) plus causal structure among the verify tokens. Rejected draft
    // tokens need no explicit rollback: their slots are excluded by the
    // pos < base_pos validity rule on the next verify and get rewritten.
    // Forces fa_window = 0 (logical windowing is meaningless in slot space).
    void set_kvflash_pager(KvFlashPager * pager) { pager_ = pager; }

    // Enable fast-rollback mode: verify will capture per-step SSM intermediates
    // so rollback_to() can restore recurrent state without replay.
    void set_fast_rollback(bool enabled) { fast_rollback_ = enabled; }

private:
    TargetWeights & w_;
    TargetCache & cache_;
    ggml_backend_t backend_;
    StepGraph & sg_;
    int kq_stride_pad_;
    int fa_window_;
    KvFlashPager * pager_ = nullptr;
    bool fast_rollback_ = false;
    int snapshot_cur_pos_ = 0;
    bool capture_target_features_ = true;
    bool force_verify_mask_ = false;
    bool ar_exact_rows_ = false;

    // SpecLA (DFLASH_SPECLA=1, docs/SPECLA.md): true when the cache was
    // migrated with factor buffers. Capture-verify then runs the
    // topology-masked factor path, never mutates durable SSM/conv state
    // (snapshot/restore become no-ops), and rollback commits via
    // DeltaConstruct instead of dense checkpoint copies.
    bool specla_active() const {
        return fast_rollback_ && pager_ == nullptr && !cache_.factor_k.empty();
    }

    // SpecLA chain commit: DeltaConstruct over the accepted prefix plus a
    // fused shift/append of raw convolution factors.
    bool rollback_to_specla(int base_pos, int commit_n);

    // Cached vector form of capture layer IDs (built once in constructor).
    std::vector<int> capture_ids_;

    // Fixed-offset feature staging for AR-exact verify graphs (lazily
    // allocated, owned here; registered with the graph builder through
    // qwen35_set_ar_exact_feat_staging while alive).
    ggml_context * staging_ctx_ = nullptr;
    ggml_backend_buffer_t staging_buf_ = nullptr;
    ggml_tensor * feat_staging_ = nullptr;
    bool ensure_feat_staging(int max_rows);
    bool flush_feat_staging(int base_pos, int n_tokens);

    // LM-head projection graph (lazily built).
    StepGraph proj_sg_;
};

}  // namespace dflash::common
