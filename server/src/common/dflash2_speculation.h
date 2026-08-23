// Exact greedy acceptance and replay helpers shared by DFlash2 tests/loop.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dflash::common {

inline int dflash2_accept_prefix(const std::vector<int32_t> & draft,
                                 const std::vector<int32_t> & target,
                                 int proposed) {
    if (draft.empty() || target.empty() || proposed <= 0) return 0;
    const int limit = std::min({proposed, (int)draft.size() - 1,
                               (int)target.size()});
    int accepted = 1;  // draft[0] is the already-committed seed.
    for (int i = 0; i + 1 < limit; ++i) {
        if (draft[(size_t)i + 1] != target[(size_t)i]) break;
        ++accepted;
    }
    return accepted;
}

inline std::vector<int32_t> dflash2_replay_tokens(
        const std::vector<int32_t> & draft,
        const std::vector<int32_t> & target,
        int accepted,
        int32_t bonus) {
    if (draft.empty() || accepted <= 0) return {};
    std::vector<int32_t> replay;
    const int keep = std::min(accepted, (int)draft.size());
    replay.reserve((size_t)keep + (bonus >= 0 ? 1u : 0u));
    replay.insert(replay.end(), draft.begin(), draft.begin() + keep);
    if (bonus >= 0) replay.push_back(bonus);
    (void)target;
    return replay;
}

inline std::vector<int32_t> dflash2_fixed_verify_tokens(
        const std::vector<int32_t> & tokens,
        int verify_len,
        int verify_width,
        int32_t pad_token) {
    if (verify_len <= 0 || verify_width < verify_len ||
        verify_len > (int)tokens.size()) {
        return {};
    }
    std::vector<int32_t> fixed(tokens.begin(), tokens.begin() + verify_len);
    fixed.resize((size_t)verify_width, pad_token);
    return fixed;
}

}  // namespace dflash::common
