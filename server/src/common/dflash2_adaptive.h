// Pure host-side adaptive control and bounded telemetry for DFlash2.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dflash::common {

struct DFlash2Telemetry {
    uint64_t proposed_tokens = 0;
    uint64_t accepted_tokens = 0;
    int last_depth = 0;
};

class DFlash2AdaptiveController {
public:
    DFlash2AdaptiveController(int max_depth, int initial_depth,
                              bool enabled = true)
        : max_depth_(std::max(0, max_depth)),
          depth_(std::clamp(initial_depth, 0, max_depth_)),
          enabled_(enabled) {}

    int choose(int budget) {
        const int bounded_budget = std::clamp(budget, 0, max_depth_);
        const int chosen = enabled_ ? std::min(depth_, bounded_budget)
                                    : bounded_budget;
        telemetry_.last_depth = chosen;
        return chosen;
    }

    void observe(int proposed, int accepted) {
        if (!enabled_ || proposed <= 0) return;
        const int bounded_proposed = std::clamp(proposed, 0, max_depth_);
        const int bounded_accepted = std::clamp(accepted, 0, bounded_proposed);
        telemetry_.proposed_tokens += (uint64_t) bounded_proposed;
        telemetry_.accepted_tokens += (uint64_t) bounded_accepted;

        const float ratio = (float) bounded_accepted / (float) bounded_proposed;
        if (ratio >= 0.80f) {
            ++high_streak_;
            low_streak_ = 0;
            if (high_streak_ >= 1) {
                depth_ = std::min(max_depth_, depth_ + 1);
                high_streak_ = 0;
            }
        } else if (ratio <= 0.35f) {
            ++low_streak_;
            high_streak_ = 0;
            if (low_streak_ >= 1) {
                depth_ = std::max(0, depth_ - 1);
                low_streak_ = 0;
            }
        } else {
            high_streak_ = 0;
            low_streak_ = 0;
        }
    }

    int depth() const { return depth_; }
    int max_depth() const { return max_depth_; }
    bool enabled() const { return enabled_; }
    const DFlash2Telemetry & telemetry() const { return telemetry_; }

private:
    int max_depth_ = 0;
    int depth_ = 0;
    bool enabled_ = true;
    int high_streak_ = 0;
    int low_streak_ = 0;
    DFlash2Telemetry telemetry_;
};

}  // namespace dflash::common
