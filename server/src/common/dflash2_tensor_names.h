// Published/canonical DFlash2 GGUF tensor-name compatibility.
#pragma once

#include <string>

namespace dflash::common {

inline std::string dflash2_tensor_alias(const std::string & canonical) {
    if (canonical == "selector.hidden_proj.weight") return "selector_hidden.weight";
    if (canonical == "selector.pred_codebook") return "selector_predecessor.weight";
    if (canonical == "selector.succ_codebook") return "selector_successor.weight";

    for (const char * side : {"attn", "ffn"}) {
        const std::string marker = std::string(".") + side + "_conv.";
        const size_t marker_pos = canonical.find(marker);
        if (marker_pos != std::string::npos && canonical.rfind("blk.", 0) == 0) {
            return canonical.substr(0, marker_pos) + "." + side + "_conv_" +
                   canonical.substr(marker_pos + marker.size());
        }
    }
    return {};
}

}  // namespace dflash::common
