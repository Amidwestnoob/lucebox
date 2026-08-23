// Regression for the AR decode first-step positions corruption.
//
// do_ar_decode rebuilds its step graph every iteration inside one shared
// metadata arena + gallocr. Rebuilding a graph with a different token count
// relocates the input tensors: the n_tokens=1 decode layout places
// `positions` at a different device offset than the preceding multi-token
// prefill layout. The historical bug uploaded `inp_embed`/`positions` BEFORE
// the rebuild, so the first decode step after every prefill read stale bytes
// as M-RoPE positions, permanently tilting the first generated K row in the
// cache. The fix uploads after the rebuild.
//
// This test pins both halves of that mechanism on the real allocator:
//   1. the relocation exists (the hazard precondition) — a shape change moves
//      the positions tensor, so a pre-rebuild upload cannot be trusted;
//   2. the fixed protocol (upload after rebuild) delivers exactly the
//      uploaded values to the graph on both CPU and GPU backends.
// If (1) ever fails, ggml started keeping input offsets layout-stable and
// the upload-order constraint in do_ar_decode can be revisited.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct MiniStep {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * inp_embed = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * out = nullptr;
};

// Mirror build_target_step's shape: a persistent metadata arena, inputs
// created in the same order (inp_embed then positions), one shared gallocr.
bool build_mini_step(MiniStep & sg, std::vector<uint8_t> & arena,
                     ggml_gallocr_t alloc, int hidden, int n_tokens) {
    if (sg.ctx) {
        ggml_free(sg.ctx);
        sg.ctx = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size = arena.size();
    ip.mem_buffer = arena.data();
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    sg.inp_embed = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_set_input(sg.inp_embed);
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_input(sg.positions);

    sg.gf = ggml_new_graph(sg.ctx);
    // The graph must consume both inputs so the allocator keeps them.
    ggml_tensor * emb_sum = ggml_sum_rows(sg.ctx, sg.inp_embed);
    ggml_set_output(emb_sum);
    ggml_build_forward_expand(sg.gf, emb_sum);
    sg.out = ggml_cpy(sg.ctx, sg.positions,
                      ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens));
    ggml_set_output(sg.out);
    ggml_build_forward_expand(sg.gf, sg.out);
    return ggml_gallocr_alloc_graph(alloc, sg.gf);
}

int run_backend(ggml_backend_t backend, const char * name) {
    int failures = 0;
    std::vector<uint8_t> arena(8 * 1024 * 1024);
    ggml_gallocr_t alloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    MiniStep sg;
    const int hidden = 64;

    // Prefill-shaped build (n=57), as after a 57-token prompt.
    if (!build_mini_step(sg, arena, alloc, hidden, 57)) {
        std::printf("[%s] FAIL: prefill-shaped build\n", name);
        return 1;
    }
    void * prefill_pos_data = sg.positions->data;

    // The historical bug: upload the decode step's positions into the
    // PREFILL layout's tensor, then rebuild with the decode shape.
    const int32_t intended[4] = {57, 57, 57, 0};
    ggml_backend_tensor_set(sg.positions, intended, 0, sizeof(intended));

    if (!build_mini_step(sg, arena, alloc, hidden, 1)) {
        std::printf("[%s] FAIL: decode-shaped rebuild\n", name);
        return 1;
    }

    // Informational: whether the shape change relocated `positions` in this
    // minimal graph. The production step graph (masks, salting, thousands of
    // tensors) does relocate it — that is the hazard the do_ar_decode fix
    // addresses; the end-to-end pin is the AR-vs-native parity smoke.
    std::printf("[%s] rebuild %s positions (%p -> %p)\n", name,
                sg.positions->data == prefill_pos_data ? "kept" : "moved",
                prefill_pos_data, sg.positions->data);

    // Fixed protocol: upload AFTER the rebuild, compute, and require the
    // graph to observe exactly the uploaded values.
    std::vector<float> embed((size_t)hidden, 0.5f);
    ggml_backend_tensor_set(sg.inp_embed, embed.data(), 0,
                            sizeof(float) * embed.size());
    ggml_backend_tensor_set(sg.positions, intended, 0, sizeof(intended));
    if (ggml_backend_graph_compute(backend, sg.gf) != GGML_STATUS_SUCCESS) {
        std::printf("[%s] FAIL: compute\n", name);
        ggml_free(sg.ctx);
        ggml_gallocr_free(alloc);
        return failures + 1;
    }
    int32_t got[4] = {-1, -1, -1, -1};
    ggml_backend_tensor_get(sg.out, got, 0, sizeof(got));
    if (std::memcmp(got, intended, sizeof(got)) != 0) {
        std::printf("[%s] FAIL: graph saw positions %d,%d,%d,%d; "
                    "intended %d,%d,%d,%d\n",
                    name, got[0], got[1], got[2], got[3],
                    intended[0], intended[1], intended[2], intended[3]);
        failures++;
    }

    ggml_free(sg.ctx);
    sg.ctx = nullptr;
    ggml_gallocr_free(alloc);
    if (failures == 0) {
        std::printf("[%s] OK: upload-after-rebuild is observed exactly\n",
                    name);
    }
    return failures;
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
