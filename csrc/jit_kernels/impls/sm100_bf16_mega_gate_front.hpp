#pragma once

// Host launcher of the fused MoE front (the stock BF16 mega gate + vLLM's MegaMoE input staging, EPLB map
// and load record in one launch). The gate part reuses `get_mega_gate_config`, the score scratch, the score barriers and the
// TMA descriptors of `sm100_bf16_mega_gate` unchanged, so the GEMM split and the top-k are the stock kernel's for every T.

#include <format>
#include <torch/csrc/stable/ops.h>
#include "../../utils/torch_compat.hpp"

#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/layout/mega_gate_front.cuh>

#include "../../runtime/jit.hpp"
#include "../../runtime/runtime.hpp"
#include "../heuristics/mega_gate.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm100_bf16_mega_gate_front(const torch::stable::Tensor& x, const torch::stable::Tensor& weight,
                                       const torch::stable::Tensor& bias,
                                       const mega_gate_layout::RoutingArgs& routing_args,
                                       const mega_gate_layout::FrontArgs& front_args,
                                       const bool& has_eplb, const bool& has_padding, const bool& has_shared_sf,
                                       const bool& has_logical_out,
                                       const torch::stable::Tensor& score_barriers,
                                       const int& num_tokens, const int& hidden,
                                       const int& num_routed_experts, const int& num_topk) {
    const auto num_aligned_experts = align(num_routed_experts, static_cast<int>(mega_gate_layout::kExpertAlignment));

    // The stock gate's config: same block tokens, split-K, expert groups and CTA pairs for every T
    const auto config = get_mega_gate_config(num_tokens, hidden, num_routed_experts, runtime->get_num_sms());

    const auto num_token_blocks = ceil_div(num_tokens, config.block_tokens);
    const auto num_scratch_bytes = mega_gate_layout::Workspace<>::get_num_scratch_bytes(
        num_token_blocks, config.num_split_k, config.block_tokens, num_aligned_experts);
    const auto scratch = torch::stable::new_empty(
        x, {static_cast<int64_t>(num_scratch_bytes)}, torch::headeronly::ScalarType::Byte);

    const auto tensor_map_x = make_tma_a_desc(cute::UMMA::Major::K, x, num_tokens, hidden,
                                              config.load_block_m, mega_gate_layout::BLOCK_K,
                                              static_cast<int>(x.stride(0)), 1, mega_gate_layout::kSwizzleMode);
    const auto tensor_map_weight = make_tma_b_desc(cute::UMMA::Major::K, weight, num_routed_experts, hidden,
                                                   config.load_block_n, mega_gate_layout::BLOCK_K,
                                                   static_cast<int>(weight.stride(0)), 1, mega_gate_layout::kSwizzleMode);

    // Compile: the stock template arguments (bias on, every other routing control off) + the front features
    const auto kernel = jit->compile("sm100_bf16_mega_gate_front", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_gate_front.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_gate_front_impl<
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, true, false,
        false, false, false,
        false, false,
        {}, {}, {}, {}
    >);
}};
)",
        hidden, num_aligned_experts,
        config.num_gate_threads,
        config.num_mma_ctas, config.num_split_k, config.num_expert_groups,
        num_topk,
        num_routed_experts != num_aligned_experts,
        has_eplb, has_padding, has_shared_sf, has_logical_out));

    // Launch: the same grid / block / cluster / SMEM as the stock gate. PDL follows DeepGEMM's global `set_pdl` (vLLM turns
    // it on for SM90+), and the kernel triggers its dependents right after its prologue.
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(config.num_launch_sms, 1, 1),
            .block_dim = dim3(mega_gate_layout::kNumNonEpilogueThreads + config.num_gate_threads, 1, 1),
            .cluster_dim = dim3(config.num_mma_ctas, 1, 1),
        },
        tensor_map_x, tensor_map_weight,
        bias.mutable_data_ptr(),
        static_cast<void*>(nullptr), static_cast<void*>(nullptr),
        static_cast<void*>(nullptr), static_cast<void*>(nullptr), static_cast<void*>(nullptr),
        routing_args, front_args,
        scratch.mutable_data_ptr(), score_barriers.mutable_data_ptr(),
        static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(config.block_tokens),
        static_cast<uint32_t>(config.num_stages),
        static_cast<uint32_t>(config.num_worker_groups),
        static_cast<uint32_t>(num_token_blocks));
}

} // namespace deep_gemm
