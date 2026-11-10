#pragma once

// `bf16_mega_gate_front`, the fused MoE front for the MegaMoE decode path.
//
// One launch = DeepGEMM's `bf16_mega_gate` (same config, same GEMM split, same sqrt-softplus + bias top-k, same weights)
// plus what vLLM's `prepare_megamoe_inputs` (kMxfp8Dynamic) then does: MXFP8 staging of x into the MegaMoE
// symmetric buffer, the shared-expert scale page copy, padding, the EPLB logical -> physical map and the expert-load record,
// with the routing written straight into the buffer's topk_idx / topk_weights. Every output is meant to be bit-identical to
// the two stock launches.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../utils/torch_compat.hpp"
#include "../torch_library_utils.hpp"

#include <cmath>

#include "../utils/compatibility.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_gate_front.hpp"
#include "../utils/layout.hpp"
#include "mega_gate.hpp"

namespace deep_gemm::mega_gate {

/**
 * Fused MoE front: BF16 gate GEMM + sqrt-softplus top-k with bias + MXFP8 staging + EPLB map / record.
 *
 * Args:
 *     x                     [T,H]          BF16 (contiguous)
 *     weight                [E,H]          BF16 (contiguous)
 *     num_topk, routed_scaling_factor, ep_rank, scoring_func ("sqrtsoftplus"): as `bf16_mega_gate`
 *     bias                  [E]            FP32 (required: the served gate ranks with a bias)
 *     x_fp8                 [>=T,H]        float8_e4m3fn, out (row stride free, column stride 1)
 *     x_sf                  [>=T,H/128]    int32, out (four UE8M0 bytes per 128 columns)
 *     shared_x_sf           [R,H/128]      int32, out, optional (fused shared expert, MN-major page of `shared_block_m`)
 *     shared_block_m        int            BLOCK_M of this MegaMoE call (ignored without shared_x_sf)
 *     topk_idx              [>=T,K]        int64, out (physical ids after EPLB, -1 for padding)
 *     topk_weights          [>=T,K]        FP32, out (0 for padding)
 *     logical_topk_idx      [>=T,K]        int64, out, optional (logical ids, for a routing capture hook)
 *     is_padding            [>=T]          bool, optional
 *     logical_to_physical   [E,R+1]        int64, optional EPLB map  } all four together, or none
 *     replica_count         [E]            int64                     }
 *     expert_load           [P]            int32, in/out             }
 *     record_enabled        []             bool                      }
 *     num_unpadded_tokens   []             int32, optional
 */
static void bf16_mega_gate_front(const torch::stable::Tensor& x,
                                 const torch::stable::Tensor& weight,
                                 const int& num_topk,
                                 const float& routed_scaling_factor,
                                 const int& ep_rank,
                                 const std::string& scoring_func,
                                 const torch::stable::Tensor& bias,
                                 const torch::stable::Tensor& x_fp8,
                                 const torch::stable::Tensor& x_sf,
                                 const std::optional<torch::stable::Tensor>& shared_x_sf,
                                 const int& shared_block_m,
                                 const torch::stable::Tensor& topk_idx,
                                 const torch::stable::Tensor& topk_weights,
                                 const std::optional<torch::stable::Tensor>& logical_topk_idx,
                                 const std::optional<torch::stable::Tensor>& is_padding,
                                 const std::optional<torch::stable::Tensor>& logical_to_physical,
                                 const std::optional<torch::stable::Tensor>& replica_count,
                                 const std::optional<torch::stable::Tensor>& expert_load,
                                 const std::optional<torch::stable::Tensor>& record_enabled,
                                 const std::optional<torch::stable::Tensor>& num_unpadded_tokens) {
    constexpr int kHiddenAlignment = 256, kNumMaxRoutedExperts = 512, kNumMaxTopk = 32;
    const auto [num_tokens, hidden] = get_shape<2>(x);
    const auto num_routed_experts = static_cast<int>(weight.size(0));
    DG_HOST_ASSERT(hidden > 0 and hidden % kHiddenAlignment == 0);
    DG_HOST_ASSERT(num_routed_experts <= kNumMaxRoutedExperts and
                   num_routed_experts % static_cast<int>(mega_gate_layout::kNumExpertsPerLaneVector) == 0);
    DG_HOST_ASSERT(mega_gate_layout::front_eplb_cache_fits(static_cast<uint32_t>(align(num_routed_experts, 128))));

    const auto device = x.device();
    const auto check_dense = [&](const torch::stable::Tensor& tensor, const torch::headeronly::IntHeaderOnlyArrayRef& shape, const torch::headeronly::ScalarType& dtype) {
        DG_HOST_ASSERT(tensor.sizes().equals(shape) and tensor.scalar_type() == dtype);
        DG_HOST_ASSERT(tensor.is_contiguous());
        DG_HOST_ASSERT(tensor.device() == device);
    };
    const auto check_rows = [&](const torch::stable::Tensor& tensor, const int64_t& cols, const torch::headeronly::ScalarType& dtype) {
        DG_HOST_ASSERT(tensor.dim() == 2 and tensor.size(0) >= num_tokens and tensor.size(1) == cols);
        DG_HOST_ASSERT(tensor.scalar_type() == dtype and tensor.device() == device);
    };
    check_dense(x, {num_tokens, hidden}, torch::headeronly::ScalarType::BFloat16);
    check_dense(weight, {num_routed_experts, hidden}, torch::headeronly::ScalarType::BFloat16);
    check_dense(bias, {num_routed_experts}, torch::headeronly::ScalarType::Float);
    DG_HOST_ASSERT(num_topk > 0 and num_topk <= num_routed_experts and num_topk <= kNumMaxTopk);
    DG_HOST_ASSERT(ep_rank >= 0);
    DG_HOST_ASSERT(std::isfinite(routed_scaling_factor));
    DG_HOST_ASSERT(scoring_func == "sqrtsoftplus");

    check_rows(x_fp8, hidden, torch::headeronly::ScalarType::Float8_e4m3fn);
    DG_HOST_ASSERT(x_fp8.stride(1) == 1);
    check_rows(x_sf, hidden / 128, torch::headeronly::ScalarType::Int);
    check_rows(topk_idx, num_topk, torch::headeronly::ScalarType::Long);
    check_rows(topk_weights, num_topk, torch::headeronly::ScalarType::Float);
    if (shared_x_sf.has_value()) {
        const auto& s = shared_x_sf.value();
        DG_HOST_ASSERT(shared_block_m > 0);
        DG_HOST_ASSERT(s.dim() == 2 and s.size(1) == hidden / 128 and s.scalar_type() == torch::headeronly::ScalarType::Int);
        const auto aligned_block_m = ceil_div(shared_block_m, 128) * 128;
        DG_HOST_ASSERT(s.size(0) >= static_cast<int64_t>(ceil_div(num_tokens, shared_block_m)) * aligned_block_m);
    }
    if (logical_topk_idx.has_value()) {
        check_rows(logical_topk_idx.value(), num_topk, torch::headeronly::ScalarType::Long);
        DG_HOST_ASSERT(logical_topk_idx->stride(1) == 1);
    }
    if (is_padding.has_value()) {
        DG_HOST_ASSERT(is_padding->dim() == 1 and is_padding->size(0) >= num_tokens);
        DG_HOST_ASSERT(is_padding->scalar_type() == torch::headeronly::ScalarType::Bool and is_padding->device() == device);
    }

    const bool has_eplb = logical_to_physical.has_value();
    DG_HOST_ASSERT(replica_count.has_value() == has_eplb and expert_load.has_value() == has_eplb and
                   record_enabled.has_value() == has_eplb);
    int64_t map_stride = 0, num_physical = 0;
    if (has_eplb) {
        const auto& m = logical_to_physical.value();
        DG_HOST_ASSERT(m.dim() == 2 and m.size(0) == num_routed_experts and m.stride(1) == 1);
        DG_HOST_ASSERT(m.scalar_type() == torch::headeronly::ScalarType::Long and m.device() == device);
        map_stride = m.stride(0);
        DG_HOST_ASSERT(m.size(1) >= 1 and m.size(1) <= 255);
        check_dense(replica_count.value(), {num_routed_experts}, torch::headeronly::ScalarType::Long);
        const auto& load = expert_load.value();
        DG_HOST_ASSERT(load.dim() == 1 and load.is_contiguous() and load.scalar_type() == torch::headeronly::ScalarType::Int);
        num_physical = load.size(0);
        // Physical ids are cached as int16; a proxy may run fewer physical experts than the gate has logical ones
        DG_HOST_ASSERT(num_physical > 0 and num_physical < 32768);
        DG_HOST_ASSERT(record_enabled->numel() == 1 and record_enabled->scalar_type() == torch::headeronly::ScalarType::Bool);
        if (num_unpadded_tokens.has_value())
            DG_HOST_ASSERT(num_unpadded_tokens->numel() == 1 and num_unpadded_tokens->scalar_type() == torch::headeronly::ScalarType::Int);
    }

    if (num_tokens == 0)
        return;
    DG_HOST_ASSERT(num_tokens <= static_cast<int>(mega_gate_layout::kNumMaxTokens));
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);

    const mega_gate_layout::RoutingArgs routing_args = {
        .to_physical_map = nullptr,
        .logical_count = nullptr,
        .topk_idx = topk_idx.mutable_data_ptr<int64_t>(),
        .unmapped_topk_idx = nullptr,
        .topk_weights = topk_weights.mutable_data_ptr<float>(),
        .unmapped_topk_idx_stride = 0,
        .num_routed_experts = static_cast<uint32_t>(num_routed_experts),
        .num_shared_experts = 0u,
        .num_duplicate_experts = 0u,
        .rank_idx = static_cast<uint32_t>(ep_rank),
        .routed_scaling_factor = routed_scaling_factor,
    };
    const mega_gate_layout::FrontArgs front_args = {
        .x = reinterpret_cast<const uint16_t*>(x.mutable_data_ptr()),
        .x_fp8 = reinterpret_cast<uint8_t*>(x_fp8.mutable_data_ptr()),
        .x_sf = x_sf.mutable_data_ptr<int32_t>(),
        .shared_x_sf = shared_x_sf ? shared_x_sf->mutable_data_ptr<int32_t>() : nullptr,
        .x_stride_m = x.stride(0), .x_fp8_stride_m = x_fp8.stride(0),
        .x_sf_stride_m = x_sf.stride(0), .x_sf_stride_k = x_sf.stride(1),
        .shared_x_sf_stride_m = shared_x_sf ? shared_x_sf->stride(0) : 0,
        .shared_x_sf_stride_k = shared_x_sf ? shared_x_sf->stride(1) : 0,
        .shared_block_m = static_cast<uint32_t>(shared_x_sf ? shared_block_m : 1),
        .topk_idx_out = topk_idx.mutable_data_ptr<int64_t>(),
        .topk_weights_out = topk_weights.mutable_data_ptr<float>(),
        .topk_idx_stride_m = topk_idx.stride(0), .topk_idx_stride_k = topk_idx.stride(1),
        .topk_weights_stride_m = topk_weights.stride(0), .topk_weights_stride_k = topk_weights.stride(1),
        .logical_topk_idx_out = logical_topk_idx ? logical_topk_idx->mutable_data_ptr<int64_t>() : nullptr,
        .logical_topk_idx_stride_m = logical_topk_idx ? logical_topk_idx->stride(0) : 0,
        .is_padding = is_padding ? reinterpret_cast<const bool*>(is_padding->mutable_data_ptr()) : nullptr,
        .is_padding_stride_m = is_padding ? is_padding->stride(0) : 0,
        .logical_to_physical = has_eplb ? logical_to_physical->mutable_data_ptr<int64_t>() : nullptr,
        .replica_count = has_eplb ? replica_count->mutable_data_ptr<int64_t>() : nullptr,
        .expert_load = has_eplb ? expert_load->mutable_data_ptr<int32_t>() : nullptr,
        .record_enabled = has_eplb ? reinterpret_cast<const bool*>(record_enabled->mutable_data_ptr()) : nullptr,
        .num_unpadded_tokens = has_eplb and num_unpadded_tokens ? num_unpadded_tokens->mutable_data_ptr<int32_t>() : nullptr,
        .map_stride = map_stride,
        .num_logical = static_cast<uint32_t>(num_routed_experts),
        .num_physical = static_cast<uint32_t>(num_physical),
    };
    sm100_bf16_mega_gate_front(x, weight, bias, routing_args, front_args,
                               has_eplb, is_padding.has_value(), shared_x_sf.has_value(), logical_topk_idx.has_value(),
                               get_score_barriers(x), num_tokens, hidden, num_routed_experts, num_topk);
}

} // namespace deep_gemm::mega_gate

namespace deep_gemm::torch_registration {
using namespace deep_gemm::torch_utils;

static void bf16_mega_gate_front(const torch::stable::Tensor& x,
                                 const torch::stable::Tensor& weight,
                                 const int64_t& num_topk,
                                 const double& routed_scaling_factor,
                                 const int64_t& ep_rank,
                                 const std::string& scoring_func,
                                 const torch::stable::Tensor& bias,
                                 const torch::stable::Tensor& x_fp8,
                                 const torch::stable::Tensor& x_sf,
                                 const std::optional<torch::stable::Tensor>& shared_x_sf,
                                 const int64_t& shared_block_m,
                                 const torch::stable::Tensor& topk_idx,
                                 const torch::stable::Tensor& topk_weights,
                                 const std::optional<torch::stable::Tensor>& logical_topk_idx,
                                 const std::optional<torch::stable::Tensor>& is_padding,
                                 const std::optional<torch::stable::Tensor>& logical_to_physical,
                                 const std::optional<torch::stable::Tensor>& replica_count,
                                 const std::optional<torch::stable::Tensor>& expert_load,
                                 const std::optional<torch::stable::Tensor>& record_enabled,
                                 const std::optional<torch::stable::Tensor>& num_unpadded_tokens) {
    mega_gate::bf16_mega_gate_front(x, weight, static_cast<int>(num_topk), static_cast<float>(routed_scaling_factor),
        static_cast<int>(ep_rank), scoring_func, bias, x_fp8, x_sf, shared_x_sf, static_cast<int>(shared_block_m),
        topk_idx, topk_weights, logical_topk_idx, is_padding, logical_to_physical, replica_count, expert_load,
        record_enabled, num_unpadded_tokens);
}
} // namespace deep_gemm::torch_registration

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("bf16_mega_gate_front(Tensor x, Tensor weight, int num_topk, float routed_scaling_factor, int ep_rank, str scoring_func, "
          "Tensor bias, Tensor(a!) x_fp8, Tensor(b!) x_sf, Tensor(c!)? shared_x_sf, int shared_block_m, "
          "Tensor(d!) topk_idx, Tensor(e!) topk_weights, Tensor(f!)? logical_topk_idx, Tensor? is_padding, "
          "Tensor? logical_to_physical, Tensor? replica_count, Tensor(g!)? expert_load, Tensor? record_enabled, "
          "Tensor? num_unpadded_tokens) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("bf16_mega_gate_front", TORCH_BOX(&deep_gemm::torch_registration::bf16_mega_gate_front));
}
