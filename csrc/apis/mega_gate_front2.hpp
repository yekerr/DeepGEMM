#pragma once

// `bf16_mega_gate_front2`, FRONT2: the MoE front for the official cutedsl_megamoe gen-phase kernel.
//
// One launch = DeepGEMM's `bf16_mega_gate` (same config, same GEMM split, same sqrt-softplus + bias / image-bias top-k, same weights)
// plus what the official chain runs after it as separate launches: the router's EPLB logical -> physical map and load record
// (vLLM `eplb_map_to_physical_and_record`), the int32 narrowing of the ids, and the official entry's staging (FlashInfer
// `fused_quant_stage` -> cutedsl_megamoe `DataPreprocess`: NVFP4 activation + E4M3 block scales into the kernel's plain-row planes,
// int32 ids and fp32 scores into its route planes, -1 over the route capacity tail). Every output is meant to be bit-identical to
// that unfused chain.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../utils/torch_compat.hpp"
#include "../torch_library_utils.hpp"

#include <cmath>

#include "../utils/compatibility.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_gate_front2.hpp"
#include "../utils/layout.hpp"
#include "mega_gate.hpp"

namespace deep_gemm::mega_gate {

/**
 * FRONT2: BF16 gate GEMM + sqrt-softplus top-k with bias (+ image bias) + EPLB map / record + the official entry's staging.
 *
 * Args:
 *     x                     [T,H]          BF16 (contiguous)
 *     weight                [E,H]          BF16 (contiguous)
 *     num_topk, routed_scaling_factor, ep_rank, scoring_func ("sqrtsoftplus"): as `bf16_mega_gate`
 *     bias                  [E]            FP32 (required)
 *     image_bias            [E]            FP32, optional (vision checkpoints: image sentinel tokens rank with it)  } together
 *     image_token_mask      [T]            bool, optional                                                           }
 *     format                int            0 = NVFP4 (e2m1 + one E4M3 scale per 16 columns, offline normalization);
 *                                          1 = MXFP8 (e4m3 + one UE8M0 per 32 columns, normal scales: the SM107 MXFP4 x MXFP8 entry)
 *     x_q                   [>=T,>=H/2]    uint8 view of the activation plane, out (NVFP4; MXFP8: [>=T,>=H]) (column stride 1)
 *     x_sf                  [>=T,>=H/16]   uint8 view of the scale plane, out (NVFP4; MXFP8: [>=T,>=H/32]) (column stride 1)
 *     norm_const, scale_multiplier: the entry's offline NVFP4 normalization (`offline_norm_const`, `offline_scale_multiplier`;
 *                                   unused by MXFP8)
 *     xs_q                  [>=T,>=H]      uint8, out, optional: the fused shared expert's MXFP8 input (FlashInfer's swizzled
 *                                          `mxfp8_quantize`, bit for bit)                                                } together
 *     xs_sf                 [>=P*S]        uint8, out, optional: its UE8M0 scales, 128x4 swizzled, P = ceil(T / 128) * 128 rows, }
 *                                          S = ceil(H / 128) * 4 columns; rows [T, P) zero
 *     topk_idx              [C,K]          int32, out: physical ids (-1 for padding), rows [T, C) set to -1 (C >= T)
 *     topk_weights          [>=T,K]        FP32, out: the gate's routing weights (kept for padding, as the entry stages them)
 *     is_padding            [>=T]          bool, optional
 *     logical_to_physical   [E,R]          int64, optional EPLB map  } all four together, or none
 *     replica_count         [E]            int64                     }
 *     expert_load           [P]            int32, in/out             }
 *     record_enabled        []             bool                      }
 *     num_unpadded_tokens   []             int32, optional
 */
static void bf16_mega_gate_front2(const torch::stable::Tensor& x,
                                  const torch::stable::Tensor& weight,
                                  const int& num_topk,
                                  const float& routed_scaling_factor,
                                  const int& ep_rank,
                                  const std::string& scoring_func,
                                  const torch::stable::Tensor& bias,
                                  const std::optional<torch::stable::Tensor>& image_bias,
                                  const std::optional<torch::stable::Tensor>& image_token_mask,
                                  const int& format,
                                  const torch::stable::Tensor& x_q,
                                  const torch::stable::Tensor& x_sf,
                                  const float& norm_const,
                                  const float& scale_multiplier,
                                  const std::optional<torch::stable::Tensor>& xs_q,
                                  const std::optional<torch::stable::Tensor>& xs_sf,
                                  const torch::stable::Tensor& topk_idx,
                                  const torch::stable::Tensor& topk_weights,
                                  const std::optional<torch::stable::Tensor>& is_padding,
                                  const std::optional<torch::stable::Tensor>& logical_to_physical,
                                  const std::optional<torch::stable::Tensor>& replica_count,
                                  const std::optional<torch::stable::Tensor>& expert_load,
                                  const std::optional<torch::stable::Tensor>& record_enabled,
                                  const std::optional<torch::stable::Tensor>& num_unpadded_tokens) {
    constexpr int kNumMaxRoutedExperts = 512, kNumMaxTopk = 32;
    const auto [num_tokens, hidden] = get_shape<2>(x);
    const auto num_routed_experts = static_cast<int>(weight.size(0));
    const bool has_image = image_bias.has_value();
    const bool is_mxfp8 = format == static_cast<int>(mega_gate_layout::Front2Format::kMxfp8);
    DG_HOST_ASSERT(format == static_cast<int>(mega_gate_layout::Front2Format::kNvfp4) or is_mxfp8);
    DG_HOST_ASSERT(hidden > 0 and hidden % 1024 == 0);
    DG_HOST_ASSERT(num_routed_experts <= kNumMaxRoutedExperts and
                   num_routed_experts % static_cast<int>(mega_gate_layout::kNumExpertsPerLaneVector) == 0);
    DG_HOST_ASSERT(mega_gate_layout::front2_eplb_cache_fits(
        static_cast<uint32_t>(align(num_routed_experts, static_cast<int>(mega_gate_layout::kExpertAlignment))), has_image));

    const auto device = x.device();
    const auto check_dense = [&](const torch::stable::Tensor& tensor, const torch::headeronly::IntHeaderOnlyArrayRef& shape, const torch::headeronly::ScalarType& dtype) {
        DG_HOST_ASSERT(tensor.sizes().equals(shape) and tensor.scalar_type() == dtype);
        DG_HOST_ASSERT(tensor.is_contiguous());
        DG_HOST_ASSERT(tensor.device() == device);
    };
    const auto check_plane = [&](const torch::stable::Tensor& tensor, const int64_t& min_rows, const int64_t& min_cols, const torch::headeronly::ScalarType& dtype) {
        DG_HOST_ASSERT(tensor.dim() == 2 and tensor.size(0) >= min_rows and tensor.size(1) >= min_cols);
        DG_HOST_ASSERT(tensor.scalar_type() == dtype and tensor.device() == device and tensor.stride(1) == 1);
    };
    check_dense(x, {num_tokens, hidden}, torch::headeronly::ScalarType::BFloat16);
    check_dense(weight, {num_routed_experts, hidden}, torch::headeronly::ScalarType::BFloat16);
    check_dense(bias, {num_routed_experts}, torch::headeronly::ScalarType::Float);
    DG_HOST_ASSERT(image_bias.has_value() == image_token_mask.has_value());
    if (has_image) {
        check_dense(image_bias.value(), {num_routed_experts}, torch::headeronly::ScalarType::Float);
        check_dense(image_token_mask.value(), {num_tokens}, torch::headeronly::ScalarType::Bool);
    }
    DG_HOST_ASSERT(num_topk > 0 and num_topk <= num_routed_experts and num_topk <= kNumMaxTopk);
    DG_HOST_ASSERT(ep_rank >= 0);
    DG_HOST_ASSERT(std::isfinite(routed_scaling_factor) and std::isfinite(norm_const) and std::isfinite(scale_multiplier));
    DG_HOST_ASSERT(scoring_func == "sqrtsoftplus");

    check_plane(x_q, num_tokens, is_mxfp8 ? hidden : hidden / 2, torch::headeronly::ScalarType::Byte);
    check_plane(x_sf, num_tokens, is_mxfp8 ? hidden / 32 : hidden / 16, torch::headeronly::ScalarType::Byte);
    DG_HOST_ASSERT(x_q.stride(0) % 16 == 0 and (is_mxfp8 or x_sf.stride(0) % 2 == 0));
    DG_HOST_ASSERT(xs_q.has_value() == xs_sf.has_value());
    const bool with_shared_x = xs_q.has_value();
    const auto xs_padded_rows = static_cast<int64_t>(ceil_div(num_tokens, 128) * 128);
    const auto xs_padded_cols = static_cast<int64_t>(ceil_div(hidden / 32, 4) * 4);
    if (with_shared_x) {
        check_plane(xs_q.value(), num_tokens, hidden, torch::headeronly::ScalarType::Byte);
        DG_HOST_ASSERT(xs_q->stride(0) % 16 == 0);
        DG_HOST_ASSERT(xs_sf->dim() == 1 and xs_sf->is_contiguous() and xs_sf->scalar_type() == torch::headeronly::ScalarType::Byte);
        DG_HOST_ASSERT(xs_sf->device() == device and xs_sf->size(0) >= xs_padded_rows * xs_padded_cols);
    }
    check_plane(topk_idx, num_tokens, num_topk, torch::headeronly::ScalarType::Int);
    DG_HOST_ASSERT(topk_idx.size(1) == num_topk);
    check_plane(topk_weights, num_tokens, num_topk, torch::headeronly::ScalarType::Float);
    DG_HOST_ASSERT(topk_weights.size(1) == num_topk);
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
        // The packed cache holds replica column 0 in 11 bits and the replica count in 5 bits
        DG_HOST_ASSERT(m.size(1) >= 1 and m.size(1) <= static_cast<int64_t>(mega_gate_layout::kFront2MaxReplicas));
        check_dense(replica_count.value(), {num_routed_experts}, torch::headeronly::ScalarType::Long);
        const auto& load = expert_load.value();
        DG_HOST_ASSERT(load.dim() == 1 and load.is_contiguous() and load.scalar_type() == torch::headeronly::ScalarType::Int);
        num_physical = load.size(0);
        DG_HOST_ASSERT(num_physical > 0 and num_physical < static_cast<int64_t>(mega_gate_layout::kFront2MaxPhysical));
        DG_HOST_ASSERT(record_enabled->numel() == 1 and record_enabled->scalar_type() == torch::headeronly::ScalarType::Bool);
        if (num_unpadded_tokens.has_value())
            DG_HOST_ASSERT(num_unpadded_tokens->numel() == 1 and num_unpadded_tokens->scalar_type() == torch::headeronly::ScalarType::Int);
    }

    // An empty batch is the caller's (the entry masks the whole route plane in Python, as FlashInfer's staging does)
    DG_HOST_ASSERT(num_tokens > 0);
    DG_HOST_ASSERT(num_tokens <= static_cast<int>(mega_gate_layout::kNumMaxTokens));
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);

    const mega_gate_layout::RoutingArgs routing_args = {
        .to_physical_map = nullptr,
        .logical_count = nullptr,
        .topk_idx = nullptr,
        .unmapped_topk_idx = nullptr,
        .topk_weights = nullptr,
        .unmapped_topk_idx_stride = 0,
        .num_routed_experts = static_cast<uint32_t>(num_routed_experts),
        .num_shared_experts = 0u,
        .num_duplicate_experts = 0u,
        .rank_idx = static_cast<uint32_t>(ep_rank),
        .routed_scaling_factor = routed_scaling_factor,
    };
    const mega_gate_layout::Front2Args front_args = {
        .x = reinterpret_cast<const uint16_t*>(x.mutable_data_ptr()),
        .x_q = reinterpret_cast<uint8_t*>(x_q.mutable_data_ptr()),
        .x_sf = reinterpret_cast<uint8_t*>(x_sf.mutable_data_ptr()),
        .x_stride_m = x.stride(0), .x_q_stride_m = x_q.stride(0), .x_sf_stride_m = x_sf.stride(0),
        .norm_const = norm_const, .scale_multiplier = scale_multiplier,
        .xs_q = with_shared_x ? reinterpret_cast<uint8_t*>(xs_q->mutable_data_ptr()) : nullptr,
        .xs_sf = with_shared_x ? reinterpret_cast<uint8_t*>(xs_sf->mutable_data_ptr()) : nullptr,
        .xs_q_stride_m = with_shared_x ? xs_q->stride(0) : 0,
        .xs_sf_padded_rows = static_cast<uint32_t>(xs_padded_rows), .xs_sf_padded_cols = static_cast<uint32_t>(xs_padded_cols),
        .topk_idx_out = topk_idx.mutable_data_ptr<int32_t>(),
        .topk_weights_out = topk_weights.mutable_data_ptr<float>(),
        .topk_idx_stride_m = topk_idx.stride(0), .topk_weights_stride_m = topk_weights.stride(0),
        .route_capacity = static_cast<uint32_t>(topk_idx.size(0)),
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
    sm100_bf16_mega_gate_front2(x, weight, bias, image_bias, image_token_mask, routing_args, front_args,
                                format, has_eplb, is_padding.has_value(), with_shared_x,
                                get_score_barriers(x), num_tokens, hidden, num_routed_experts, num_topk);
}

} // namespace deep_gemm::mega_gate

namespace deep_gemm::torch_registration {
using namespace deep_gemm::torch_utils;

static void bf16_mega_gate_front2(const torch::stable::Tensor& x,
                                  const torch::stable::Tensor& weight,
                                  const int64_t& num_topk,
                                  const double& routed_scaling_factor,
                                  const int64_t& ep_rank,
                                  const std::string& scoring_func,
                                  const torch::stable::Tensor& bias,
                                  const std::optional<torch::stable::Tensor>& image_bias,
                                  const std::optional<torch::stable::Tensor>& image_token_mask,
                                  const int64_t& format,
                                  const torch::stable::Tensor& x_q,
                                  const torch::stable::Tensor& x_sf,
                                  const double& norm_const,
                                  const double& scale_multiplier,
                                  const std::optional<torch::stable::Tensor>& xs_q,
                                  const std::optional<torch::stable::Tensor>& xs_sf,
                                  const torch::stable::Tensor& topk_idx,
                                  const torch::stable::Tensor& topk_weights,
                                  const std::optional<torch::stable::Tensor>& is_padding,
                                  const std::optional<torch::stable::Tensor>& logical_to_physical,
                                  const std::optional<torch::stable::Tensor>& replica_count,
                                  const std::optional<torch::stable::Tensor>& expert_load,
                                  const std::optional<torch::stable::Tensor>& record_enabled,
                                  const std::optional<torch::stable::Tensor>& num_unpadded_tokens) {
    // The f32 multipliers are passed through f64 by the schema; the caller hands the f32 value, so the round trip is exact
    mega_gate::bf16_mega_gate_front2(x, weight, static_cast<int>(num_topk), static_cast<float>(routed_scaling_factor),
        static_cast<int>(ep_rank), scoring_func, bias, image_bias, image_token_mask, static_cast<int>(format), x_q, x_sf,
        static_cast<float>(norm_const), static_cast<float>(scale_multiplier), xs_q, xs_sf, topk_idx, topk_weights, is_padding,
        logical_to_physical, replica_count, expert_load, record_enabled, num_unpadded_tokens);
}
} // namespace deep_gemm::torch_registration

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("bf16_mega_gate_front2(Tensor x, Tensor weight, int num_topk, float routed_scaling_factor, int ep_rank, str scoring_func, "
          "Tensor bias, Tensor? image_bias, Tensor? image_token_mask, int format, Tensor(a!) x_q, Tensor(b!) x_sf, "
          "float norm_const, float scale_multiplier, Tensor(f!)? xs_q, Tensor(g!)? xs_sf, Tensor(c!) topk_idx, Tensor(d!) topk_weights, Tensor? is_padding, "
          "Tensor? logical_to_physical, Tensor? replica_count, Tensor(e!)? expert_load, Tensor? record_enabled, "
          "Tensor? num_unpadded_tokens) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("bf16_mega_gate_front2", TORCH_BOX(&deep_gemm::torch_registration::bf16_mega_gate_front2));
}
