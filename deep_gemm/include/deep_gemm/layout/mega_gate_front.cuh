#pragma once

// Host/device ABI of the fused MoE front (DeepGEMM's BF16 mega gate + vLLM's MegaMoE input staging).
// Host-safe: the JIT launcher fills `FrontArgs` and passes it by value as a `__grid_constant__` kernel parameter.

#include <cstdint>

#include <deep_gemm/layout/mega_gate.cuh>

namespace deep_gemm::layout::mega_gate {

// Everything the fused front reads or writes beyond the stock gate's `RoutingArgs`.
struct FrontArgs {
    // MXFP8 staging: x [T, H] BF16 bits (stride_k 1) -> x_fp8 [T, H] e4m3 (stride_k 1), x_sf [T, H / 128] int32
    const uint16_t* x;
    uint8_t* x_fp8;
    int32_t* x_sf;
    int32_t* shared_x_sf;  // nullptr without a fused shared expert
    int64_t x_stride_m, x_fp8_stride_m, x_sf_stride_m, x_sf_stride_k;
    int64_t shared_x_sf_stride_m, shared_x_sf_stride_k;
    uint32_t shared_block_m;

    // Routing in the MegaMoE symmetric buffer: physical int64 ids (or -1), fp32 weights
    int64_t* topk_idx_out;
    float* topk_weights_out;
    int64_t topk_idx_stride_m, topk_idx_stride_k, topk_weights_stride_m, topk_weights_stride_k;

    // Logical ids for a routing capture hook (nullptr when no hook is set)
    int64_t* logical_topk_idx_out;
    int64_t logical_topk_idx_stride_m;

    // Padding mask [T] (nullptr: no padding)
    const bool* is_padding;
    int64_t is_padding_stride_m;

    // EPLB state of this layer (all nullptr without EPLB): logical_to_physical [num_logical, map_stride] int64,
    // replica_count [num_logical] int64, expert_load [num_physical] int32, record_enabled scalar bool,
    // num_unpadded_tokens scalar int32 (nullptr: no limit)
    const int64_t* logical_to_physical;
    const int64_t* replica_count;
    int32_t* expert_load;
    const bool* record_enabled;
    const int32_t* num_unpadded_tokens;
    int64_t map_stride;
    uint32_t num_logical, num_physical;
};

// EPLB cache in the gate's 4 KiB metadata cache, after the routed bias: replica column 0 as int16, replica count as uint8
CUTLASS_HOST_DEVICE constexpr uint32_t front_eplb_cache_offset_bytes(const uint32_t num_routed_experts) {
    return num_routed_experts * static_cast<uint32_t>(sizeof(float));
}
CUTLASS_HOST_DEVICE constexpr bool front_eplb_cache_fits(const uint32_t num_routed_experts) {
    return front_eplb_cache_offset_bytes(num_routed_experts) + num_routed_experts * 3 <= kNumMaxMetadataCacheBytes;
}

} // namespace deep_gemm::layout::mega_gate
