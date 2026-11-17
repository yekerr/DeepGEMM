#pragma once

// Host/device ABI of FRONT2: DeepGEMM's BF16 mega gate + the official gen-phase MegaMoE entry's input staging in one launch.
// Host-safe: the JIT launcher fills `Front2Args` and passes it by value as a `__grid_constant__` kernel parameter.

#include <cstdint>

#include <deep_gemm/layout/mega_gate.cuh>

namespace deep_gemm::layout::mega_gate {

// Activation formats, each bit for bit the official entry's staging (FlashInfer fused_quant_stage -> cutedsl_megamoe DataPreprocess)
enum class Front2Format : uint32_t {
    kNvfp4 = 0,      // e2m1 values (low nibble = even column), one E4M3 scale per 16 columns, offline per-tensor normalization
    kMxfp8 = 1,      // e4m3 values (one byte per column), one UE8M0 scale per 32 columns in plain rows, scales clamped to normal FP32
                     // (FlashInfer fused_quant_stage(quant_type="mxfp8_e4m3", mxfp8_normal_scales=True): the SM107 MXFP4 x MXFP8 entry)
};

struct Front2Args {
    // Staging: x [T, H] BF16 (stride_k 1) -> x_q [>= T, H / 2] bytes, x_sf [>= T, H / 16] bytes (NVFP4) or x_q [>= T, H],
    // x_sf [>= T, H / 32] bytes (MXFP8) (row strides in bytes)
    const uint16_t* x;
    uint8_t* x_q;
    uint8_t* x_sf;
    int64_t x_stride_m, x_q_stride_m, x_sf_stride_m;
    // NVFP4 offline normalization, DataPreprocess's `offline_norm_const` and `offline_scale_multiplier` (f32 bits as passed)
    float norm_const, scale_multiplier;

    // The fused shared expert's input X_s (nullptr: off): MXFP8 e4m3 [T, H] (row stride in bytes) and UE8M0 scales in the 128x4
    // swizzled (blocked) layout of `xs_sf_padded_rows` x `xs_sf_padded_cols`, rows [T, padded) zero: FlashInfer's shared-expert
    // staging (`quantize_mxfp8_block32` + `to_blocked`), bit for bit
    uint8_t* xs_q;
    uint8_t* xs_sf;
    int64_t xs_q_stride_m;
    uint32_t xs_sf_padded_rows, xs_sf_padded_cols;

    // The official route planes: int32 physical ids (-1 = no route) and fp32 scores, column stride 1.
    // Rows [num_tokens, route_capacity) of the id plane are set to -1 every launch (FlashInfer's fused tail mask).
    int32_t* topk_idx_out;
    float* topk_weights_out;
    int64_t topk_idx_stride_m, topk_weights_stride_m;
    uint32_t route_capacity;

    // Padding mask [T] (nullptr: no padding). Padded tokens route to -1; their scores keep the gate's value
    // (FlashInfer stages with zero_masked_scores=False).
    const bool* is_padding;
    int64_t is_padding_stride_m;

    // EPLB state of this layer (all nullptr without EPLB), as `eplb_map_to_physical_and_record`:
    // logical_to_physical [num_logical, map_stride] int64, replica_count [num_logical] int64, expert_load [num_physical] int32,
    // record_enabled scalar bool, num_unpadded_tokens scalar int32 (nullptr: no limit)
    const int64_t* logical_to_physical;
    const int64_t* replica_count;
    int32_t* expert_load;
    const bool* record_enabled;
    const int32_t* num_unpadded_tokens;
    int64_t map_stride;
    uint32_t num_logical, num_physical;
};

// EPLB cache in the gate's 4 KiB metadata cache, after the bias (and the image bias): one uint16 per logical expert,
// replica column 0 in the low 11 bits (0x7ff = no replica) and the replica count in the high 5 bits.
static constexpr uint32_t kFront2Col0Bits = 11, kFront2Col0Mask = (1u << kFront2Col0Bits) - 1u;
static constexpr uint32_t kFront2MaxPhysical = kFront2Col0Mask, kFront2MaxReplicas = 31;

CUTLASS_HOST_DEVICE constexpr uint32_t front2_eplb_cache_offset_bytes(const uint32_t num_routed_experts, const bool has_image_bias) {
    return num_routed_experts * static_cast<uint32_t>(sizeof(float)) * (has_image_bias ? 2u : 1u);
}
CUTLASS_HOST_DEVICE constexpr bool front2_eplb_cache_fits(const uint32_t num_routed_experts, const bool has_image_bias) {
    return front2_eplb_cache_offset_bytes(num_routed_experts, has_image_bias) + num_routed_experts * 2u <= kNumMaxMetadataCacheBytes;
}

} // namespace deep_gemm::layout::mega_gate
