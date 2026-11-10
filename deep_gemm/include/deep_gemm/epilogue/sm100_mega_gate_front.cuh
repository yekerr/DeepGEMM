#pragma once

// The parts of the fused MoE front that the stock BF16 mega gate does not have.
//
// The fused front = DeepGEMM's sm100_bf16_mega_gate (gate GEMM + sqrt-softplus + bias + top-k, unchanged) plus, in the
// same launch, what vLLM's `_prepare_megamoe_inputs_kernel` (vllm/model_executor/layers/fused_moe/prepare_megamoe.py)
// does afterwards:
//   - MXFP8 staging of the BF16 hidden states into the MegaMoE symmetric buffer (e4m3 values, one UE8M0 exponent per 32
//     columns, four exponents packed little-endian per int32 per 128 columns), plus the shared-expert copy of the packed
//     scales in the MN-major UTCCP page layout of the call's BLOCK_M;
//   - the EPLB logical -> physical replica map (Knuth hash of the token index, `eplb_map_and_record`) and the expert-load
//     record (one atomic add per routed (token, slot) when recording is on and the token is not padding);
//   - the padding mask (padded tokens route to -1 with weight 0);
//   - the int64 / fp32 routing written straight into the symmetric buffer's topk_idx / topk_weights.
// Every fp32 operation of the staging is the instruction Triton emits for the stock kernel: an fp32 '/' is div.full.f32,
// '*' is mul.rn.f32, the fp8 cast is cvt.rn.satfinite.e4m3x2.f32, the max reductions are NaN-suppressing max.f32.

// Arch-neutral on purpose (no tcgen05): the staging and EPLB parts also build for sm_90 so a Hopper functional test can
// compare them with vLLM's Triton staging kernel bit for bit (`cvt ... e4m3x2` needs sm_89+).
#include <cstdint>

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/layout/mega_gate_front.cuh>
#include <deep_gemm/ptx/utils.cuh>


namespace deep_gemm::epilogue::mega_gate_front {

using namespace layout::mega_gate;

// Triton lowers an fp32 '/' to div.full.f32; the staging scales must use the same instruction to match bit for bit
CUTLASS_DEVICE float div_full(const float& a, const float& b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Two fp32 -> e4m3 bytes, `lo` in the low byte (cvt puts its first source operand in the upper byte)
CUTLASS_DEVICE uint32_t cvt_e4m3x2(const float& lo, const float& hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

CUTLASS_DEVICE uint32_t shared_sf_row(const uint32_t& token_idx, const uint32_t& shared_block_m) {
    const auto m_block_idx = token_idx / shared_block_m, m_in_block = token_idx % shared_block_m;
    const auto aligned_block_m = math::ceil_div(shared_block_m, 128u) * 128u;
    return m_block_idx * aligned_block_m + (m_in_block / 128) * 128 + (m_in_block % 32) * 4 + (m_in_block % 128) / 32;
}

// One 128-column chunk of MXFP8 staging by a whole warp (lane l holds columns 4l .. 4l + 3, a 32-column group is 8 lanes).
// Used for hidden sizes that are not a multiple of 1024.
template <bool kHasSharedSf>
CUTLASS_DEVICE void stage_chunk_128(const FrontArgs& args, const uint32_t& token_idx, const uint32_t& chunk_idx, const uint32_t& lane_idx) {
    const auto raw = __ldg(reinterpret_cast<const uint2*>(args.x + token_idx * args.x_stride_m + chunk_idx * 128) + lane_idx);
    // BF16 -> FP32 is exact (Triton's `.to(tl.float32)`)
    const float v0 = __uint_as_float(raw.x << 16), v1 = __uint_as_float(raw.x & 0xffff0000u);
    const float v2 = __uint_as_float(raw.y << 16), v3 = __uint_as_float(raw.y & 0xffff0000u);
    // amax over the 32-column group: |x| then NaN-suppressing max (Triton tl.max / tl.maximum)
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, 1.0e-4f);
    const float scale = div_full(amax, 448.0f);
    const uint32_t scale_bits = __float_as_uint(scale);
    uint32_t scale_exp = ((scale_bits >> 23) & 0xffu) + ((scale_bits & 0x7fffffu) != 0 ? 1u : 0u);
    scale_exp = cute::min(cute::max(scale_exp, 1u), 254u);
    const float inv_scale = div_full(1.0f, __uint_as_float(scale_exp << 23));
    const uint32_t packed_fp8 = cvt_e4m3x2(__fmul_rn(v0, inv_scale), __fmul_rn(v1, inv_scale)) |
                                cvt_e4m3x2(__fmul_rn(v2, inv_scale), __fmul_rn(v3, inv_scale)) << 16;
    reinterpret_cast<uint32_t*>(args.x_fp8 + token_idx * args.x_fp8_stride_m + chunk_idx * 128)[lane_idx] = packed_fp8;

    // Four UE8M0 exponents per int32, group g (lanes 8g .. 8g + 7) in byte g
    const uint32_t e0 = __shfl_sync(0xffffffff, scale_exp, 0), e1 = __shfl_sync(0xffffffff, scale_exp, 8);
    const uint32_t e2 = __shfl_sync(0xffffffff, scale_exp, 16), e3 = __shfl_sync(0xffffffff, scale_exp, 24);
    if (lane_idx == 0) {
        const auto packed_sf = static_cast<int32_t>(e0 | e1 << 8 | e2 << 16 | e3 << 24);
        args.x_sf[token_idx * args.x_sf_stride_m + chunk_idx * args.x_sf_stride_k] = packed_sf;
        if constexpr (kHasSharedSf) {
            args.shared_x_sf[static_cast<int64_t>(shared_sf_row(token_idx, args.shared_block_m)) * args.shared_x_sf_stride_m +
                             chunk_idx * args.shared_x_sf_stride_k] = packed_sf;
        }
    }
}

// One 1024-column segment of MXFP8 staging by a whole warp: lane l owns the 32-column scale group l of the segment, so the
// group's amax is computed inside the lane (no shuffles), the lane loads 64 contiguous bytes (4 x 16 B) and stores 32
// contiguous fp8 bytes (2 x 16 B), and the 8 packed scale words of the segment are written by 8 lanes at once.
// Per element it runs the same instructions as stage_chunk_128 (exact BF16 -> FP32, |x|, NaN-suppressing max, div.full, the
// UE8M0 rounding, mul.rn, cvt.rn.satfinite.e4m3x2); the NaN-suppressing max of non-negative values does not depend on the
// order it is folded in, so the bytes are the same as stage_chunk_128's and as vLLM's Triton staging kernel.
template <bool kHasSharedSf>
CUTLASS_DEVICE void stage_segment_1024(const FrontArgs& args, const uint32_t& token_idx, const uint32_t& seg_idx, const uint32_t& lane_idx) {
    const auto src = reinterpret_cast<const uint4*>(args.x + token_idx * args.x_stride_m + seg_idx * 1024 + lane_idx * 32);
    uint4 raw[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i)
        raw[i] = __ldg(src + i);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(raw);
    // amax over the lane's 32 values; starting from +0 is exact because every |x| is >= +0 or NaN (suppressed)
    float amax = 0.0f;
    #pragma unroll
    for (uint32_t w = 0; w < 16; ++ w) {
        amax = fmaxf(amax, fabsf(__uint_as_float(words[w] << 16)));
        amax = fmaxf(amax, fabsf(__uint_as_float(words[w] & 0xffff0000u)));
    }
    amax = fmaxf(amax, 1.0e-4f);
    const float scale = div_full(amax, 448.0f);
    const uint32_t scale_bits = __float_as_uint(scale);
    uint32_t scale_exp = ((scale_bits >> 23) & 0xffu) + ((scale_bits & 0x7fffffu) != 0 ? 1u : 0u);
    scale_exp = cute::min(cute::max(scale_exp, 1u), 254u);
    const float inv_scale = div_full(1.0f, __uint_as_float(scale_exp << 23));
    uint32_t packed[8];
    #pragma unroll
    for (uint32_t q = 0; q < 8; ++ q) {
        const uint32_t wa = words[2 * q], wb = words[2 * q + 1];
        packed[q] = cvt_e4m3x2(__fmul_rn(__uint_as_float(wa << 16), inv_scale), __fmul_rn(__uint_as_float(wa & 0xffff0000u), inv_scale)) |
                    cvt_e4m3x2(__fmul_rn(__uint_as_float(wb << 16), inv_scale), __fmul_rn(__uint_as_float(wb & 0xffff0000u), inv_scale)) << 16;
    }
    auto dst = reinterpret_cast<uint4*>(args.x_fp8 + token_idx * args.x_fp8_stride_m + seg_idx * 1024 + lane_idx * 32);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);

    // Four UE8M0 exponents per int32: chunk c of the segment = lanes 4c .. 4c + 3, group g = lane 4c + g in byte g
    const uint32_t e1 = __shfl_down_sync(0xffffffff, scale_exp, 1);
    const uint32_t e2 = __shfl_down_sync(0xffffffff, scale_exp, 2);
    const uint32_t e3 = __shfl_down_sync(0xffffffff, scale_exp, 3);
    if (lane_idx % 4 == 0) {
        const uint32_t chunk_idx = seg_idx * 8 + lane_idx / 4;
        const auto packed_sf = static_cast<int32_t>(scale_exp | e1 << 8 | e2 << 16 | e3 << 24);
        args.x_sf[token_idx * args.x_sf_stride_m + chunk_idx * args.x_sf_stride_k] = packed_sf;
        if constexpr (kHasSharedSf) {
            args.shared_x_sf[static_cast<int64_t>(shared_sf_row(token_idx, args.shared_block_m)) * args.shared_x_sf_stride_m +
                             chunk_idx * args.shared_x_sf_stride_k] = packed_sf;
        }
    }
}

// MXFP8 staging of rows [token_begin, token_end) by the gate warps of one logical CTA. The block's units are dealt round-robin over
// the worker group's logical CTAs, then over the CTA's gate warps.
//
// The unit is a (token, 1024-column segment) when 1024 divides the hidden size (5 per row at H 5120), else a (token,
// 128-column chunk). Staging by 128-column chunks cost ~50-90 ns per chunk per CTA whatever the number of gate warps or
// loads in flight (EP1, front minus a no-staging control = +1.3 / +1.9 / +5.1 / +5.0 / +9.2 / +17.8 us at
// T 24 / 114 / 234 / 342 / 684 / 1824, and 8-deep load batching did not move it), i.e. it is bound by per-chunk
// instruction and memory-instruction issue (an 8-byte load, a 128-byte store, two 4-byte stores, 7 shuffles and two
// divides per 128 columns), not by load latency. The segment unit moves 8x the bytes per memory instruction and drops the
// shuffles.
template <uint32_t kHidden, uint32_t kNumGateWarps, uint32_t kNumLogicalCtas, bool kHasSharedSf>
CUTLASS_DEVICE void stage_mxfp8(const FrontArgs& args, const uint32_t& token_begin, const uint32_t& token_end,
                                const uint32_t& logical_cta_rank, const uint32_t& gate_warp_idx, const uint32_t& lane_idx) {
    DG_STATIC_ASSERT(kHidden % 128 == 0, "MXFP8 staging packs four 32-column scales per 128 columns");
    constexpr uint32_t kUnitStride = kNumGateWarps * kNumLogicalCtas;
    if constexpr (kHidden % 1024 == 0) {
        constexpr uint32_t kNumSegments = kHidden / 1024;
        const uint32_t num_units = (token_end - token_begin) * kNumSegments;
        for (uint32_t unit_idx = logical_cta_rank + gate_warp_idx * kNumLogicalCtas; unit_idx < num_units; unit_idx += kUnitStride)
            stage_segment_1024<kHasSharedSf>(args, token_begin + unit_idx / kNumSegments, unit_idx % kNumSegments, lane_idx);
    } else {
        constexpr uint32_t kNumChunks = kHidden / 128;
        const uint32_t num_units = (token_end - token_begin) * kNumChunks;
        #pragma unroll 2
        for (uint32_t unit_idx = logical_cta_rank + gate_warp_idx * kNumLogicalCtas; unit_idx < num_units; unit_idx += kUnitStride)
            stage_chunk_128<kHasSharedSf>(args, token_begin + unit_idx / kNumChunks, unit_idx % kNumChunks, lane_idx);
    }
}

// Gate-warp prologue of the fused front: the EPLB column-0 / replica-count cache next to the bias cache
template <uint32_t kNumRoutedExperts, uint32_t kNumGateThreads>
CUTLASS_DEVICE void fill_eplb_cache(const FrontArgs& args, uint8_t* metadata_cache_bytes, const uint32_t& gate_thread_idx) {
    DG_STATIC_ASSERT(front_eplb_cache_fits(kNumRoutedExperts), "EPLB cache exceeds the metadata cache");
    auto col0 = reinterpret_cast<int16_t*>(metadata_cache_bytes + front_eplb_cache_offset_bytes(kNumRoutedExperts));
    auto reps = reinterpret_cast<uint8_t*>(col0 + kNumRoutedExperts);
    for (uint32_t i = gate_thread_idx; i < kNumRoutedExperts; i += kNumGateThreads) {
        const bool valid = i < args.num_logical;
        col0[i] = valid ? static_cast<int16_t>(__ldg(args.logical_to_physical + i * args.map_stride)) : static_cast<int16_t>(-1);
        reps[i] = valid ? static_cast<uint8_t>(__ldg(args.replica_count + i)) : static_cast<uint8_t>(1);
    }
}

// The stock `store_topk_token` (weights bit for bit) followed by `_store_topk` of the staging kernel: padding, EPLB map,
// load record, symmetric-buffer stores. Lanes >= kNumTopk store nothing (no shared expert routed as a slot).
template <uint32_t kNumRoutedExperts, uint32_t kNumTopk, bool kHasEplb, bool kHasPadding, bool kHasLogicalOut>
CUTLASS_DEVICE void store_front_token(const RoutingArgs& routing_args, const FrontArgs& args, const uint8_t* metadata_cache_bytes,
                                      const bool& record_on, const uint32_t& num_unpadded,
                                      const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                      const int& selected_expert_idx, const float& selected_unbiased_score) {
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k slots exceed a warp");
    // NOTES: verbatim from `epilogue::mega_gate::store_topk_token` (5cdb1457), so the weights keep their bits
    constexpr uint32_t kSumWidth = kNumTopk <= 8 ? 8 : (kNumTopk <= 16 ? 16 : 32);
    const auto topk_sum = math::warp_reduce_sum<kSumWidth>(selected_unbiased_score) + 1e-20f;
    const auto selected_weight = lane_idx < kNumTopk ? selected_unbiased_score / topk_sum * routing_args.routed_scaling_factor : 1.0f;
    if (lane_idx >= kNumTopk)
        return;

    const auto logical_expert_idx = static_cast<int64_t>(selected_expert_idx);
    if constexpr (kHasLogicalOut)
        args.logical_topk_idx_out[global_token_idx * args.logical_topk_idx_stride_m + lane_idx] = logical_expert_idx;

    bool is_padding = false;
    if constexpr (kHasPadding)
        is_padding = args.is_padding[global_token_idx * args.is_padding_stride_m];

    int64_t physical_expert_idx = is_padding ? -1 : logical_expert_idx;
    if constexpr (kHasEplb) {
        if (not is_padding and logical_expert_idx >= 0 and logical_expert_idx < static_cast<int64_t>(args.num_logical)) {
            const auto col0 = reinterpret_cast<const int16_t*>(metadata_cache_bytes + front_eplb_cache_offset_bytes(kNumRoutedExperts));
            const auto reps = reinterpret_cast<const uint8_t*>(col0 + kNumRoutedExperts);
            const auto num_replicas = static_cast<uint64_t>(reps[logical_expert_idx]);
            DG_TRAP_ONLY_DEVICE_ASSERT(num_replicas > 0);
            // floor(2^32 / phi): `(token * 2654435769) & 0xFFFFFFFF` in int64, then `% replicas`
            const auto hashed = (static_cast<uint64_t>(global_token_idx) * 2654435769ull) & 0xffffffffull;
            const auto slot = hashed % num_replicas;
            physical_expert_idx = slot == 0 ? static_cast<int64_t>(col0[logical_expert_idx]) :
                                  __ldg(args.logical_to_physical + logical_expert_idx * args.map_stride + static_cast<int64_t>(slot));
        } else {
            physical_expert_idx = -1;
        }
        if (record_on and physical_expert_idx >= 0 and physical_expert_idx < static_cast<int64_t>(args.num_physical) and
            global_token_idx < num_unpadded)
            atomicAdd(args.expert_load + physical_expert_idx, 1);
    }
    args.topk_idx_out[global_token_idx * args.topk_idx_stride_m + lane_idx * args.topk_idx_stride_k] = physical_expert_idx;
    args.topk_weights_out[global_token_idx * args.topk_weights_stride_m + lane_idx * args.topk_weights_stride_k] =
        is_padding ? 0.0f : selected_weight;
}

} // namespace deep_gemm::epilogue::mega_gate_front
