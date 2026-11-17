#pragma once

// FRONT2: the parts of the fused MoE front for the official cutedsl_megamoe gen-phase kernel that the stock BF16 mega gate does
// not have. One launch = DeepGEMM's sm100_bf16_mega_gate (GEMM, split, sqrt-softplus, bias / image bias, top-k: unchanged) plus
// what the official chain runs after it as separate launches:
//   - the router's EPLB map + load record (vLLM `eplb_map_to_physical_and_record`: Knuth hash of the token index, one atomic add
//     per routed (token, slot) when recording is on and the token is not padding) and the int32 narrowing of the ids;
//   - the official entry's activation staging (FlashInfer `fused_quant_stage` -> cutedsl_megamoe `DataPreprocess`), written straight
//     into the official kernel's plain-row planes, and its route repack (int32 ids, fp32 scores, -1 capacity tail).
// The NVFP4 arithmetic is DataPreprocess's PTX instruction for instruction (read from its sm_107a PTX, CuTe DSL 4.8.1):
//   amax = max.f32 fold of abs.f32 from +0; s = mul.f32(multiplier, amax); sf = cvt.rn.satfinite.e4m3x2.f32(0, s) (low byte);
//   sf32 = cvt.f32.f16(cvt.rn.f16x2.e4m3x2(sf)); acc = mul.f32(min.f32(mul.f32(norm, rcp.approx.ftz.f32(sf32)), FLT_MAX),
//   min.f32(mul.f32(sf32, 1e30), 1)); q = mul.f32(x, acc); byte = cvt.rn.satfinite.e2m1x2.f32(q[2i + 1], q[2i]).
// Every operation is inline PTX so no compiler flag (ftz, fast math, contraction) can change a bit.

#include <cstdint>

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/layout/mega_gate_front2.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::epilogue::mega_gate_front2 {

using namespace layout::mega_gate;

CUTLASS_DEVICE float f32_from_bf16(const uint16_t& bits) {
    float r;
    asm("cvt.f32.bf16 %0, %1;" : "=f"(r) : "h"(bits));
    return r;
}
CUTLASS_DEVICE float f32_abs(const float& a) {
    float r;
    asm("abs.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
CUTLASS_DEVICE float f32_max(const float& a, const float& b) {
    float r;
    asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
CUTLASS_DEVICE float f32_min(const float& a, const float& b) {
    float r;
    asm("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
CUTLASS_DEVICE float f32_mul(const float& a, const float& b) {
    float r;
    asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
CUTLASS_DEVICE float f32_rcp_approx_ftz(const float& a) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
// e4m3 of `s` in the low byte (cvt places its first source in the upper byte; DataPreprocess passes +0 there)
CUTLASS_DEVICE uint32_t e4m3_byte(const float& s) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(s));
    return static_cast<uint32_t>(r) & 0xffu;
}
CUTLASS_DEVICE float f32_from_e4m3_byte(const uint32_t& byte) {
    uint32_t h2;
    const uint16_t in = static_cast<uint16_t>(byte & 0xffu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(in));
    float r;
    const uint16_t lo = static_cast<uint16_t>(h2 & 0xffffu);
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(lo));
    return r;
}
// Eight e2m1 values -> four bytes; byte i = (q[2i + 1] << 4) | q[2i]
CUTLASS_DEVICE uint32_t e2m1x8(const float* q) {
    uint32_t r;
    asm("{\n"
        "  .reg .b8 b0, b1, b2, b3;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "  mov.b32 %0, {b0, b1, b2, b3};\n"
        "}\n"
        : "=r"(r)
        : "f"(q[0]), "f"(q[1]), "f"(q[2]), "f"(q[3]), "f"(q[4]), "f"(q[5]), "f"(q[6]), "f"(q[7]));
    return r;
}

// ---- the shared expert's MXFP8 input: FlashInfer's shared-expert staging (`quantize_mxfp8_block32` + `to_blocked`, 128x4):
// amax = max.bf16x2 tree of (word & 0x7fff7fff), then max.f32 of the two bf16 halves and of the two 16-element halves;
// e = ceil-ue8m0(max.f32(mul.f32(amax, 1 / 448), 2^-126)) (its exact bit construction; zero and tiny blocks get e = 1, the
// floor of FlashInfer's `_ceil_log2_exponent`); inv = 2^(127 - e) = 1 / scale exactly;
// byte = cvt.rn.satfinite.e4m3x2.f32 of max.f32(min.f32(mul.f32(x, inv), 448), -448) (= e4m3(x / scale), |x / scale| <= 448)
CUTLASS_DEVICE uint32_t bf16x2_max(const uint32_t& a, const uint32_t& b) {
    uint32_t r;
    asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
CUTLASS_DEVICE float bf16x2_hmax_to_f32(const uint32_t& x) {
    return f32_max(__uint_as_float((x & 0xffffu) << 16), __uint_as_float((x >> 16) << 16));
}
CUTLASS_DEVICE float bf16x8_max_abs(const uint32_t* w) {
    const uint32_t m01 = bf16x2_max(w[0] & 0x7fff7fffu, w[1] & 0x7fff7fffu), m23 = bf16x2_max(w[2] & 0x7fff7fffu, w[3] & 0x7fff7fffu);
    const uint32_t m45 = bf16x2_max(w[4] & 0x7fff7fffu, w[5] & 0x7fff7fffu), m67 = bf16x2_max(w[6] & 0x7fff7fffu, w[7] & 0x7fff7fffu);
    return bf16x2_hmax_to_f32(bf16x2_max(bf16x2_max(m01, m23), bf16x2_max(m45, m67)));
}
CUTLASS_DEVICE uint32_t float_to_ue8m0_fast(const float& value) {
    uint32_t r;
    asm("{\n"
        "  .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
        "  .reg .u32 bits, exp_biased, mantissa, bump, result;\n"
        "  setp.le.f32 p_zero, %1, 0f00000000;\n"
        "  mov.b32 bits, %1;\n"
        "  shr.b32 exp_biased, bits, 23;\n"
        "  and.b32 exp_biased, exp_biased, 255;\n"
        "  and.b32 mantissa, bits, 0x7FFFFF;\n"
        "  setp.ne.u32 p_has_mant, mantissa, 0;\n"
        "  selp.u32 bump, 1, 0, p_has_mant;\n"
        "  setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
        "  setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
        "  and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
        "  @p_tiny_sub mov.u32 bump, 0;\n"
        "  add.u32 result, exp_biased, bump;\n"
        "  setp.gt.u32 p_ovf, result, 254;\n"
        "  selp.u32 result, 254, result, p_ovf;\n"
        "  selp.u32 %0, 0, result, p_zero;\n"
        "}\n" : "=r"(r) : "f"(value));
    return r;
}
CUTLASS_DEVICE float ue8m0_to_inv_scale_fast(const uint32_t& e) {
    float r;
    asm("{\n"
        "  .reg .s32 new_exp;\n"
        "  .reg .b32 float_bits;\n"
        "  .reg .pred p_zero;\n"
        "  setp.eq.u32 p_zero, %1, 0;\n"
        "  sub.s32 new_exp, 254, %1;\n"
        "  max.s32 new_exp, new_exp, 0;\n"
        "  shl.b32 float_bits, new_exp, 23;\n"
        "  mov.b32 %0, float_bits;\n"
        "  @p_zero mov.b32 %0, 0;\n"
        "}\n" : "=f"(r) : "r"(e));
    return r;
}
// Two bf16 (one word) -> two e4m3 bytes (low byte = low bf16)
CUTLASS_DEVICE uint32_t bf16x2_to_e4m3x2_scaled(const uint32_t& w, const float& inv) {
    float f0 = f32_mul(__uint_as_float((w & 0xffffu) << 16), inv), f1 = f32_mul(__uint_as_float((w >> 16) << 16), inv);
    f0 = f32_max(f32_min(f0, __uint_as_float(0x43e00000u)), __uint_as_float(0xc3e00000u));
    f1 = f32_max(f32_min(f1, __uint_as_float(0x43e00000u)), __uint_as_float(0xc3e00000u));
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(f1), "f"(f0));
    return static_cast<uint32_t>(r);
}
CUTLASS_DEVICE uint32_t sf_index_swizzled_128x4(const uint32_t& row, const uint32_t& col, const uint32_t& padded_cols) {
    return (col % 4) + (col / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (row / 128) * 128 * padded_cols;
}
// One 32-column MXFP8 block of X_s from the lane's 16 bf16 words (the same registers the NVFP4 staging read)
CUTLASS_DEVICE void stage_shared_mxfp8_block(const Front2Args& args, const uint32_t& token_idx, const uint32_t& col_block,
                                             const uint32_t* words) {
    const float amax = f32_max(bf16x8_max_abs(words), bf16x8_max_abs(words + 8));
    const uint32_t e = float_to_ue8m0_fast(f32_max(f32_mul(amax, __uint_as_float(0x3b124925u)), __uint_as_float(0x00800000u)));
    const float inv = ue8m0_to_inv_scale_fast(e);
    uint32_t packed[8];
    #pragma unroll
    for (uint32_t q = 0; q < 8; ++ q)
        packed[q] = bf16x2_to_e4m3x2_scaled(words[2 * q], inv) | (bf16x2_to_e4m3x2_scaled(words[2 * q + 1], inv) << 16);
    auto dst = reinterpret_cast<uint4*>(args.xs_q + token_idx * args.xs_q_stride_m + col_block * 32);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    args.xs_sf[sf_index_swizzled_128x4(token_idx, col_block, args.xs_sf_padded_cols)] = static_cast<uint8_t>(e);
}

// ---- the routed MXFP8 staging of the SM107 MXFP4 x MXFP8 entry: FlashInfer's `fused_quant_stage(quant_type="mxfp8_e4m3",
// mxfp8_normal_scales=True)` (the normal-scale DataPreprocess), instruction for instruction from its sm_107a PTX (CuTe DSL 4.8.1):
// amax = max.f32 fold of abs.f32 from +0 (two 16-column halves joined by max.f32); u = max.f32(mul.f32(amax, 1 / 448), 2^-126);
// cvt.rp.satfinite.ue8m0x2.f32 (+0, u); scale = cvt.f32.bf16(cvt.rn.bf16x2.ue8m0x2(...)) (low half); the stored byte =
// cvt.rp.ue8m0x2.bf16x2({scale_bf16, 0}) (low);
// acc = min.f32(rcp.approx.ftz.f32(scale), FLT_MAX); byte pair = cvt.rn.satfinite.e4m3x2.f32(mul.f32(odd, acc), mul.f32(even, acc)).
CUTLASS_DEVICE uint32_t ue8m0_byte_rp_and_scale(const float& u, float& scale) {
    uint32_t packed;
    asm("{\n"
        "  .reg .b16 bf_lo;\n"
        "  cvt.rp.satfinite.ue8m0x2.f32 bf_lo, 0f00000000, %1;\n"
        "  cvt.rn.bf16x2.ue8m0x2 %0, bf_lo;\n"
        "}\n" : "=r"(packed) : "f"(u));
    const uint16_t bf16_lo = static_cast<uint16_t>(packed & 0xffffu);
    asm("cvt.f32.bf16 %0, %1;" : "=f"(scale) : "h"(bf16_lo));
    // the stored byte: `scale.to(Float8E8M0FNU)` = cvt.rp.ue8m0x2.bf16x2 of {scale_bf16 (low), 0}, low byte (its st.global.b8)
    uint16_t sf2;
    const uint32_t pair = static_cast<uint32_t>(bf16_lo);
    asm("cvt.rp.ue8m0x2.bf16x2 %0, %1;" : "=h"(sf2) : "r"(pair));
    return static_cast<uint32_t>(sf2) & 0xffu;
}
CUTLASS_DEVICE uint32_t f32x2_to_e4m3x2(const float& lo, const float& hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return static_cast<uint32_t>(r);
}
// One 32-column routed MXFP8 block from the lane's 16 bf16 words: 32 data bytes + returns the scale byte
CUTLASS_DEVICE uint32_t routed_mxfp8_block(const uint32_t* words, uint32_t (&packed)[8]) {
    float v[32];
    #pragma unroll
    for (uint32_t w = 0; w < 16; ++ w) {
        v[2 * w] = f32_from_bf16(static_cast<uint16_t>(words[w] & 0xffffu));
        v[2 * w + 1] = f32_from_bf16(static_cast<uint16_t>(words[w] >> 16));
    }
    float amax0 = 0.0f, amax1 = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 16; ++ i) {
        amax0 = f32_max(amax0, f32_abs(v[i]));
        amax1 = f32_max(amax1, f32_abs(v[16 + i]));
    }
    const float u = f32_max(f32_mul(f32_max(amax0, amax1), __uint_as_float(0x3b124925u)), __uint_as_float(0x00800000u));
    float scale;
    const uint32_t sf = ue8m0_byte_rp_and_scale(u, scale);
    const float acc = f32_min(f32_rcp_approx_ftz(scale), __uint_as_float(0x7f7fffffu));
    #pragma unroll
    for (uint32_t q = 0; q < 8; ++ q) {
        const uint32_t b01 = f32x2_to_e4m3x2(f32_mul(v[4 * q], acc), f32_mul(v[4 * q + 1], acc));
        const uint32_t b23 = f32x2_to_e4m3x2(f32_mul(v[4 * q + 2], acc), f32_mul(v[4 * q + 3], acc));
        packed[q] = b01 | (b23 << 16);
    }
    return sf;
}

// One 1024-column segment of MXFP8 staging by a whole warp: lane l owns columns 32l .. 32l + 31 = one UE8M0 block: a 64-byte load,
// two 16-byte data stores, one scale byte (plain row); X_s (the fused shared expert's swizzled MXFP8) from the same registers
template <bool kWithSharedX>
CUTLASS_DEVICE void stage_mxfp8_segment_1024(const Front2Args& args, const uint32_t& token_idx, const uint32_t& seg_idx,
                                             const uint32_t& lane_idx) {
    const auto src = reinterpret_cast<const uint4*>(args.x + token_idx * args.x_stride_m + seg_idx * 1024 + lane_idx * 32);
    uint4 raw[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i)
        raw[i] = __ldg(src + i);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(raw);
    uint32_t packed[8];
    const uint32_t sf = routed_mxfp8_block(words, packed);
    auto dst = reinterpret_cast<uint4*>(args.x_q + token_idx * args.x_q_stride_m + seg_idx * 1024 + lane_idx * 32);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    args.x_sf[token_idx * args.x_sf_stride_m + seg_idx * 32 + lane_idx] = static_cast<uint8_t>(sf);
    if constexpr (kWithSharedX)
        stage_shared_mxfp8_block(args, token_idx, seg_idx * 32 + lane_idx, words);
}

// One NVFP4 16-column block: returns the E4M3 scale byte, writes the 8 data bytes as two words
CUTLASS_DEVICE uint32_t nvfp4_block(const float (&v)[16], const float& norm_const, const float& scale_multiplier,
                                    uint32_t& w0, uint32_t& w1) {
    float amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 16; ++ i)
        amax = f32_max(amax, f32_abs(v[i]));
    const uint32_t sf = e4m3_byte(f32_mul(scale_multiplier, amax));
    const float sf32 = f32_from_e4m3_byte(sf);
    const float acc = f32_mul(f32_min(f32_mul(norm_const, f32_rcp_approx_ftz(sf32)), __uint_as_float(0x7f7fffffu)),
                              f32_min(f32_mul(sf32, __uint_as_float(0x7149f2cau)), 1.0f));
    float q[16];
    #pragma unroll
    for (uint32_t i = 0; i < 16; ++ i)
        q[i] = f32_mul(v[i], acc);
    w0 = e2m1x8(q);
    w1 = e2m1x8(q + 8);
    return sf;
}

// One 1024-column segment of NVFP4 staging by a whole warp: lane l owns columns 32l .. 32l + 31 (two 16-column blocks): a 64-byte
// load, one 16-byte data store, one 2-byte scale store
template <bool kWithSharedX>
CUTLASS_DEVICE void stage_nvfp4_segment_1024(const Front2Args& args, const uint32_t& token_idx, const uint32_t& seg_idx,
                                             const uint32_t& lane_idx) {
    const auto src = reinterpret_cast<const uint4*>(args.x + token_idx * args.x_stride_m + seg_idx * 1024 + lane_idx * 32);
    uint4 raw[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i)
        raw[i] = __ldg(src + i);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(raw);
    float v0[16], v1[16];
    #pragma unroll
    for (uint32_t w = 0; w < 8; ++ w) {
        v0[2 * w] = f32_from_bf16(static_cast<uint16_t>(words[w] & 0xffffu));
        v0[2 * w + 1] = f32_from_bf16(static_cast<uint16_t>(words[w] >> 16));
        v1[2 * w] = f32_from_bf16(static_cast<uint16_t>(words[8 + w] & 0xffffu));
        v1[2 * w + 1] = f32_from_bf16(static_cast<uint16_t>(words[8 + w] >> 16));
    }
    uint32_t d[4];
    const uint32_t sf0 = nvfp4_block(v0, args.norm_const, args.scale_multiplier, d[0], d[1]);
    const uint32_t sf1 = nvfp4_block(v1, args.norm_const, args.scale_multiplier, d[2], d[3]);
    *reinterpret_cast<uint4*>(args.x_q + token_idx * args.x_q_stride_m + seg_idx * 512 + lane_idx * 16) = make_uint4(d[0], d[1], d[2], d[3]);
    *reinterpret_cast<uint16_t*>(args.x_sf + token_idx * args.x_sf_stride_m + seg_idx * 64 + lane_idx * 2) =
        static_cast<uint16_t>(sf0 | (sf1 << 8));
    // The lane's 32 columns are exactly one MXFP8 scale block of the shared expert's input
    if constexpr (kWithSharedX)
        stage_shared_mxfp8_block(args, token_idx, seg_idx * 32 + lane_idx, words);
}

// Staging of rows [token_begin, token_end) by the gate warps of one logical CTA: (token, 1024-column segment) units dealt over the
// worker group's logical CTAs, then over the CTA's gate warps
template <uint32_t kHidden, uint32_t kNumGateWarps, uint32_t kNumLogicalCtas, Front2Format kFormat, bool kWithSharedX>
CUTLASS_DEVICE void stage_front2(const Front2Args& args, const uint32_t& token_begin, const uint32_t& token_end,
                                 const uint32_t& logical_cta_rank, const uint32_t& gate_warp_idx, const uint32_t& lane_idx) {
    DG_STATIC_ASSERT(kFormat == Front2Format::kNvfp4 or kFormat == Front2Format::kMxfp8, "Unknown FRONT2 staging format");
    DG_STATIC_ASSERT(kHidden % 1024 == 0, "FRONT2 staging works in 1024-column segments");
    constexpr uint32_t kUnitStride = kNumGateWarps * kNumLogicalCtas;
    constexpr uint32_t kNumSegments = kHidden / 1024;
    const uint32_t num_units = (token_end - token_begin) * kNumSegments;
    for (uint32_t unit_idx = logical_cta_rank + gate_warp_idx * kNumLogicalCtas; unit_idx < num_units; unit_idx += kUnitStride) {
        if constexpr (kFormat == Front2Format::kMxfp8)
            stage_mxfp8_segment_1024<kWithSharedX>(args, token_begin + unit_idx / kNumSegments, unit_idx % kNumSegments, lane_idx);
        else
            stage_nvfp4_segment_1024<kWithSharedX>(args, token_begin + unit_idx / kNumSegments, unit_idx % kNumSegments, lane_idx);
    }
}

// Gate-warp prologue: the packed EPLB cache (replica column 0 | replica count << 11) after the bias / image-bias cache
template <uint32_t kNumRoutedExperts, uint32_t kNumGateThreads, bool kHasImageBias>
CUTLASS_DEVICE void fill_eplb_cache(const Front2Args& args, uint8_t* metadata_cache_bytes, const uint32_t& gate_thread_idx) {
    DG_STATIC_ASSERT(front2_eplb_cache_fits(kNumRoutedExperts, kHasImageBias), "EPLB cache exceeds the metadata cache");
    auto cache = reinterpret_cast<uint16_t*>(metadata_cache_bytes + front2_eplb_cache_offset_bytes(kNumRoutedExperts, kHasImageBias));
    for (uint32_t i = gate_thread_idx; i < kNumRoutedExperts; i += kNumGateThreads) {
        uint32_t packed = kFront2Col0Mask | (1u << kFront2Col0Bits);
        if (i < args.num_logical) {
            const auto col0 = __ldg(args.logical_to_physical + i * args.map_stride);
            const auto reps = __ldg(args.replica_count + i);
            packed = (col0 < 0 ? kFront2Col0Mask : static_cast<uint32_t>(col0)) | (static_cast<uint32_t>(reps) << kFront2Col0Bits);
        }
        cache[i] = static_cast<uint16_t>(packed);
    }
}

// The stock `store_topk_token` weights (verbatim, so they keep their bits), then the router's EPLB map + record and the official
// entry's route repack: int32 physical ids (-1 for padding / unmapped), fp32 scores, into the official planes.
template <uint32_t kNumRoutedExperts, uint32_t kNumTopk, bool kHasImageBias, bool kHasEplb, bool kHasPadding>
CUTLASS_DEVICE void store_front2_token(const RoutingArgs& routing_args, const Front2Args& args, const uint8_t* metadata_cache_bytes,
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
    bool is_padding = false;
    if constexpr (kHasPadding)
        is_padding = args.is_padding[global_token_idx * args.is_padding_stride_m];

    int64_t physical_expert_idx = is_padding ? -1 : logical_expert_idx;
    if constexpr (kHasEplb) {
        physical_expert_idx = -1;
        if (not is_padding and logical_expert_idx >= 0 and logical_expert_idx < static_cast<int64_t>(args.num_logical)) {
            const auto cache = reinterpret_cast<const uint16_t*>(metadata_cache_bytes +
                                                                 front2_eplb_cache_offset_bytes(kNumRoutedExperts, kHasImageBias));
            const uint32_t packed = cache[logical_expert_idx];
            const auto num_replicas = static_cast<uint64_t>(packed >> kFront2Col0Bits);
            DG_TRAP_ONLY_DEVICE_ASSERT(num_replicas > 0);
            // floor(2^32 / phi): `(token * 2654435769) & 0xFFFFFFFF` in int64, then `% replicas`
            const auto hashed = (static_cast<uint64_t>(global_token_idx) * 2654435769ull) & 0xffffffffull;
            const auto slot = hashed % num_replicas;
            const uint32_t col0 = packed & kFront2Col0Mask;
            physical_expert_idx = slot == 0 ? (col0 == kFront2Col0Mask ? -1 : static_cast<int64_t>(col0)) :
                                  __ldg(args.logical_to_physical + logical_expert_idx * args.map_stride + static_cast<int64_t>(slot));
        }
        if (record_on and physical_expert_idx >= 0 and physical_expert_idx < static_cast<int64_t>(args.num_physical) and
            global_token_idx < num_unpadded)
            atomicAdd(args.expert_load + physical_expert_idx, 1);
    }
    args.topk_idx_out[global_token_idx * args.topk_idx_stride_m + lane_idx] = static_cast<int32_t>(physical_expert_idx);
    args.topk_weights_out[global_token_idx * args.topk_weights_stride_m + lane_idx] = selected_weight;
}

// FlashInfer's fused tail mask: rows [num_tokens, route_capacity) of the id plane route nowhere (-1); with X_s, the scale rows
// [num_tokens, padded) of its swizzled plane are zero (FlashInfer's shared staging clears those rows; its data rows are cleared by
// FlashInfer's stage_external)
template <uint32_t kNumTopk, uint32_t kNumGateThreads, bool kWithSharedX>
CUTLASS_DEVICE void mask_route_tail(const Front2Args& args, const uint32_t& num_tokens, const uint32_t& gate_thread_idx) {
    if (args.route_capacity > num_tokens) {
        const uint32_t num_slots = (args.route_capacity - num_tokens) * kNumTopk;
        for (uint32_t i = gate_thread_idx; i < num_slots; i += kNumGateThreads)
            args.topk_idx_out[(num_tokens + i / kNumTopk) * args.topk_idx_stride_m + i % kNumTopk] = -1;
    }
    if constexpr (kWithSharedX) {
        if (args.xs_sf_padded_rows > num_tokens) {
            const uint32_t num_sf = (args.xs_sf_padded_rows - num_tokens) * args.xs_sf_padded_cols;
            for (uint32_t i = gate_thread_idx; i < num_sf; i += kNumGateThreads)
                args.xs_sf[sf_index_swizzled_128x4(num_tokens + i / args.xs_sf_padded_cols, i % args.xs_sf_padded_cols,
                                                   args.xs_sf_padded_cols)] = 0;
        }
    }
}

} // namespace deep_gemm::epilogue::mega_gate_front2
