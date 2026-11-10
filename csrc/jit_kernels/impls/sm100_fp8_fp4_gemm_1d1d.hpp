#pragma once

#include <format>
#include <torch/csrc/stable/ops.h>
#include "../../utils/torch_compat.hpp"

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "epilogue_class.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100FP8FP4Gemm1D1DRuntime final {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        deep_jit::cuda::LaunchOptions options;
        std::shared_ptr<EpilogueClass> epilogue_class;

        // TODO: move into descriptor
        int gran_k_a, gran_k_b, k_alignment;

        // The FP8-output batched GEMM may run the epilogue-v3 kernel (two epilogue warpgroups, packed SF words)
        bool epilogue_v3 = false;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        // NOTES: an FP4xFP4 pair uses the MXF4 MMA; other combinations use the MXF8F6F4 MMA.
        //        Both share the same kernel template and only differ in the operand dtype strings.
        const auto a_dtype = to_string(args.gemm_desc.a_dtype, not args.gemm_desc.is_mxf4_mma());
        const auto b_dtype = to_string(args.gemm_desc.b_dtype, not args.gemm_desc.is_mxf4_mma());
        const auto kernel = jit->compile(args.epilogue_v3 ? tag + "_v3" : tag, std::format(R"(
#include <deep_gemm/impls/{}>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {}
    >);
}};
)",
        args.epilogue_v3 ? "sm100_fp8_fp4_gemm_1d1d_v3.cuh" : "sm100_fp8_fp4_gemm_1d1d.cuh",
        args.epilogue_v3 ? "sm100_fp8_fp4_gemm_1d1d_v3_impl" : "sm100_fp8_fp4_gemm_1d1d_impl",
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        args.gran_k_a, args.gran_k_b, args.k_alignment,
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_desc.num_groups,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages, args.gemm_config.pipeline_config.num_tma_store_stages,
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        args.gemm_config.layout.swap_ab, args.gemm_desc.ensure_zero_padding,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        a_dtype, b_dtype, to_string(args.gemm_desc.cd_dtype),
        args.epilogue_class->get_epilogue_operator_type()), args.epilogue_class->compiler_options());

        // Launch
        jit->launch(
            kernel, args.options,
            args.grouped_layout, args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.epilogue_class->make_epilogue_operator_args(args.gemm_desc.m, args.gemm_desc.n),
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd
        );
    }
};

// Compute the SF block sizes for the SF TMA descriptors.
// Both MXF4 and MXF8F6F4 use UTCCP-aligned SF blocks, with a per-128 K sub-block count (`block_k / 128`).
static std::tuple<int, int, int> get_sf_block_config(const GemmConfig& config, const GemmDesc& desc) {
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(
        config.layout.block_m, config.layout.block_n, desc.get_mma_kind());
    return {sf_block_m, sf_block_n, config.layout.block_k / 128};
}

static void sm100_fp8_fp4_gemm_1d1d(const torch::stable::Tensor& a, const torch::stable::Tensor& sfa,
                                    const torch::stable::Tensor& b, const torch::stable::Tensor& sfb,
                                    const std::optional<torch::stable::Tensor>& c,
                                    const torch::stable::Tensor& d,
                                    const int& m, const int& n, const int& k,
                                    const int& gran_k_a, const int& gran_k_b,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const std::string& compiled_dims,
                                    const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode, 0, false, not desc.is_mxf4_mma());
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode, 0, false, not desc.is_mxf4_mma());
    const auto tensor_map_cd = make_tma_cd_desc(d, m, static_cast<int>(d.size(-1)),
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);
    const auto [sf_block_mn_a, sf_block_mn_b, sf_block_k] = get_sf_block_config(config, desc);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 sf_block_mn_a, gran_k_a, 1, 0, 0, false, sf_block_k, static_cast<int>(sfa.stride(-1)));
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 sf_block_mn_b, gran_k_b, 1, 0, 0, false, sf_block_k, static_cast<int>(sfb.stride(-1)));

    // Compile and launch
    SM100FP8FP4Gemm1D1DRuntime::compile_and_launch("sm100_fp8_fp4_gemm_1d1d", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .epilogue_class = epilogue_class,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        // NOTES: `k_alignment` is only used by k-grouped psum, dummy here
        .k_alignment = gran_k_a,
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    });
}

static void sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(const torch::stable::Tensor& a, const torch::stable::Tensor& sfa,
                                                         const torch::stable::Tensor& b, const torch::stable::Tensor& sfb,
                                                         const torch::stable::Tensor& d,
                                                         const torch::stable::Tensor& grouped_layout,
                                                         const int& num_groups, const int& m, const int& n, const int& k,
                                                         const int& gran_k_a, const int& gran_k_b,
                                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                         const std::string& compiled_dims,
                                                         const bool& use_psum_layout,
                                                         const bool& ensure_zero_padding,
                                                         const std::optional<int>& expected_m_for_psum_layout,
                                                         const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    // Only psum layout can use expected m
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    // NOTES: If actual M is dynamic, estimate config via `num_groups` and `expected_m`.
    //        Otherwise, treat the contiguous layout as a whole.
    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .ensure_zero_padding = ensure_zero_padding,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode, 0, false, not desc.is_mxf4_mma());
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode, 0, false, not desc.is_mxf4_mma());
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);
    const auto [sf_block_mn_a, sf_block_mn_b, sf_block_k] = get_sf_block_config(config, desc);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 sf_block_mn_a, gran_k_a, 1, 0, 0, false, sf_block_k, static_cast<int>(sfa.stride(-1)));
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 sf_block_mn_b, gran_k_b, num_groups, 0, 0, false, sf_block_k, static_cast<int>(sfb.stride(-1)));

    // Compile and launch
    SM100FP8FP4Gemm1D1DRuntime::compile_and_launch("sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .epilogue_class = epilogue_class,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        // NOTES: `k_alignment` is only used by k-grouped psum, dummy here
        .k_alignment = gran_k_a,
        .grouped_layout = grouped_layout.mutable_data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    });
}

static void sm100_m_grouped_fp8_fp4_gemm_masked_1d1d(const torch::stable::Tensor& a, const torch::stable::Tensor& sfa,
                                                     const torch::stable::Tensor& b, const torch::stable::Tensor& sfb,
                                                     const torch::stable::Tensor& d,
                                                     const torch::stable::Tensor& masked_m,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const int& expected_m,
                                                     const int& gran_k_a, const int& gran_k_b,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims,
                                                     const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode, 0, false, not desc.is_mxf4_mma());
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode, 0, false, not desc.is_mxf4_mma());
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);
    const auto [sf_block_mn_a, sf_block_mn_b, sf_block_k] = get_sf_block_config(config, desc);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 sf_block_mn_a, gran_k_a, num_groups, 0, 0, false, sf_block_k, static_cast<int>(sfa.stride(-1)));
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 sf_block_mn_b, gran_k_b, num_groups, 0, 0, false, sf_block_k, static_cast<int>(sfb.stride(-1)));

    // Compile and launch
    SM100FP8FP4Gemm1D1DRuntime::compile_and_launch("sm100_m_grouped_fp8_fp4_gemm_masked_1d1d", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .epilogue_class = epilogue_class,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        // NOTES: `k_alignment` is only used by k-grouped psum, dummy here
        .k_alignment = gran_k_a,
        .grouped_layout = masked_m.mutable_data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    });
}

static void sm100_k_grouped_fp8_gemm_1d1d(const torch::stable::Tensor& a, const torch::stable::Tensor& sfa,
                                          const torch::stable::Tensor& b, const torch::stable::Tensor& sfb,
                                          const std::optional<torch::stable::Tensor>& c,
                                          const torch::stable::Tensor& d,
                                          const int& m, const int& n,
                                          const torch::stable::Tensor& grouped_layout,
                                          const int& gran_k,
                                          const int& k_alignment,
                                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                          const std::string& compiled_dims,
                                          const bool& use_psum_layout,
                                          const std::shared_ptr<EpilogueClass>& epilogue_class) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);
    const auto num_groups = static_cast<int>(grouped_layout.numel());
    const auto sum_k = static_cast<int>(a.size(0));
    const auto expected_k = ceil_div(sum_k, num_groups);

    const auto desc = GemmDesc {
        .gemm_type = use_psum_layout ? GemmType::KGroupedContiguousWithPsumLayout : GemmType::KGroupedContiguous,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        // NOTES: expected_k is not used in SM100 get_best_config yet.
        .expected_m = m, .expected_n = n, .expected_k = expected_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);
    DG_HOST_ASSERT(k_alignment % config.layout.block_k == 0);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::MN, a, m, sum_k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(0)), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::MN, b, n, sum_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(0)), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_3d_desc(d, n, m, num_groups,
                                                config.storage_config.store_block_n,
                                                config.storage_config.store_block_m, 1,
                                                static_cast<int>(d.stride(1)),
                                                static_cast<int>(d.stride(0)),
                                                config.storage_config.swizzle_cd_mode);
    const auto [sf_block_mn_a, sf_block_mn_b, sf_block_k] = get_sf_block_config(config, desc);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, static_cast<int>(sfa.size(0)) * gran_k * 4,
                                                 sf_block_mn_a, gran_k, 1, 0, 0, false, sf_block_k, static_cast<int>(sfa.stride(0)));
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, static_cast<int>(sfb.size(0)) * gran_k * 4,
                                                 sf_block_mn_b, gran_k, 1, 0, 0, false, sf_block_k, static_cast<int>(sfb.stride(0)));

    // Compile and launch
    SM100FP8FP4Gemm1D1DRuntime::compile_and_launch("sm100_k_grouped_fp8_gemm_1d1d", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .epilogue_class = epilogue_class,
        .gran_k_a = gran_k,
        .gran_k_b = gran_k,
        .k_alignment = k_alignment,
        .grouped_layout = grouped_layout.mutable_data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    });
}

// Layout and epilogue of the FP8-output batched GEMM of the DeepSeek-V4.1 wo_a einsum ("bhr,hdr->bhd", recipe (1, 1, 32),
// G = 8, N = 1024, K = 4096) per M range, measured on VR200 (212 SMs) at the served decode shapes. `epilogue`: 0 = stock store,
// 4 = epilogue v3. `grid_sms` > 0 sizes the persistent grid so the tiles fill whole waves. Shapes outside every row keep
// DeepGEMM's own heuristic.
struct EinsumLayoutRule {
    int num_groups, n, k, num_sms, m_lo, m_hi;
    int swap_ab, block_m, block_n, cluster_m, cluster_n, epilogue;
    int grid_sms;
};
static constexpr EinsumLayoutRule kEinsumLayoutRules[] = {
    {8, 1024, 4096, 212, 1, 48, 1, 16, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 49, 96, 1, 32, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 97, 144, 1, 48, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 145, 192, 1, 64, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 193, 240, 1, 80, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 241, 288, 1, 96, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 289, 336, 1, 112, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 337, 384, 1, 128, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 385, 432, 1, 144, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 433, 480, 1, 160, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 481, 528, 1, 176, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 529, 576, 1, 192, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 577, 624, 1, 208, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 625, 672, 1, 224, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 673, 720, 1, 240, 128, 1, 2, 4, 0},
    {8, 1024, 4096, 212, 721, 744, 1, 128, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 793, 864, 1, 144, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 865, 960, 1, 160, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 961, 1056, 1, 176, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1057, 1152, 1, 192, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1153, 1248, 1, 208, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1249, 1344, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1345, 1440, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1441, 1464, 1, 176, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1561, 1584, 1, 176, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1585, 1728, 1, 192, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1729, 1872, 1, 208, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 1873, 2016, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 2017, 2160, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 2161, 2304, 1, 192, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 2305, 2496, 1, 208, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 2497, 2688, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 2689, 2880, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 2881, 3120, 1, 208, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 3121, 3360, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 3361, 3600, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 3601, 3744, 1, 208, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 3745, 4032, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 4033, 4320, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 4321, 4368, 1, 208, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 4369, 4704, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 4705, 5040, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 5041, 5376, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 5377, 5760, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 5761, 6048, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 6049, 6480, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 6481, 6720, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 6721, 7200, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 7201, 7392, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 7393, 7920, 1, 240, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 7921, 8064, 1, 224, 128, 1, 2, 4, 192},
    {8, 1024, 4096, 212, 8065, 1073741824, 1, 240, 128, 1, 2, 4, 192}
};

static void sm100_fp8_bmm(const torch::stable::Tensor& a, const torch::stable::Tensor& sfa,
                          const torch::stable::Tensor& b, const torch::stable::Tensor& sfb,
                          const std::optional<torch::stable::Tensor>& c,
                          const torch::stable::Tensor& d,
                          const int& batch_size, const int& m, const int& n, const int& k,
                          const int& gran_k_a, const int& gran_k_b,
                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                          const std::string& compiled_dims,
                          const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    auto config = get_best_config<SM100ArchSpec>(desc);
    bool epilogue_v3 = false;
    // Switch DG_MOSCA_EINSUM_RULE (read per call, before any CUDA-graph capture). Unset or 0 = DeepGEMM's own heuristic
    // for every shape (stock, the default); 1 = the kEinsumLayoutRules rows (bitwise to stock, measured on VR200).
    if (deep_jit::get_env<int>("DG_MOSCA_EINSUM_RULE") > 0 and
        d.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn and gran_k_a == 32 and gran_k_b == 32) {
        for (const auto& r: kEinsumLayoutRules) {
            if (batch_size == r.num_groups and n == r.n and k == r.k and desc.num_sms == r.num_sms and r.m_lo <= m and m <= r.m_hi) {
                auto rule_desc = desc;
                if (r.grid_sms > 0 and r.grid_sms <= desc.num_sms)
                    rule_desc.num_sms = r.grid_sms;
                config = get_config_for_layout<SM100ArchSpec>(
                    rule_desc, Layout{r.swap_ab, r.block_m, r.block_n, config.layout.block_k, r.cluster_m, r.cluster_n});
                if (r.epilogue == 4 and config.layout.swap_ab and config.layout.block_m <= 240) {
                    // Two epilogue warpgroups with two TMA-store buffers each, plus one packed-SF word per token row
                    epilogue_v3 = true;
                    config.launch_config.num_epilogue_threads = 256;
                    config.launch_config.num_threads = config.launch_config.num_non_epilogue_threads + 256;
                    config.pipeline_config.num_tma_store_stages = 4;
                    config.pipeline_config.smem_size += 2 * config.storage_config.store_block_m * config.storage_config.store_block_n +
                                                        config.layout.block_m * 4;
                    if (config.pipeline_config.smem_size > SM100ArchSpec::smem_capacity) {
                        const int per_stage = config.storage_config.load_block_m * config.layout.block_k +
                                              config.storage_config.load_block_n * config.layout.block_k +
                                              align(config.layout.block_m, 128) * config.layout.block_k / 32 +
                                              align(config.layout.block_n, 128) * config.layout.block_k / 32;
                        config.pipeline_config.num_stages -= 1;
                        config.pipeline_config.smem_size -= per_stage;
                    }
                    DG_HOST_ASSERT(config.pipeline_config.smem_size <= SM100ArchSpec::smem_capacity);
                }
                break;
            }
        }
    }

    const int load_block_m = config.storage_config.load_block_m;
    const auto [inner_dim_a, outer_dim_a] = get_inner_outer_dims(major_a, k, m);
    const auto [inner_block_a, outer_block_a] = get_inner_outer_dims(major_a, config.layout.block_k, load_block_m);
    const auto tensor_map_a = make_tma_3d_desc(a, inner_dim_a, outer_dim_a, batch_size,
                                               inner_block_a, outer_block_a, 1,
                                               a.stride(major_a == cute::UMMA::Major::K ? 1 : 2),
                                               a.stride(0),
                                               config.storage_config.swizzle_a_mode);

    const int load_block_n = config.storage_config.load_block_n;
    const auto [inner_dim_b, outer_dim_b] = get_inner_outer_dims(major_b, k, n);
    const auto [inner_block_b, outer_block_b] = get_inner_outer_dims(major_b, config.layout.block_k, load_block_n);
    const auto tensor_map_b = make_tma_3d_desc(b, inner_dim_b, outer_dim_b, batch_size,
                                               inner_block_b, outer_block_b, 1,
                                               b.stride(major_b == cute::UMMA::Major::K ? 1 : 2),
                                               b.stride(0),
                                               config.storage_config.swizzle_b_mode);

    const int store_block_m = config.storage_config.store_block_m;
    const int store_block_n = config.storage_config.store_block_n;
    const auto tensor_map_cd = make_tma_3d_desc(d, n, m, batch_size,
                                                store_block_n, store_block_m, 1,
                                                d.stride(1), d.stride(0),
                                                config.storage_config.swizzle_cd_mode);

    const auto [sf_block_mn_a, sf_block_mn_b, sf_block_k] = get_sf_block_config(config, desc);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 sf_block_mn_a, gran_k_a, batch_size, 0, 0, false, sf_block_k, static_cast<int>(sfa.stride(-1)));
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 sf_block_mn_b, gran_k_b, batch_size, 0, 0, false, sf_block_k, static_cast<int>(sfb.stride(-1)));

    // Compile and launch
    SM100FP8FP4Gemm1D1DRuntime::compile_and_launch("sm100_fp8_gemm_1d1d", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .epilogue_class = epilogue_class,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        // NOTES: `k_alignment` is only used by k-grouped psum, dummy here
        .k_alignment = gran_k_a,
        .epilogue_v3 = epilogue_v3,
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    });
}

static void sm100_k_grouped_fp4_gemm_1d1d(
    const torch::stable::Tensor& a, const torch::stable::Tensor& sfa,
    const torch::stable::Tensor& b, const torch::stable::Tensor& sfb,
    const std::optional<torch::stable::Tensor>& c,
    const torch::stable::Tensor& d,
    const int& m, const int& n, const int& sum_k,
    const torch::stable::Tensor& grouped_layout,
    const int& k_alignment,
    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
    const std::string& compiled_dims,
    const bool& use_psum_layout,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    constexpr int gran_k = 32;
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    const auto num_groups = static_cast<int>(grouped_layout.numel());
    const auto expected_k = ceil_div(sum_k, num_groups);

    const auto desc = GemmDesc {
        .gemm_type = use_psum_layout ? GemmType::KGroupedContiguousWithPsumLayout : GemmType::KGroupedContiguous,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = expected_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);
    DG_HOST_ASSERT(k_alignment % config.layout.block_k == 0);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, sum_k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(0)), 1,
                                              config.storage_config.swizzle_a_mode, 0, false, false);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, sum_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(0)), 1,
                                              config.storage_config.swizzle_b_mode, 0, false, false);
    const auto tensor_map_cd = make_tma_3d_desc(d, n, m, num_groups,
                                                config.storage_config.store_block_n,
                                                config.storage_config.store_block_m, 1,
                                                static_cast<int>(d.stride(1)),
                                                static_cast<int>(d.stride(0)),
                                                config.storage_config.swizzle_cd_mode);
    const auto [sf_block_mn_a, sf_block_mn_b, sf_block_k] = get_sf_block_config(config, desc);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, static_cast<int>(sfa.size(0)) * gran_k * 4,
                                                 sf_block_mn_a, gran_k, 1, 0, 0, false, sf_block_k, static_cast<int>(sfa.stride(0)));
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, static_cast<int>(sfb.size(0)) * gran_k * 4,
                                                 sf_block_mn_b, gran_k, 1, 0, 0, false, sf_block_k, static_cast<int>(sfb.stride(0)));

    // Compile and launch
    SM100FP8FP4Gemm1D1DRuntime::compile_and_launch("sm100_k_grouped_fp4_gemm_1d1d", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
        },
        .epilogue_class = epilogue_class,
        .gran_k_a = gran_k,
        .gran_k_b = gran_k,
        .k_alignment = k_alignment,
        .grouped_layout = grouped_layout.mutable_data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    });
}

} // namespace deep_gemm
