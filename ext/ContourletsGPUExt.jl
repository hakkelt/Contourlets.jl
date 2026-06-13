module ContourletsGPUExt

using Contourlets
using GPUArrays
using GPUArrays: AbstractGPUMatrix, AbstractGPUArray
using KernelAbstractions
using KernelAbstractions: Adapt

import Contourlets:
    conv2d_sep!, lp_decompose, lp_decompose!, lp_reconstruct, lp_reconstruct!,
    nsp_decompose, nsp_decompose!, nsp_reconstruct, nsp_reconstruct!,
    shear!, inv_shear!, rect_downsample!, rect_upsample!,
    qx_downsample, qx_downsample!, qx_upsample, qx_upsample!,
    qfb_decompose, qfb_decompose!, qfb_reconstruct, qfb_reconstruct!,
    dfb_decompose, dfb_reconstruct,
    nsdfb_decompose, nsdfb_reconstruct,
    ct_forward, ct_inverse, nsct_forward, nsct_inverse,
    FilterPair, QuincunxFilterPair, ContourletParams,
    ContourletCoefficients, NSCTCoefficients,
    upsample_filter

include("ContourletsGPUExt/utils_gpu.jl")
include("ContourletsGPUExt/primitives_gpu.jl")
include("ContourletsGPUExt/pyramid_gpu.jl")
include("ContourletsGPUExt/transforms_gpu.jl")

end # module ContourletsGPUExt
