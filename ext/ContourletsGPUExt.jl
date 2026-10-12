module ContourletsGPUExt

using Contourlets: Contourlets,
    ContourletParams, QuincunxFilterPair,
    ContourletCoefficients, NSCTCoefficients, TICTCoefficients,
    _scratch_like,
    lp_decompose, lp_reconstruct, nsp_decompose,
    dfb_decompose, dfb_reconstruct, nsdfb_decompose,
    qx_downsample!, qx_upsample!,
    ct_forward!, ct_inverse!, nsct_forward!
using GPUArrays: GPUArrays, AbstractGPUMatrix, AbstractGPUArray

const GPUSubArray{T, N} = SubArray{T, N, <:AbstractGPUArray}
const _AbstractGPUMatrix{T} = Union{AbstractGPUMatrix{T}, GPUSubArray{T, 2}}
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, Adapt

# Only the device primitives and top-level transforms are specialised here.  The
# Laplacian/nonsubsampled pyramid functions are reused unchanged from the main
# package: their allocating methods are written with broadcasts and dispatch to
# these GPU primitives (conv2d_sep!, rect_*sample!, …) when given device arrays.
# Every method that extends a Contourlets function is a qualified definition
# (`function Contourlets.conv2d_sep!(...)`).

include("ContourletsGPUExt/utils_gpu.jl")
include("ContourletsGPUExt/primitives_gpu.jl")
include("ContourletsGPUExt/qfb_gpu.jl")
include("ContourletsGPUExt/bank_gpu.jl")
include("ContourletsGPUExt/transforms_gpu.jl")

end # module ContourletsGPUExt
