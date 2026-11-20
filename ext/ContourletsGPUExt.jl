module ContourletsGPUExt

using Contourlets: Contourlets, QuincunxFilterPair, _scratch_like, qx_downsample!, qx_upsample!
using GPUArrays: GPUArrays, AbstractGPUMatrix, AbstractGPUArray

const GPUSubArray{T, N} = SubArray{T, N, <:AbstractGPUArray}
const _AbstractGPUMatrix{T} = Union{AbstractGPUMatrix{T}, GPUSubArray{T, 2}}
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const

# Only the device primitives are specialised here.  The transforms and the
# Laplacian/nonsubsampled pyramid functions are reused unchanged from the main
# package: their allocating methods are written with broadcasts and dispatch to
# these GPU primitives (conv2d_sep!, rect_*sample!, …) when given device arrays,
# and their coefficients (stacked arrays, `SubbandVector`s) are allocated like
# the input, so they stay on the device.
# Every method that extends a Contourlets function is a qualified definition
# (`function Contourlets.conv2d_sep!(...)`).

include("ContourletsGPUExt/utils_gpu.jl")
include("ContourletsGPUExt/primitives_gpu.jl")
include("ContourletsGPUExt/qfb_gpu.jl")

end # module ContourletsGPUExt
