module ContourletsGPUExt

using Contourlets
using GPUArrays
using GPUArrays: AbstractGPUMatrix, AbstractGPUArray
using KernelAbstractions
using KernelAbstractions: Adapt

# Only the device primitives and top-level transforms are specialised here.  The
# Laplacian/nonsubsampled pyramid functions are reused unchanged from the main
# package: their allocating methods are written with broadcasts and dispatch to
# these GPU primitives (conv2d_sep!, rect_*sample!, …) when given device arrays.
import Contourlets:
    conv2d_sep!,
    shear!, inv_shear!, rect_downsample!, rect_upsample!,
    qx_downsample, qx_downsample!, qx_upsample, qx_upsample!,
    qfb_decompose, qfb_reconstruct,
    nsdfb_decompose, nsdfb_reconstruct,
    ct_forward, ct_inverse, nsct_forward, nsct_inverse

include("ContourletsGPUExt/utils_gpu.jl")
include("ContourletsGPUExt/primitives_gpu.jl")
include("ContourletsGPUExt/dfb_gpu.jl")
include("ContourletsGPUExt/nsdfb_gpu.jl")
include("ContourletsGPUExt/transforms_gpu.jl")

end # module ContourletsGPUExt
