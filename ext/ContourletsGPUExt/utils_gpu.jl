# GPU utilities — backend detection and filter transfer helpers.

"""
    _gpu_backend(A::AbstractGPUArray) -> Backend

Extract the KernelAbstractions backend from a GPU array.
"""
_gpu_backend(A::AbstractGPUArray) = KernelAbstractions.get_backend(A)

"""
    _to_device(backend, x::AbstractArray) -> GPU array

Copy `x` to `backend` device if it is not already there.
"""
function _to_device(backend, x::AbstractArray{T}) where {T}
    out = KernelAbstractions.allocate(backend, T, size(x))
    copyto!(out, x)
    return out
end

# If already on device, return as-is.
_to_device(::Any, x::AbstractGPUArray) = x

"""
    _ensure_gpu(backend, x) -> GPU vector/matrix

Move a CPU Vector/Matrix to the GPU device; no-op if already there.
"""
_ensure_gpu(backend, x::AbstractVector) = _to_device(backend, x)
_ensure_gpu(backend, x::AbstractMatrix) = _to_device(backend, x)
_ensure_gpu(::Any, x::AbstractGPUArray) = x
