# QuincunxFilterPair: the quincunx ladder filter pair of the DFB stage, with its
# element-type conversion.
#
# `FilterPair` and the element-type convention (`_data_eltype` / `_filter_eltype`)
# come from MultiscaleFilterBanks.jl, the shared core: filters are always real
# (`Tf <: AbstractFloat`) while image data may be real or complex (`Td <: Number`),
# and the transforms thread the two independently so complex images are filtered
# by real kernels.

"""
    QuincunxFilterPair{T}
    QuincunxFilterPair(f_ladder)
    QuincunxFilterPair{T}(f_ladder)

A two-channel quincunx filter pair for the DFB stage: the quincunx ladder bank of
Phoong et al. (1995), built from the symmetric, even-length 1-D ladder filter
`f_ladder` (their β).  Perfect reconstruction is structural — exact for any
`f_ladder`.  The DFB, NSDFB and the shear-based [`qfb_decompose`](@ref) all derive
their filters from `f_ladder` (see [`Q2345`](@ref)).

# Examples
```jldoctest
julia> using Contourlets

julia> qfp = QuincunxFilterPair(Float32.(Q2345.f_ladder));

julia> eltype(qfp)
Float32
```
"""
struct QuincunxFilterPair{T <: AbstractFloat}
    f_ladder::Vector{T}  # 1-D ladder (lifting) filter β, unmodulated
end

QuincunxFilterPair(f_ladder::AbstractVector{T}) where {T <: AbstractFloat} =
    QuincunxFilterPair{T}(collect(T, f_ladder))

Base.eltype(::QuincunxFilterPair{T}) where {T} = T

# Convert a QuincunxFilterPair to element type T (no-op when already T).
# Both methods bound `T <: AbstractFloat`: with an unbounded `T` the diagonal
# identity method is *less* specific than the converting one, so it never ran and
# every call built a fresh copy.
_convert_qfp(::Type{T}, qfp::QuincunxFilterPair{T}) where {T <: AbstractFloat} = qfp
_convert_qfp(::Type{T}, qfp::QuincunxFilterPair) where {T <: AbstractFloat} =
    QuincunxFilterPair{T}(T.(qfp.f_ladder))
