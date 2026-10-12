# ContourletParams: the decomposition descriptor shared by every transform
# (pyramid depth, per-level DFB depth, filters), its precision conversion, and
# the `parabolic_levels` helper that builds a parabolic-scaling `L_array`.

"""
    ContourletParams

Describes a Contourlet or NSCT decomposition: number of pyramid levels `J`, the
per-level DFB depth `L_array` (length `J`), and optional filter overrides.

```jldoctest
julia> using Contourlets

julia> ContourletParams(J = 3, L_array = [1, 2, 3])
ContourletParams(J=3, L_array=[1, 2, 3], …)
```
"""
struct ContourletParams{T <: AbstractFloat}
    J::Int
    L_array::Vector{Int}
    lp_filters::FilterPair{T}
    dfb_filters::QuincunxFilterPair{T}
end

function ContourletParams(;
        J::Int,
        L_array::AbstractVector{<:Integer},
        lp_filters::FilterPair = CDF97,
        dfb_filters::QuincunxFilterPair = Q2345
    )
    J >= 1               || throw(ArgumentError("J must be ≥ 1"))
    length(L_array) == J || throw(ArgumentError("length(L_array) must equal J"))
    all(>=(0), L_array)  || throw(ArgumentError("L_array entries must be ≥ 0"))
    T = promote_type(eltype(lp_filters), eltype(dfb_filters))
    return ContourletParams{T}(
        J, collect(Int, L_array),
        FilterPair{T}(T.(lp_filters.h), T.(lp_filters.g)),
        _convert_qfp(T, dfb_filters)
    )
end

Base.eltype(::ContourletParams{T}) where {T} = T

function Base.show(io::IO, p::ContourletParams)
    return print(io, "ContourletParams(J=$(p.J), L_array=$(p.L_array), …)")
end

# Convert a ContourletParams to real filter precision `Tf` (no-op when already Tf).
# `Tf <: AbstractFloat` on the identity method too, or it loses specificity to the
# converting method below (see `_convert_qfp`).
_convert_params(::Type{Tf}, p::ContourletParams{Tf}) where {Tf <: AbstractFloat} = p
_convert_params(::Type{Tf}, p::ContourletParams) where {Tf <: AbstractFloat} =
    ContourletParams{Tf}(
    p.J, p.L_array,
    FilterPair{Tf}(Tf.(p.lp_filters.h), Tf.(p.lp_filters.g)),
    _convert_qfp(Tf, p.dfb_filters)
)

"""
    parabolic_levels(J, l_j0=1) -> Vector{Int}

Compute the per-scale DFB depth array following the parabolic scaling law

    l_j = l_j0 + ⌊(j0 - j) / 2⌋,    j0 = J,

so that the number of directions doubles every other scale — the discrete
counterpart of parabolic scaling, width ∝ length², in Do & Vetterli (2005).  Index `j` matches
the scale ordering of [`ct_forward`](@ref) and [`nsct_forward`](@ref): `j = 1`
is the *finest* scale (first pyramid split) and `j = J` the coarsest, hence
finer scales receive more directional subbands.  `l_j0` is the DFB depth at the
coarsest scale.

# Examples
```jldoctest
julia> using Contourlets

julia> parabolic_levels(4)
4-element Vector{Int64}:
 2
 2
 1
 1

julia> parabolic_levels(5, 2)
5-element Vector{Int64}:
 4
 3
 3
 2
 2
```
"""
function parabolic_levels(J::Int, l_j0::Int = 1)::Vector{Int}
    j0 = J
    return [max(0, l_j0 + (j0 - j) ÷ 2) for j in 1:J]
end
