# Types shared across the whole package.
# FilterPair: 1-D biorthogonal filter pair (used for the Laplacian Pyramid stage).
# QuincunxFilterPair: 2-D quincunx filter pair (used for the DFB stage).
# ContourletParams / ContourletCoefficients / NSCTCoefficients: top-level descriptors.

"""
    FilterPair{T}

A 1-D biorthogonal filter pair holding an analysis (`h`) and synthesis (`g`) low-pass
filter vector.  Both filters are stored as coefficient vectors whose centre tap is at
index `ceil(Int, length/2)` (1-based Julia indexing).

# Examples
```jldoctest
julia> using Contourlets
julia> fp = FilterPair([0.5, 1.0, 0.5], [0.25, 0.5, 0.25])
FilterPair{Float64}
```
"""
struct FilterPair{T <: AbstractFloat}
    h::Vector{T}   # analysis (low-pass)
    g::Vector{T}   # synthesis (low-pass)
end

FilterPair(h::AbstractVector, g::AbstractVector) =
    FilterPair(promote_type(eltype(h), eltype(g)).(h), promote_type(eltype(h), eltype(g)).(g))

Base.eltype(::FilterPair{T}) where {T} = T

"""
    QuincunxFilterPair{T}

A 2-D quincunx filter pair.  `h_q` is the analysis low-pass kernel and `g_q` is the
synthesis low-pass kernel, both stored as dense `Matrix{T}`.  The origin (zero-lag) tap
is at `(c_h[1], c_h[2])` and `(c_g[1], c_g[2])` respectively (1-based).

For the standard Q2345 pair the high-pass filters are derived by modulation:
`h_high[n1,n2] = (-1)^(n1+n2) * g_q[n1,n2]` (analysis high-pass) and
`g_high[n1,n2] = (-1)^(n1+n2) * h_q[n1,n2]` (synthesis high-pass).

# Examples
```jldoctest
julia> using Contourlets
julia> qfp = QuincunxFilterPair([0.5 0.0; 0.5 0.0], [1.0 0.0; 1.0 0.0], (1,1), (2,1))
QuincunxFilterPair{Float64}
```
"""
struct QuincunxFilterPair{T <: AbstractFloat}
    h_q::Matrix{T}       # analysis low-pass kernel
    g_q::Matrix{T}       # synthesis low-pass kernel
    c_h::Tuple{Int, Int}  # origin index in h_q (1-based row, col)
    c_g::Tuple{Int, Int}  # origin index in g_q (1-based row, col)
end

function QuincunxFilterPair(
        h_q::AbstractMatrix, g_q::AbstractMatrix,
        c_h::Tuple{Int, Int}, c_g::Tuple{Int, Int}
    )
    T = promote_type(eltype(h_q), eltype(g_q))
    return QuincunxFilterPair{T}(T.(h_q), T.(g_q), c_h, c_g)
end

Base.eltype(::QuincunxFilterPair{T}) where {T} = T

"""
    ContourletParams

Describes a Contourlet or NSCT decomposition: number of pyramid levels `J`, the
per-level DFB depth `L_array` (length `J`), and optional filter overrides.

```jldoctest
julia> using Contourlets
julia> p = ContourletParams(J=3, L_array=[1,2,3])
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
    length(L_array) == J || throw(ArgumentError("length(L_array) must equal J"))
    all(>=(0), L_array)  || throw(ArgumentError("L_array entries must be ≥ 0"))
    T = promote_type(eltype(lp_filters), eltype(dfb_filters))
    return ContourletParams{T}(
        J, collect(Int, L_array),
        FilterPair{T}(T.(lp_filters.h), T.(lp_filters.g)),
        QuincunxFilterPair{T}(
            T.(dfb_filters.h_q),
            T.(dfb_filters.g_q),
            dfb_filters.c_h,
            dfb_filters.c_g
        )
    )
end

function Base.show(io::IO, p::ContourletParams)
    return print(io, "ContourletParams(J=$(p.J), L_array=$(p.L_array), …)")
end

"""
    ContourletCoefficients{T}

Output of `ct_forward`.  Contains:
- `coarse`: the low-pass residual `Matrix{T}` after `J` LP stages.
- `subbands`: `Vector` of length `J`; `subbands[j]` is a `Vector` of `2^L_array[j]`
  directional subband matrices.
- `params`: the `ContourletParams` used to produce these coefficients.
"""
struct ContourletCoefficients{T <: AbstractFloat}
    coarse::Matrix{T}
    subbands::Vector{Vector{Matrix{T}}}   # [scale][direction]
    params::ContourletParams{T}
end

"""
    NSCTCoefficients{T}

Output of `nsct_forward`.  Same structure as `ContourletCoefficients` but each subband
matrix has the same spatial size as the input image (no downsampling).
"""
struct NSCTCoefficients{T <: AbstractFloat}
    coarse::Matrix{T}
    subbands::Vector{Vector{Matrix{T}}}
    params::ContourletParams{T}
end

"""
    parabolic_levels(J, l_j0=1) -> Vector{Int}

Compute the per-scale DFB depth array following the parabolic scaling law:

    l_j = l_j0 + floor((j0 - j) / 2)

where `j` runs from 1 (coarsest) to `J` (finest) and `j0 = J`.  The returned
vector has `J` elements with `L_array[j]` corresponding to scale `j`.

# Examples
```jldoctest
julia> using Contourlets
julia> parabolic_levels(4)
4-element Vector{Int64}:
 1
 1
 2
 3
```
"""
function parabolic_levels(J::Int, l_j0::Int = 1)::Vector{Int}
    j0 = J
    return [max(0, l_j0 + (j0 - j) ÷ 2) for j in 1:J]
end
