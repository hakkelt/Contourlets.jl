# Types specific to the 2-D contourlet transforms.
# QuincunxFilterPair: 2-D quincunx filter pair (used for the DFB stage).
# ContourletParams / ContourletCoefficients / NSCTCoefficients: top-level descriptors.
#
# `FilterPair` and the element-type convention (`_data_eltype` / `_filter_eltype`)
# come from MultiscaleFilterBanks.jl, the shared core: filters are always real
# (`Tf <: AbstractFloat`) while image data may be real or complex (`Td <: Number`),
# and the transforms thread the two independently so complex images are filtered
# by real kernels.

"""
    QuincunxFilterPair{T}

A two-channel filter pair for the DFB stage.  `h_q` is the analysis low-pass kernel
and `g_q` is the synthesis low-pass kernel, both stored as dense `Matrix{T}`.  The
origin (zero-lag) tap is at `(c_h[1], c_h[2])` and `(c_g[1], c_g[2])` respectively
(1-based).

Two operating modes exist:

- **Modulation mode** (`f_ladder` empty): the high-pass filters are derived by
  modulation, `H₁(z) = H₀(-z)` and `G₁(z) = G₀(-z)`.  Perfect reconstruction is
  the user's responsibility (it holds e.g. for the Haar pair).
- **Ladder (lifting) mode** (`f_ladder` non-empty): the filter bank is the
  quincunx ladder bank of Phoong et al. (1995) built from the symmetric,
  even-length 1-D filter `f_ladder` (their β).  Perfect reconstruction is
  structural — exact for any `f_ladder`.  The DFB and NSDFB use this mode; in it
  `h_q`/`g_q` hold the equivalent low-pass filters of the 1-D ladder, used by the
  shear-based [`qfb_decompose`](@ref) (see [`Q2345`](@ref)).

# Examples
```jldoctest
julia> using Contourlets

julia> qfp = QuincunxFilterPair([0.5 0.5], [1.0 1.0], (1, 1), (1, 2));

julia> eltype(qfp)
Float64
```
"""
struct QuincunxFilterPair{T <: AbstractFloat}
    h_q::Matrix{T}       # analysis low-pass kernel (equivalent filter in ladder mode)
    g_q::Matrix{T}       # synthesis low-pass kernel (equivalent filter in ladder mode)
    c_h::Tuple{Int, Int}  # origin index in h_q (1-based row, col)
    c_g::Tuple{Int, Int}  # origin index in g_q (1-based row, col)
    f_ladder::Vector{T}  # 1-D lifting filter; empty → modulation mode
end

function QuincunxFilterPair(
        h_q::AbstractMatrix, g_q::AbstractMatrix,
        c_h::Tuple{Int, Int}, c_g::Tuple{Int, Int},
        f_ladder::AbstractVector = Float64[]
    )
    T = promote_type(eltype(h_q), eltype(g_q))
    return QuincunxFilterPair{T}(T.(h_q), T.(g_q), c_h, c_g, T.(f_ladder))
end

# Backward-compatible 4-argument inner-style constructor (modulation mode).
QuincunxFilterPair{T}(
    h_q::AbstractMatrix, g_q::AbstractMatrix,
    c_h::Tuple{Int, Int}, c_g::Tuple{Int, Int}
) where {T <: AbstractFloat} =
    QuincunxFilterPair{T}(T.(h_q), T.(g_q), c_h, c_g, T[])

Base.eltype(::QuincunxFilterPair{T}) where {T} = T

"""
    is_ladder(qfp::QuincunxFilterPair) -> Bool

Return `true` if `qfp` operates in ladder (lifting) mode.
"""
is_ladder(qfp::QuincunxFilterPair) = !isempty(qfp.f_ladder)

# Convert a QuincunxFilterPair to element type T (no-op when already T).
# Both methods bound `T <: AbstractFloat`: with an unbounded `T` the diagonal
# identity method is *less* specific than the converting one, so it never ran and
# every call built a fresh copy.
_convert_qfp(::Type{T}, qfp::QuincunxFilterPair{T}) where {T <: AbstractFloat} = qfp
_convert_qfp(::Type{T}, qfp::QuincunxFilterPair) where {T <: AbstractFloat} =
    QuincunxFilterPair{T}(T.(qfp.h_q), T.(qfp.g_q), qfp.c_h, qfp.c_g, T.(qfp.f_ladder))

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
    ContourletCoefficients{Td, A}

Output of `ct_forward`.  `Td` is the data element type (real or complex) and
`A <: AbstractMatrix{Td}` the storage array type — so the coefficients live wherever
they were computed (a host `Matrix` for CPU input, a device array such as `CuMatrix`
for a GPU input — no implicit transfer).  Contains:
- `coarse`: the low-pass residual after `J` LP stages.
- `subbands`: `Vector` of length `J`; `subbands[j]` is a `Vector` of `2^L_array[j]`
  directional subband matrices.
"""
struct ContourletCoefficients{Td <: Number, A <: AbstractMatrix{Td}}
    coarse::A                      # storage array type A (host or device)
    subbands::Vector{Vector{A}}    # [scale][direction]
end

# Infer (Td, A) from the arguments.
ContourletCoefficients(coarse::A, subbands::Vector{Vector{A}}) where {A <: AbstractMatrix} =
    ContourletCoefficients{eltype(A), A}(coarse, subbands)

"""
    NSCTCoefficients{Td, A}

Output of `nsct_forward`.  Same structure as `ContourletCoefficients` (data type `Td`,
storage type `A`) but each subband matrix has the same spatial size as the input image
(no downsampling).
"""
struct NSCTCoefficients{Td <: Number, A <: AbstractMatrix{Td}}
    coarse::A
    subbands::Vector{Vector{A}}
end

NSCTCoefficients(coarse::A, subbands::Vector{Vector{A}}) where {A <: AbstractMatrix} =
    NSCTCoefficients{eltype(A), A}(coarse, subbands)

"""
    TICTCoefficients{Td, A}

Output of [`tict_forward`](@ref) and [`stict_forward`](@ref).  `coarse` is the
lowpass image and `subbands[i][c][d]` the `d`-th directional subband of the `c`-th
of the four highpass channels at pyramid level `i` (`i = 1` finest) — Eslami &
Radha's `ρ_{i,k}^{(d)}` / `η_{i,k}^{(d)}`.  Channels 1 and 4 carry a full
directional filter bank (`2^L_array[i]` subbands), channels 2 and 3 a half tree
(`2^(L_array[i]-1) + 1` subbands; see [`tict_forward`](@ref)).  TICT subbands all
have the input's size; STICT subbands are critically sampled from full-size
channels.
"""
struct TICTCoefficients{Td <: Number, A <: AbstractMatrix{Td}}
    coarse::A
    subbands::Vector{Vector{Vector{A}}}    # [level][channel 1..4][direction]
end

TICTCoefficients(coarse::A, subbands::Vector{Vector{Vector{A}}}) where {A <: AbstractMatrix} =
    TICTCoefficients{eltype(A), A}(coarse, subbands)

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
