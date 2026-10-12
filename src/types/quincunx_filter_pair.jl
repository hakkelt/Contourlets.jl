# QuincunxFilterPair: the 2-D quincunx filter pair of the DFB stage, with its
# ladder-mode predicate and element-type conversion.
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
