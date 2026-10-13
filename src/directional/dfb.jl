# l-level Directional Filter Bank (DFB).
#
# The DFB of Bamberger & Smith in the formulation of Do (2001, thesis §3.3) and
# Do & Vetterli (2005), with the quincunx fan filters of Phoong et al. (1995)
# (the default ladder pair is Q2345).  It is computed as the multichannel bank of
# the thesis — each channel filters with its equivalent filter and downsamples by
# a diagonal matrix — in the frequency domain (see `bank_design.jl`).  The 2ˡ
# subbands tile the frequency plane into directional wedges, perfect
# reconstruction is exact, and each subband is equivariant to circular shifts on
# its own sampling lattice.
#
# Subband sizes after l levels (input n₁ × n₂):
#   l=1: 2 subbands  n₁/2 × n₂      (quincunx lattice, stored column by column)
#   l=2: 4 subbands  n₁/2 × n₂/2
#   l≥3: 2^(l-1) subbands n₁/2^(l-1) × n₂/2, then 2^(l-1) subbands n₁/2 × n₂/2^(l-1)
#
# References: R. H. Bamberger & M. J. T. Smith, "A filter bank for the directional
# decomposition of images: theory and design", IEEE TSP 40(4), 1992; M. N. Do,
# "Directional multiresolution image representations", PhD thesis, EPFL, 2001,
# §3.3; M. N. Do & M. Vetterli, "The contourlet transform: an efficient
# directional multiresolution image representation", IEEE TIP 14(12), 2005.

# Bank for `bandpass`, or the matching one carried by a workspace.
function _dfb_runner(bandpass::AbstractMatrix, l::Int, qfp::QuincunxFilterPair, half::Symbol, ::Type{Tf}) where {Tf}
    bank = _directional_bank(Tf, size(bandpass), l, qfp.f_ladder; decimated = true, half = half)
    return _bank_runner(_adapt_bank(bandpass, bank), bandpass)
end

# Move a host bank to the array family of `ref` (identity on the host).
function _adapt_bank(ref::AbstractArray, bank::_DirectionalBank)
    A = typeof(similar(ref, eltype(eltype(bank.ana)), 0, 0))
    V = typeof(similar(ref, eltype(eltype(bank.phase_r)), 0))
    return _DirectionalBank(
        A[_like(ref, a) for a in bank.ana], A[_like(ref, a) for a in bank.syn],
        V[_like(ref, v) for v in bank.phase_r], V[_like(ref, v) for v in bank.phase_c],
        bank.delay, bank.sampling, bank.dims, bank.levels
    )
end
_adapt_bank(::Array, bank::_DirectionalBank) = bank
_adapt_bank(ref::SubArray, bank::_DirectionalBank) = _adapt_bank(parent(ref), bank)

# Call `f(runner)` with the workspace's runner when it has one for `x`, otherwise
# with a freshly built one.  Each branch calls `f` on its own concrete runner type,
# so the workspace path stays type-stable and allocation-free.
function _with_runner(f, workspace, x::AbstractMatrix, l::Int, qfp::QuincunxFilterPair, half::Symbol, ::Type{Tf}) where {Tf}
    r = _workspace_runner(workspace, size(x), l, half, qfp, Tf)
    r === nothing || return f(r)
    return f(_dfb_runner(x, l, qfp, half, Tf))
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    dfb_decompose(bandpass, l_levels, qfp::QuincunxFilterPair; half=:both, workspace=nothing)
        -> Vector{Matrix}

Decompose `bandpass` into `2^l_levels` directional subbands using an
`l_levels`-deep DFB.

This is the directional filter bank of Do & Vetterli (2005) built from the ladder
pair `qfp` (default [`Q2345`](@ref)): the subbands tile the frequency plane into
`2^l_levels` directional wedges.  Both image dimensions must be divisible by
`2^max(l_levels - 1, 1)`.

`half = :first` or `:second` gives the half trees of Eslami & Radha (2006): only one
of the two level-1 fans is split further, so there are `2^(l_levels-1) + 1`
subbands (the split fan's wedges, which equal the corresponding subbands of the full
tree, and the other fan, unsplit, on the quincunx lattice).  `:first` splits the
fan whose wedges come first in the full tree.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(3), 32, 32);

julia> sbs = dfb_decompose(x, 2, Q2345);

julia> length(sbs), size(sbs[1])
(4, (16, 16))
```
"""
function dfb_decompose(
        bandpass::AbstractMatrix, l_levels::Int,
        qfp::QuincunxFilterPair;
        half::Symbol = :both,
        workspace = nothing
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    _check_half(half)
    l_levels == 0 && return [copy(bandpass)]
    Td = _data_eltype(bandpass)      # data type (real or complex)
    Tf = _filter_eltype(Td)          # real filter precision
    img = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    return _with_runner(workspace, img, l_levels, qfp, half, Tf) do r
        _bank_analyze!([similar(img, s) for s in _subband_sizes(r.bank)], img, r)
    end
end

"""
    dfb_decompose!(subbands::Vector{<:AbstractMatrix}, bandpass, l_levels, qfp::QuincunxFilterPair;
                   half=:both, workspace=nothing) -> subbands

In-place DFB analysis into preallocated `subbands` of the sizes
[`dfb_subband_sizes`](@ref) gives.  A workspace from [`make_workspace`](@ref) whose
image size and depth match supplies the precomputed filters and FFT buffers, so the
call allocates nothing.
"""
function dfb_decompose!(
        subbands::Vector{<:AbstractMatrix}, bandpass::AbstractMatrix, l_levels::Int,
        qfp::QuincunxFilterPair;
        half::Symbol = :both,
        workspace = nothing
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    _check_half(half)
    length(subbands) == _bank_n_subbands(l_levels, half) ||
        throw(ArgumentError("subbands must hold $(_bank_n_subbands(l_levels, half)) matrices"))
    l_levels == 0 && return (copyto!(subbands[1], bandpass); subbands)
    Tf = _filter_eltype(_data_eltype(bandpass))
    return _with_runner(r -> _bank_analyze!(subbands, bandpass, r), workspace, bandpass, l_levels, qfp, half, Tf)
end

"""
    dfb_reconstruct(subbands, qfp::QuincunxFilterPair; half=:both, workspace=nothing) -> bandpass

Reconstruct a bandpass image from its directional subbands (pass the same `half`
as the analysis).

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(3), 32, 32);

julia> sbs = dfb_decompose(x, 2, Q2345);

julia> rec = dfb_reconstruct(sbs, Q2345);

julia> maximum(abs, rec .- x) < 1e-10
true
```
"""
function dfb_reconstruct(
        subbands::Vector{<:AbstractMatrix},
        qfp::QuincunxFilterPair;
        half::Symbol = :both,
        workspace = nothing
    )
    n = length(subbands)
    n >= 1 || throw(ArgumentError("subbands must be non-empty"))
    _check_half(half)
    n == 1 && return copy(subbands[1])
    l_levels = _dfb_levels_from_count(n, half)
    Td = _data_eltype(subbands[1])
    Tf = _filter_eltype(Td)
    dims = _dfb_image_size(subbands, l_levels, half)
    out = similar(subbands[1], Td, dims)
    return _dfb_reconstruct_bank!(out, subbands, l_levels, qfp, half, workspace, Tf)
end

"""
    dfb_reconstruct!(bandpass::AbstractMatrix, subbands::Vector{<:AbstractMatrix}, qfp::QuincunxFilterPair;
                     half=:both, workspace=nothing) -> bandpass

In-place DFB synthesis into `bandpass`.  A matching workspace from
[`make_workspace`](@ref) makes the call allocation-free.
"""
function dfb_reconstruct!(
        bandpass::AbstractMatrix, subbands::Vector{<:AbstractMatrix}, qfp::QuincunxFilterPair;
        half::Symbol = :both,
        workspace = nothing
    )
    _check_half(half)
    n = length(subbands)
    n == 1 && return copyto!(bandpass, subbands[1])
    l_levels = _dfb_levels_from_count(n, half)
    Tf = _filter_eltype(_data_eltype(bandpass))
    return _dfb_reconstruct_bank!(bandpass, subbands, l_levels, qfp, half, workspace, Tf)
end

function _dfb_reconstruct_bank!(out, subbands, l_levels, qfp, half, workspace, ::Type{Tf}) where {Tf}
    return _with_runner(r -> _bank_synthesize!(out, subbands, r), workspace, out, l_levels, qfp, half, Tf)
end

# Tree depth from the subband count of a full (2ˡ) or half (2^(l-1) + 1) tree.
function _dfb_levels_from_count(n::Int, half::Symbol)
    if half === :both
        ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
        return round(Int, log2(n))
    end
    (n >= 2 && ispow2(n - 1)) || throw(ArgumentError("a half tree has 2^(l-1) + 1 subbands"))
    return round(Int, log2(n - 1)) + 1
end

# Size of the bandpass image the subbands of a ladder DFB came from.
function _dfb_image_size(subbands, l::Int, half::Symbol)
    m1, m2 = size(subbands[1])
    l == 1 && return (2m1, m2)
    # the first subband is a split channel unless the first fan is left unsplit
    half === :second && return (2m1, m2)
    return (m1 * 2^(l - 1), 2m2)
end

"""
    dfb_subband_sizes(n1, n2, l; half=:both) -> Vector{Tuple{Int,Int}}

Per-subband `(rows, cols)` sizes produced by an `l`-level DFB applied to an
`n1 × n2` bandpass image.  With `l ≥ 2` the first `2^(l-1)` subbands are
`n1/2^(l-1) × n2/2` and the remaining `2^(l-1)` are `n1/2 × n2/2^(l-1)`; with
`l = 1` both are `n1/2 × n2` (the quincunx lattice).  A half tree
(`half = :first`/`:second`) keeps the split fan's sizes and adds one `n1/2 × n2`
quincunx subband for the unsplit fan.
"""
function dfb_subband_sizes(n1::Int, n2::Int, l::Int; half::Symbol = :both)
    l <= 0 && return [(n1, n2)]
    _check_half(half)
    q = (n1 >> 1, n2)
    l == 1 && return fill(q, 2)
    m = 2^(l - 1)
    first_sz = (n1 ÷ m, n2 >> 1)   # wedges below the first fan
    second_sz = (n1 >> 1, n2 ÷ m)  # wedges below the second fan
    half === :first && return vcat(fill(first_sz, m), [q])
    half === :second && return vcat([q], fill(second_sz, m))
    return vcat(fill(first_sz, m), fill(second_sz, m))
end
