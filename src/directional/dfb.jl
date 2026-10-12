# l-level Directional Filter Bank (DFB).
#
# Two constructions are provided:
#
#  • Ladder filter pairs (the default, Q2345): the DFB of Bamberger & Smith in the
#    formulation of Do (2001, thesis §3.3) and Do & Vetterli (2005), with the
#    quincunx fan filters of Phoong et al. (1995).  It is computed as the
#    multichannel bank of the thesis — each channel filters with its equivalent
#    filter and downsamples by a diagonal matrix — in the frequency domain (see
#    `directional_bank.jl`).  The 2ˡ subbands tile the frequency plane into
#    directional wedges, perfect reconstruction is exact, and each subband is
#    equivariant to circular shifts on its own sampling lattice.
#
#  • Modulation-mode pairs (e.g. a user-supplied Haar pair): a simpler shear +
#    1-D split tree.  It is perfect-reconstructing but only weakly directional; it
#    is the only decimated DFB available for filter pairs that are not ladders.
#
# Subband sizes after l levels (input n₁ × n₂, ladder pairs):
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

# Move a host bank to the array family of `ref` (identity on the host; the GPU
# extension adds methods).
_adapt_bank(::AbstractArray, bank::_DirectionalBank) = bank

# The runner a workspace holds for this size and depth, or `nothing`.
_workspace_runner(::Nothing, dims, l, half) = nothing

# Call `f(runner)` with the workspace's runner when it has one for `x`, otherwise
# with a freshly built one.  Each branch calls `f` on its own concrete runner type,
# so the workspace path stays type-stable and allocation-free.
function _with_runner(f, workspace, x::AbstractMatrix, l::Int, qfp::QuincunxFilterPair, half::Symbol, ::Type{Tf}) where {Tf}
    r = _workspace_runner(workspace, size(x), l, half)
    r === nothing || return f(r)
    return f(_dfb_runner(x, l, qfp, half, Tf))
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    dfb_decompose(bandpass, l_levels, qfp::QuincunxFilterPair; half=:both, threading=Auto())
        -> Vector{Matrix}

Decompose `bandpass` into `2^l_levels` directional subbands using an
`l_levels`-deep DFB.

For a ladder filter pair such as the default [`Q2345`](@ref) this is the directional
filter bank of Do & Vetterli (2005): the subbands tile the frequency plane into
`2^l_levels` directional wedges.  Both image dimensions must be divisible by
`2^max(l_levels - 1, 1)`.  For modulation-mode pairs a simpler shear-based tree is
used (perfect-reconstructing but only weakly directional).

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
        workspace = nothing,
        threading::ThreadingPolicy = Auto()
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    _check_half(half)
    l_levels == 0 && return [copy(bandpass)]
    Td = _data_eltype(bandpass)      # data type (real or complex)
    Tf = _filter_eltype(Td)          # real filter precision
    img = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    if is_ladder(qfp)
        return _with_runner(workspace, img, l_levels, qfp, half, Tf) do r
            _bank_analyze!([similar(img, s) for s in _subband_sizes(r.bank)], img, r)
        end
    end
    half === :both || throw(ArgumentError("half trees need a ladder filter pair"))
    return _dfb_split(img, l_levels, 1, _convert_qfp(Tf, qfp), _use_threading(threading, Td))
end

"""
    dfb_decompose!(subbands::Vector{<:AbstractMatrix}, bandpass, l_levels, qfp::QuincunxFilterPair;
                   half=:both, workspace=nothing, threading=Auto()) -> subbands

In-place DFB analysis into preallocated `subbands` of the sizes
[`dfb_subband_sizes`](@ref) gives.  A workspace from [`make_workspace`](@ref) whose
image size and depth match supplies the precomputed filters and FFT buffers, so the
call allocates nothing.

**Keyword arguments:**
- `threading::ThreadingPolicy`: Sets the threading strategy of the modulation-mode
  tree. Defaults to `Auto()`, which uses SIMD (single-threaded) for Real arrays, and
  `Polyester.@batch` (multithreaded) for Complex arrays.
"""
function dfb_decompose!(
        subbands::Vector{<:AbstractMatrix}, bandpass::AbstractMatrix, l_levels::Int,
        qfp::QuincunxFilterPair;
        half::Symbol = :both,
        workspace = nothing,
        threading::ThreadingPolicy = Auto()
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    _check_half(half)
    length(subbands) == _bank_n_subbands(l_levels, half) ||
        throw(ArgumentError("subbands must hold $(_bank_n_subbands(l_levels, half)) matrices"))
    l_levels == 0 && return (copyto!(subbands[1], bandpass); subbands)
    if is_ladder(qfp)
        Tf = _filter_eltype(_data_eltype(bandpass))
        return _with_runner(r -> _bank_analyze!(subbands, bandpass, r), workspace, bandpass, l_levels, qfp, half, Tf)
    end
    sbs = dfb_decompose(bandpass, l_levels, qfp; half = half, threading = threading)
    for k in eachindex(sbs)
        copyto!(subbands[k], sbs[k])
    end
    return subbands
end

"""
    dfb_reconstruct(subbands, qfp::QuincunxFilterPair; half=:both, threading=Auto()) -> bandpass

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
        workspace = nothing,
        threading::ThreadingPolicy = Auto()
    )
    n = length(subbands)
    n >= 1 || throw(ArgumentError("subbands must be non-empty"))
    _check_half(half)
    n == 1 && return copy(subbands[1])
    l_levels = _dfb_levels_from_count(n, half)
    Td = _data_eltype(subbands[1])
    Tf = _filter_eltype(Td)
    if is_ladder(qfp)
        dims = _dfb_image_size(subbands, l_levels, half)
        out = similar(subbands[1], Td, dims)
        return _dfb_reconstruct_ladder!(out, subbands, l_levels, qfp, half, workspace, Tf)
    end
    half === :both || throw(ArgumentError("half trees need a ladder filter pair"))
    return _dfb_merge(subbands, l_levels, 1, _convert_qfp(Tf, qfp), _use_threading(threading, Td))
end

"""
    dfb_reconstruct!(bandpass::AbstractMatrix, subbands::Vector{<:AbstractMatrix}, qfp::QuincunxFilterPair;
                     half=:both, workspace=nothing, threading=Auto()) -> bandpass

In-place DFB synthesis into `bandpass`.  A matching workspace from
[`make_workspace`](@ref) makes the call allocation-free.
"""
function dfb_reconstruct!(
        bandpass::AbstractMatrix, subbands::Vector{<:AbstractMatrix}, qfp::QuincunxFilterPair;
        half::Symbol = :both,
        workspace = nothing,
        threading::ThreadingPolicy = Auto()
    )
    _check_half(half)
    n = length(subbands)
    n == 1 && return copyto!(bandpass, subbands[1])
    if is_ladder(qfp)
        l_levels = _dfb_levels_from_count(n, half)
        Tf = _filter_eltype(_data_eltype(bandpass))
        return _dfb_reconstruct_ladder!(bandpass, subbands, l_levels, qfp, half, workspace, Tf)
    end
    copyto!(bandpass, dfb_reconstruct(subbands, qfp; half = half, threading = threading))
    return bandpass
end

function _dfb_reconstruct_ladder!(out, subbands, l_levels, qfp, half, workspace, ::Type{Tf}) where {Tf}
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
    dfb_subband_sizes(n1, n2, l; ladder=true, half=:both) -> Vector{Tuple{Int,Int}}

Per-subband `(rows, cols)` sizes produced by an `l`-level DFB applied to an
`n1 × n2` bandpass image.  For the ladder (default) DFB with `l ≥ 2` the first
`2^(l-1)` subbands are `n1/2^(l-1) × n2/2` and the remaining `2^(l-1)` are
`n1/2 × n2/2^(l-1)`; with `l = 1` both are `n1/2 × n2` (the quincunx lattice).  A
half tree (`half = :first`/`:second`) keeps the split fan's sizes and adds one
`n1/2 × n2` quincunx subband for the unsplit fan.  For the modulation-mode DFB all
subbands are equal-sized.
"""
function dfb_subband_sizes(n1::Int, n2::Int, l::Int; ladder::Bool = true, half::Symbol = :both)
    l <= 0 && return [(n1, n2)]
    if !ladder
        sub_n1 = n1 >> (l ÷ 2)
        sub_n2 = n2 >> cld(l, 2)
        return fill((sub_n1, sub_n2), 2^l)
    end
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

# ── Modulation-mode shear path (modulation-mode pairs only) ───────────────────

function _dfb_split(
        img::AbstractMatrix, remaining::Int, depth::Int,
        qfp::QuincunxFilterPair, threaded::Bool = false
    )
    shear_dir = isodd(depth) ? :h : :v
    qfb_dir = isodd(depth) ? :col : :row
    sh = shear(img, shear_dir)
    _sb0, _sb1 = qfb_decompose(sh, qfp; dir = qfb_dir)
    sb0 = inv_shear(_sb0, shear_dir)
    sb1 = inv_shear(_sb1, shear_dir)
    remaining == 1 && return [sb0, sb1]

    if threaded
        t0 = Threads.@spawn _dfb_split(sb0, remaining - 1, depth + 1, qfp, threaded)
        r1 = _dfb_split(sb1, remaining - 1, depth + 1, qfp, threaded)
        r0 = fetch(t0)::typeof(r1)
        return vcat(r0, r1)
    else
        return vcat(
            _dfb_split(sb0, remaining - 1, depth + 1, qfp, threaded),
            _dfb_split(sb1, remaining - 1, depth + 1, qfp, threaded)
        )
    end
end

function _dfb_merge(
        sbs::Vector{<:AbstractMatrix}, l::Int, depth::Int,
        qfp::QuincunxFilterPair, threaded::Bool = false
    )
    shear_dir = isodd(depth) ? :h : :v
    qfb_dir = isodd(depth) ? :col : :row
    half = length(sbs) ÷ 2
    if l == 1
        sb0, sb1 = sbs[1], sbs[2]
    elseif threaded
        t0 = Threads.@spawn _dfb_merge(sbs[1:half], l - 1, depth + 1, qfp, threaded)
        sb1 = _dfb_merge(sbs[(half + 1):end], l - 1, depth + 1, qfp, threaded)
        sb0 = fetch(t0)::typeof(sb1)
    else
        sb0 = _dfb_merge(sbs[1:half], l - 1, depth + 1, qfp, threaded)
        sb1 = _dfb_merge(sbs[(half + 1):end], l - 1, depth + 1, qfp, threaded)
    end
    sh0 = shear(sb0, shear_dir)
    sh1 = shear(sb1, shear_dir)
    rec = qfb_reconstruct(sh0, sh1, qfp; dir = qfb_dir)
    return inv_shear(rec, shear_dir)
end
