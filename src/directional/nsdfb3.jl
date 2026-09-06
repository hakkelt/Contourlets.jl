# 3-D nonsubsampled directional filter bank (NDFB) — hourglass stage refined
# into wedges.
#
# Reference: Y. M. Lu and M. N. Do, "Multidimensional Directional Filter Banks
# and Surfacelets", IEEE Trans. Image Processing 16(4):918-931, 2007.
#
# Structure (paper, Fig. 7).  Each of the three hourglass subbands `P_i` is fed
# through two 2-D filter banks operating on the two planes that contain axis `i`:
# `l_a` levels on the first, `l_b` on the second, giving `2^(l_a + l_b)` wedges
# per hourglass and `3 · 2^L` directional subbands in total for `L = l_a + l_b`.
# The ideal passbands are rectangular-based pyramids radiating from the origin.
#
# Wedge geometry.  Inside `P_i` the two transverse ratios
#
#     s = ω_a / ω_i ∈ [-1, 1],    t = ω_b / ω_i ∈ [-1, 1]
#
# parameterise direction, and an `l`-level refinement cuts each into `2^l` equal
# slabs.  Both ratios are invariant under `ω → -ω`, so the masks stay symmetric
# and real volumes keep real subbands.  Each ratio involves only two frequency
# axes, so a wedge mask is a *matrix*, not a volume — the whole bank stores
# `2·(2^l_a + 2^l_b)` planes regardless of how large the data is.
#
# Deviation from the paper, deliberate: Lu & Do build these wedges with a chain
# of iteratively resampled checkerboard filter banks (IRC), because their
# transform is critically sampled inside each hourglass and the resampling is
# what keeps it so.  This bank is nonsubsampled, so there is nothing to resample
# for; the wedge boundaries are placed directly in the frequency domain instead.
# The resulting *ideal* partition is the same one, and getting it directly avoids
# the failure mode the 2-D `nsdfb` hit when its wedges were approximated by
# separable 1-D filtering — that yields stripe/block tilings, not wedges.
#
# Perfect reconstruction stays structural: for each ratio the soft steps
# telescope to 1, so the wedge masks of a hourglass sum to that hourglass's mask,
# the three hourglass masks sum to 1, and reconstruction is a plain sum.

"""
    NSDFB3{T}

3-D nonsubsampled directional filter bank: [`HourglassFB3`](@ref) followed by a
wedge refinement of each hourglass.  Build with [`make_nsdfb3`](@ref).
"""
struct NSDFB3{T <: AbstractFloat, FB <: HourglassFB3{T}}
    fb::FB
    levels::NTuple{2, Int}          # (l_a, l_b) refinement levels per hourglass
    # Wedge masks, `[hourglass][which plane][wedge]`, each a matrix over
    # (reference axis, transverse axis) of the folded DFT grid.
    wedges::NTuple{3, NTuple{2, Vector{Matrix{T}}}}
end

"""
    make_nsdfb3(dims::NTuple{3,Int}, ::Type{T}=Float64;
                levels::Int=2, order::Int=12, poly::Int=1,
                transition::Real=1.0, threading=Auto(), flags=FFTW.MEASURE)

Build a 3-D NSDFB producing `3 * 2^levels` directional subbands, each the full
size of the input.

`levels` is the total refinement depth `L`; it is split as evenly as possible
between the two transverse planes of each hourglass (`l_a = ceil(L/2)`,
`l_b = floor(L/2)`), matching the balanced expansion rule of the paper.
`levels = 0` leaves the plain three-subband hourglass bank.

`order` and `poly` are forwarded to [`make_hourglass3`](@ref).  `transition`
scales the width of the overlap between neighbouring wedges, as a fraction of
the wedge width: `0` gives hard boundaries, `1` (the default) makes the
transition as wide as one wedge.
"""
function make_nsdfb3(
        dims::NTuple{3, Int}, ::Type{T} = Float64;
        levels::Int = 2, order::Int = 12, poly::Int = 1,
        transition::Real = 1.0,
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    levels >= 0 || throw(ArgumentError("levels must be ≥ 0"))
    0 <= transition || throw(ArgumentError("transition must be ≥ 0"))
    fb = make_hourglass3(dims, T; order = order, poly = poly, threading = threading, flags = flags)
    la = cld(levels, 2)
    lb = fld(levels, 2)
    wedges = ntuple(i -> _wedge_masks(fb, i, (la, lb), poly, T(transition)), 3)
    return NSDFB3{T, typeof(fb)}(fb, (la, lb), wedges)
end

function make_nsdfb3(dims::NTuple{3, Int}, ::Type{Complex{T}}; kwargs...) where {T <: AbstractFloat}
    return make_nsdfb3(dims, T; kwargs...)
end

make_nsdfb3(n1::Int, n2::Int, n3::Int, ::Type{Td} = Float64; kwargs...) where {Td} =
    make_nsdfb3((n1, n2, n3), Td; kwargs...)

# The two transverse axes of hourglass `i`, in increasing order.
@inline _transverse_axes(i::Int) = i == 1 ? (2, 3) : (i == 2 ? (1, 3) : (1, 2))

# Folded angular frequencies of the DFT grid: ω ∈ (-π, π].  Axis 1 is the rfft
# axis, whose stored half already lies in [0, π].
function _folded_frequencies(::Type{T}, n::Int, axis::Int) where {T}
    if axis == 1
        return dft_frequencies(T, n; half = true)
    end
    return T[
        let w = 2 * T(π) * (k - 1) / n
                w > T(π) ? w - 2 * T(π) : w
        end for k in 1:n
    ]
end

# Soft step from 0 to 1 across `b ± w`, built from the same halfband polynomial
# as the hourglass split so that consecutive steps telescope exactly.
@inline function _soft_step(s::T, b::T, w::T, poly::Int) where {T}
    w <= 0 && return s >= b ? one(T) : zero(T)
    return halfband_split(clamp((s - b) / w, -one(T), one(T)), poly)
end

# Wedge masks for one hourglass: one vector of matrices per transverse plane.
function _wedge_masks(
        fb::HourglassFB3{T}, i::Int, levels::NTuple{2, Int}, poly::Int, transition::T
    ) where {T}
    a, b = _transverse_axes(i)
    return (
        _wedge_masks_plane(fb, i, a, levels[1], poly, transition),
        _wedge_masks_plane(fb, i, b, levels[2], poly, transition),
    )
end

function _wedge_masks_plane(
        fb::HourglassFB3{T}, ref::Int, tr::Int, l::Int, poly::Int, transition::T
    ) where {T}
    n = fb.dims
    wref = _folded_frequencies(T, n[ref], ref)
    wtr = _folded_frequencies(T, n[tr], tr)
    nw = 1 << l
    masks = [Matrix{T}(undef, length(wref), length(wtr)) for _ in 1:nw]
    l == 0 && (fill!(masks[1], one(T)); return masks)

    delta = T(2) / nw
    width = transition * delta
    @inbounds for (q, wt) in enumerate(wtr), (p, wr) in enumerate(wref)
        # Ratio parameterising direction inside the hourglass; the value at
        # ω_ref = 0 is arbitrary because the masks sum to 1 for any `s`.
        s = wr == 0 ? zero(T) : wt / wr
        # The outermost steps are hard (1 and 0) so the telescoping sum is
        # exactly 1 for every `s`, including |s| > 1 outside the hourglass.
        prev = one(T)
        for k in 1:nw
            cur = k == nw ? zero(T) : _soft_step(s, -one(T) + k * delta, width, poly)
            masks[k][p, q] = prev - cur
            prev = cur
        end
    end
    return masks
end

"""
    nsdfb3_nsubbands(fb::NSDFB3) -> Int

Number of directional subbands, `3 * 2^levels`.
"""
nsdfb3_nsubbands(fb::NSDFB3) = 3 * (1 << sum(fb.levels))

"""
    nsdfb3_subband_index(fb::NSDFB3, hourglass, p, q) -> Int

Linear index of the subband refining `hourglass` (1-3) at wedge `p` of its first
transverse plane and wedge `q` of its second.  Subbands are ordered hourglass
first, then `p`, then `q`.
"""
function nsdfb3_subband_index(fb::NSDFB3, hourglass::Int, p::Int, q::Int)
    na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
    1 <= hourglass <= 3 || throw(ArgumentError("hourglass index must be 1, 2 or 3"))
    1 <= p <= na || throw(ArgumentError("wedge index p out of range 1:$(na)"))
    1 <= q <= nb || throw(ArgumentError("wedge index q out of range 1:$(nb)"))
    return ((hourglass - 1) * na + (p - 1)) * nb + q
end

# Inverse of `nsdfb3_subband_index`.
function _subband_triple(fb::NSDFB3, idx::Int)
    na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
    1 <= idx <= 3 * na * nb || throw(ArgumentError("subband index out of range"))
    z = idx - 1
    q = z % nb + 1
    z ÷= nb
    p = z % na + 1
    return (z ÷ na + 1, p, q)
end

"""
    nsdfb3_mask(fb::NSDFB3, idx::Int) -> Array{T,3}

Frequency-domain mask of directional subband `idx` on the rfft grid.  For
inspection and tests; the transform never materialises one.  The masks sum to 1
over all subbands.
"""
function nsdfb3_mask(fb::NSDFB3{T}, idx::Int) where {T}
    i, p, q = _subband_triple(fb, idx)
    out = similar(fb.fb.m1)
    _fill_mask!(out, fb, i, p, q)
    return out
end

# Shared mask kernel: hourglass mask times the two wedge factors.  `wa` and `wb`
# are indexed by (reference axis, transverse axis), so which of the three loop
# indices to use depends on the hourglass.
function _fill_mask!(out::AbstractArray{T, 3}, fb::NSDFB3{T}, i::Int, p::Int, q::Int) where {T}
    hg = fb.fb
    wa = fb.wedges[i][1][p]
    wb = fb.wedges[i][2][q]
    a, b = _transverse_axes(i)
    poly = hg.poly
    @inbounds for k in axes(out, 3), j in axes(out, 2)
        w2 = _stage2_weight(hg.m2[j, k], poly, i)
        for pp in axes(out, 1)
            ii = (pp, j, k)
            va = wa[ii[i], ii[a]]
            vb = wb[ii[i], ii[b]]
            out[pp, j, k] = _stage1_weight(hg.m1[pp, j, k], poly, i) * w2 * va * vb
        end
    end
    return out
end

"""
    nsdfb3_decompose(volume, fb::NSDFB3) -> Vector

All `3 * 2^levels` directional subbands, each the size of `volume`.

Memory warning: at 256³ `Float32` a single subband is 64 MB, so `levels = 3`
(24 subbands) materialises 1.5 GB.  Use the callback form below to process
subbands one at a time instead.
"""
function nsdfb3_decompose(volume::AbstractArray{Td, 3}, fb::NSDFB3) where {Td}
    out = [Array{Td, 3}(undef, fb.fb.dims) for _ in 1:nsdfb3_nsubbands(fb)]
    nsdfb3_decompose!(out, volume, fb)
    return out
end

"""
    nsdfb3_decompose!(dst, volume, fb::NSDFB3) -> dst

In-place [`nsdfb3_decompose`](@ref); `dst` is an indexable collection of
`nsdfb3_nsubbands(fb)` arrays.
"""
function nsdfb3_decompose!(dst, volume::AbstractArray{Td, 3}, fb::NSDFB3) where {Td}
    length(dst) == nsdfb3_nsubbands(fb) ||
        throw(DimensionMismatch("dst must hold $(nsdfb3_nsubbands(fb)) subbands"))
    nsdfb3_decompose(volume, fb) do idx, band
        copyto!(dst[idx], band)
    end
    return dst
end

"""
    nsdfb3_decompose(f, volume, fb::NSDFB3)

Streaming decomposition: calls `f(idx, subband)` once per directional subband,
reusing a single scratch buffer between calls.  `subband` is only valid until
`f` returns — copy or reduce it there.

This is the form to use on real volumes: it holds one subband at a time instead
of `3 * 2^levels` of them.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(4), 16, 16, 16);

julia> fb = make_nsdfb3(size(x), Float64; levels = 2, order = 6);

julia> energy = zeros(nsdfb3_nsubbands(fb));

julia> nsdfb3_decompose(x, fb) do i, band
           energy[i] = sum(abs2, band)
       end;

julia> abs(sum(energy) / sum(abs2, x) - 1) < 2.0
true
```
"""
function nsdfb3_decompose(f, volume::AbstractArray{T, 3}, fb::NSDFB3{T}) where {T <: AbstractFloat}
    hg = fb.fb
    size(volume) == hg.dims || throw(DimensionMismatch("volume size does not match the plan"))
    copyto!(hg.rbuf, volume)
    mul!(hg.spec, hg.fwd, hg.rbuf)
    _stream_subbands(f, fb)
    return nothing
end

function nsdfb3_decompose(
        f, volume::AbstractArray{Complex{T}, 3}, fb::NSDFB3{T}
    ) where {T <: AbstractFloat}
    hg = fb.fb
    size(volume) == hg.dims || throw(DimensionMismatch("volume size does not match the plan"))
    # Real and imaginary parts need to be recombined per subband, so unlike the
    # real path this one keeps a complex scratch volume alive.
    tmp = Array{Complex{T}, 3}(undef, hg.dims)
    @inbounds @simd for i in eachindex(hg.rbuf, volume)
        hg.rbuf[i] = real(volume[i])
    end
    mul!(hg.spec, hg.fwd, hg.rbuf)
    parts = Vector{Array{T, 3}}(undef, nsdfb3_nsubbands(fb))
    _stream_subbands(fb) do idx, band
        parts[idx] = copy(band)
    end
    @inbounds @simd for i in eachindex(hg.rbuf, volume)
        hg.rbuf[i] = imag(volume[i])
    end
    mul!(hg.spec, hg.fwd, hg.rbuf)
    _stream_subbands(fb) do idx, band
        @inbounds @simd for i in eachindex(tmp, band)
            tmp[i] = Complex{T}(parts[idx][i], band[i])
        end
        f(idx, tmp)
    end
    return nothing
end

# Walk every (hourglass, p, q) triple, applying its mask to the spectrum already
# sitting in `hg.spec` and handing the inverse transform to `g`.
function _stream_subbands(g, fb::NSDFB3{T}) where {T}
    hg = fb.fb
    na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
    mask = Array{T, 3}(undef, size(hg.m1))
    for i in 1:3, p in 1:na, q in 1:nb
        _fill_mask!(mask, fb, i, p, q)
        @inbounds for t in eachindex(hg.work, hg.spec, mask)
            hg.work[t] = hg.spec[t] * mask[t]
        end
        mul!(hg.rbuf, hg.inv, hg.work)
        g(nsdfb3_subband_index(fb, i, p, q), hg.rbuf)
    end
    return nothing
end

"""
    nsdfb3_reconstruct(subbands, fb::NSDFB3) -> volume

Invert [`nsdfb3_decompose`](@ref).  The masks are a partition of unity, so this
is a plain sum and perfect reconstruction is exact.
"""
function nsdfb3_reconstruct(subbands, fb::NSDFB3)
    out = similar(first(subbands))
    return nsdfb3_reconstruct!(out, subbands, fb)
end

"""
    nsdfb3_reconstruct!(volume, subbands, fb::NSDFB3) -> volume

In-place [`nsdfb3_reconstruct`](@ref).
"""
function nsdfb3_reconstruct!(volume::AbstractArray, subbands, fb::NSDFB3)
    size(volume) == fb.fb.dims || throw(DimensionMismatch("volume size does not match the plan"))
    length(subbands) == nsdfb3_nsubbands(fb) ||
        throw(DimensionMismatch("expected $(nsdfb3_nsubbands(fb)) subbands"))
    fill!(volume, zero(eltype(volume)))
    for b in subbands
        @. volume += b
    end
    return volume
end
