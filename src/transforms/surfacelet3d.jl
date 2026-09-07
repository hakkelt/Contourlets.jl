# 3-D surfacelet transform — the decimated counterpart of the 3-D NSCT.
#
# Reference for the target frequency partitioning: Y. M. Lu and M. N. Do,
# "Multidimensional Directional Filter Banks and Surfacelets", IEEE TIP 16(4),
# 2007, Section VI.  The ideal passbands are pairs of concentric cubes radiating
# out from the origin at different scales and directions.
#
# Why this exists.  The 3-D NSCT is exactly shift-invariant and correspondingly
# redundant: `1 + Σ_j 3·2^{L_j}` copies of the input, which is 49 volumes (3.1 GB
# at 256³ Float32) for J=3, L=[2,2,3].  The surfacelet transform trades shift
# invariance for a redundancy of a small constant, which is what makes it usable
# as a sparsifying transform rather than only as an analysis tool.
#
# How it differs from the reference construction — deliberately, and worth
# reading before comparing coefficients with SurfBox:
#
#   * Lu & Do decimate with a multiscale pyramid whose first level resamples by
#     the non-integer factor 3/2 per axis, plus a critically sampled chain of
#     iteratively resampled checkerboard filter banks inside each hourglass.  The
#     numeric filter parameters for that pyramid are given in tables in the paper
#     which are not recoverable from its text, and the reference implementation
#     (SurfBox) is no longer distributed, so reproducing it exactly is not
#     possible from the sources at hand.
#
#   * Instead, decimation here is done by frequency-domain support extraction:
#     each subband mask has *compact* support in frequency, so the masked
#     spectrum is restricted to the smallest symmetric box containing it and
#     inverse-transformed at that reduced size.  This is the same mechanism the
#     wrapping-based discrete curvelet transform uses.  The resulting redundancy
#     is measured, not assumed — see `surfacelet3d_redundancy`.
#
# How far that gets, honestly.  Measured against the nonsubsampled transform at
# the same parameters:
#
#   | J, L          | surfacelet3d | NSCT3D |
#   |---------------|-------------|-------|
#   | 1, [1]        |  6.2        |   7   |
#   | 2, [1, 1]     |  7.7        |  13   |
#   | 2, [2, 2]     | 15.3        |  25   |
#   | 3, [2, 2, 3]  | 16.0        |  49   |
#
# So this is a partially decimated transform: about 3× leaner than the NSCT3D at
# useful settings, not the redundancy of about 4 that Lu & Do report.  The reason
# is specific and worth knowing before trying to tune the parameters for it: an
# *axis-aligned* box recovers the radial decimation in full (a coarser shell
# shrinks by 2 per axis) but almost none of the directional decimation, because
# the bounding box of a pyramid radiating from the origin is nearly as large as
# the pyramid's enclosing cube.  Closing that gap needs the box to be aligned
# with the wedge — sheared wrapping, or equivalently Lu & Do's resampled
# checkerboard chain — which in turn needs the shear to be an integer index
# remap on the sampling grid.  It is not, at dyadic wedge boundaries, unless the
# decimation is interleaved with the refinement the way the reference
# construction does it.  That is the missing piece, not a tuning problem.
#
#   * Consequently the directional masks are *not* the Bernstein mapping kernels
#     of `hourglass3d.jl`.  Those are the right choice there and the wrong one
#     here: a mapping kernel is a polynomial in `cos ω`, so `f₀(M(ω))` is nonzero
#     almost everywhere and its support box is the whole grid — nothing to
#     decimate.  The requirement inverts, so the construction does too: masks
#     here are built from soft steps that are *exactly* 0 and 1 outside a
#     transition band.
#
# Perfect reconstruction.  The masks `W` form a partition of unity and the
# transform uses `V = √W`, so `Σ V² = 1` and synthesis is the analysis adjoint:
# extraction is lossless because each mask vanishes outside its box, so
# `fft(ifft(restrict(Y))) == restrict(Y)` exactly.  Reconstruction is therefore
# exact, and the frame is tight in the weighted sense
# `Σ_i N_i ‖s_i‖² = N ‖x‖²` where `N_i` is the subband's element count.

"""
    SurfaceletParams3D

Describes a 3-D surfacelet decomposition.

- `J` — number of scales.  Scale `j` covers the frequency shell
  `‖ω‖_∞ ∈ [π/2^j, π/2^{j-1}]`, so it decimates by roughly `2^{j-1}` per axis.
- `L_array` — directional depth per scale, length `J`; scale `j` has
  `3 · 2^{L_array[j]}` directional subbands.
- `poly` — order of the soft-step polynomial (transition smoothness).
- `transition` — width of the overlap between neighbouring wedges, as a fraction
  of the wedge width.
- `radial_transition` — width of the overlap between neighbouring scale shells,
  as a fraction of the shell width.
- `angular_transition` — width, in radians of `|ω_i| - |ω_j|`, of the overlap
  between neighbouring hourglasses.

Wider transitions mean smoother, better-localised surfacelets and larger support
boxes; that trade-off is exactly the redundancy dial.

```jldoctest
julia> using Contourlets

julia> SurfaceletParams3D(J = 2, L_array = [1, 2])
SurfaceletParams3D(J=2, L_array=[1, 2], …)
```
"""
struct SurfaceletParams3D{T <: AbstractFloat}
    J::Int
    L_array::Vector{Int}
    poly::Int
    transition::T
    radial_transition::T
    angular_transition::T
end

function SurfaceletParams3D(;
        J::Int,
        L_array::AbstractVector{<:Integer},
        poly::Int = 1,
        transition::Real = 0.5,
        radial_transition::Real = 0.3,
        angular_transition::Real = 0.3,
        precision::Type{T} = Float64
    ) where {T <: AbstractFloat}
    J >= 1 || throw(ArgumentError("J must be ≥ 1"))
    length(L_array) == J || throw(ArgumentError("length(L_array) must equal J"))
    all(>=(0), L_array) || throw(ArgumentError("L_array entries must be ≥ 0"))
    poly >= 0 || throw(ArgumentError("poly must be ≥ 0"))
    transition >= 0 || throw(ArgumentError("transition must be ≥ 0"))
    radial_transition > 0 || throw(ArgumentError("radial_transition must be > 0"))
    angular_transition > 0 || throw(ArgumentError("angular_transition must be > 0"))
    return SurfaceletParams3D{T}(
        J, collect(Int, L_array), poly,
        T(transition), T(radial_transition), T(angular_transition)
    )
end

Base.eltype(::SurfaceletParams3D{T}) where {T} = T

function Base.show(io::IO, p::SurfaceletParams3D)
    return print(io, "SurfaceletParams3D(J=$(p.J), L_array=$(p.L_array), …)")
end

"""
    surfacelet3d_nsubbands(p::SurfaceletParams3D, j::Int) -> Int

Number of directional subbands at scale `j`, `3 * 2^L_array[j]`.
"""
surfacelet3d_nsubbands(p::SurfaceletParams3D, j::Int) = 3 * (1 << p.L_array[j])

# One subband's frequency box: per-axis index maps into the box, and its size.
struct SubbandBox
    size::NTuple{3, Int}
    maps::NTuple{3, Vector{Int}}   # full-grid index → box index, 0 when outside
end

"""
    Surfacelet3DPlan{T}

Precomputed masks metadata and FFT plans for one volume size.  Build with
[`make_surfacelet3d`](@ref).

The per-subband masks are *not* stored — at 256³ that would be gigabytes.  Only
the 1-D frequency tables, the per-scale wedge boundaries, and each subband's
frequency box are kept; masks are evaluated on the fly.
"""
struct Surfacelet3DPlan{T <: AbstractFloat, P}
    dims::NTuple{3, Int}
    params::SurfaceletParams3D{T}
    wfreq::NTuple{3, Vector{T}}      # ω_d folded to (-π, π]
    signed::NTuple{3, Vector{Int}}   # signed frequency index per axis
    negidx::NTuple{3, Vector{Int}}   # index of the negated frequency, per axis
    radii::Vector{T}                 # shell boundaries π, π/2, …, π/2^J
    boxes::Vector{Vector{SubbandBox}}   # [scale][subband]
    coarse_box::SubbandBox
    fwd::P
    spec::Array{Complex{T}, 3}
    work::Array{Complex{T}, 3}
end

"""
    make_surfacelet3d(params::SurfaceletParams3D, dims::NTuple{3,Int}, ::Type{T}=Float64;
                     threading=Auto(), flags=FFTW.MEASURE) -> Surfacelet3DPlan

Plan the surfacelet transform for `dims`-sized volumes.

Planning scans every subband mask once to find its frequency support, so it
costs one full-grid pass per subband.  Reuse the plan.
"""
function make_surfacelet3d(
        params::SurfaceletParams3D, dims::NTuple{3, Int}, ::Type{T} = Float64;
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    all(>(0), dims) || throw(ArgumentError("volume dimensions must be positive"))
    p = _convert_sparams(T, params)
    wfreq = ntuple(d -> _folded_frequencies(T, dims[d]), 3)
    signed = ntuple(d -> _signed_indices(dims[d]), 3)
    negidx = ntuple(d -> _negation_indices(dims[d]), 3)
    radii = T[T(π) / 2^j for j in 0:p.J]

    FFTW.set_num_threads(_fftw_threads(threading))
    buf = zeros(Complex{T}, dims)
    fwd = plan_fft(buf; flags = flags)
    spec = Array{Complex{T}, 3}(undef, dims)
    work = Array{Complex{T}, 3}(undef, dims)

    plan = Surfacelet3DPlan{T, typeof(fwd)}(
        dims, p, wfreq, signed, negidx, radii,
        [SubbandBox[] for _ in 1:p.J], _full_box(dims), fwd, spec, work
    )
    # Support boxes, found by scanning each mask once.
    mask = Array{T, 3}(undef, dims)
    for j in 1:p.J
        for k in 1:surfacelet3d_nsubbands(p, j)
            _fill_surfacelet_mask!(mask, plan, j, k)
            push!(plan.boxes[j], _support_box(mask, plan))
        end
    end
    _fill_coarse_mask!(mask, plan)
    return Surfacelet3DPlan{T, typeof(fwd)}(
        dims, p, wfreq, signed, negidx, radii, plan.boxes,
        _support_box(mask, plan), fwd, spec, work
    )
end

make_surfacelet3d(params::SurfaceletParams3D, dims::Vararg{Int, 3}; kwargs...) =
    make_surfacelet3d(params, dims; kwargs...)

_convert_sparams(::Type{T}, p::SurfaceletParams3D{T}) where {T} = p
_convert_sparams(::Type{T}, p::SurfaceletParams3D) where {T <: AbstractFloat} =
    SurfaceletParams3D{T}(
    p.J, p.L_array, p.poly,
    T(p.transition), T(p.radial_transition), T(p.angular_transition)
)

# Signed frequency index of each DFT bin: 0, 1, …, ⌊n/2⌋, then negative.
_signed_indices(n::Int) = Int[k <= n ÷ 2 ? k : k - n for k in 0:(n - 1)]

# Index of the bin holding -ω, per axis.  This is an involution, and it is *not*
# always the sign flip of the signed index: at the Nyquist bin of an even-length
# axis, π ≡ -π maps to itself.
_negation_indices(n::Int) = Int[mod(-(k - 1), n) + 1 for k in 1:n]

_full_box(dims::NTuple{3, Int}) =
    SubbandBox(dims, ntuple(d -> collect(1:dims[d]), 3))

"""
    Surfacelet3DCoefficients{T}

Output of [`surfacelet3d_forward`](@ref).  Unlike the NSCT containers the arrays
here have *different* sizes: each subband is stored at its own decimated
resolution, which is the whole point.
"""
struct Surfacelet3DCoefficients{T <: AbstractFloat}
    dims::NTuple{3, Int}     # size of the volume this came from
    coarse::Array{T, 3}
    subbands::Vector{Vector{Array{T, 3}}}
end

Base.eltype(::Surfacelet3DCoefficients{T}) where {T} = T

function Base.show(io::IO, c::Surfacelet3DCoefficients)
    ns = sum(length, c.subbands; init = 0)
    n = length(c.coarse) + sum(sum(length, s; init = 0) for s in c.subbands; init = 0)
    return print(
        io, "Surfacelet3DCoefficients(dims=$(c.dims), J=$(length(c.subbands)), ",
        "subbands=$(ns), coefficients=$(n))"
    )
end

"""
    surfacelet3d_redundancy(plan::Surfacelet3DPlan) -> Float64

Total stored coefficients divided by the number of input voxels.  This is
measured from the plan's actual support boxes, so it reflects the transition
widths that were chosen rather than an idealised figure.
"""
function surfacelet3d_redundancy(plan::Surfacelet3DPlan)
    total = prod(plan.coarse_box.size)
    for scale in plan.boxes, box in scale
        total += prod(box.size)
    end
    return total / prod(plan.dims)
end

# ── Mask construction ────────────────────────────────────────────────────────
#
# Every factor below is exactly 0 or 1 outside a transition band, which is what
# makes the support boxes small.  `halfband_split` on a clamped argument gives
# that: `f₀(-1) = 0` and `f₀(1) = 1` exactly, and `f₀(x) + f₀(-x) = 1` makes the
# factors telescope to a partition of unity.

# Soft step: 0 for u ≤ -1, 1 for u ≥ 1, monotone in between.
@inline _step(u::T, poly::Int) where {T} = halfband_split(clamp(u, -one(T), one(T)), poly)

# Radial (cube-shell) weight of scale `j`, from r = ‖ω‖_∞.  Scale 1 is finest.
# The shells telescope: the sum over j = 1..J plus the coarse weight is 1.
@inline function _radial_weight(r::T, plan::Surfacelet3DPlan{T}, j::Int) where {T}
    p = plan.params
    inner = _radial_edge(r, plan, j)          # 1 below the outer shell edge
    outer = j == 1 ? one(T) : _radial_edge(r, plan, j - 1)
    return outer - inner
end

# 1 well *below* the boundary between scales j and j+1, 0 well above it: a soft
# low-pass indicator of that shell edge, which is what makes the differences in
# `_radial_weight` telescope to 1.
@inline function _radial_edge(r::T, plan::Surfacelet3DPlan{T}, j::Int) where {T}
    b = plan.radii[j + 1]
    w = plan.params.radial_transition * b
    return one(T) - _step((r - b) / w, plan.params.poly)
end

@inline _coarse_weight(r::T, plan::Surfacelet3DPlan{T}) where {T} =
    _radial_edge(r, plan, plan.params.J)

# Unnormalised hourglass weight of axis `i`: 1 where |ω_i| dominates both other
# axes by more than the transition width, 0 where either other axis dominates by
# that much.  Normalising by the sum over i gives the partition of unity, and
# the sum is bounded away from zero because whichever axis is largest
# contributes at least `f₀(0)²`.
@inline function _hourglass_weight(a::NTuple{3, T}, i::Int, delta::T, poly::Int) where {T}
    w = one(T)
    for j in 1:3
        j == i && continue
        w *= _step((a[i] - a[j]) / delta, poly)
    end
    return w
end

# Wedge weight along one transverse ratio: which of the `nw` equal slabs of
# [-1, 1] the ratio `s` falls in, softened.  Outermost edges are hard so the
# slabs telescope to 1 for any `s`, including |s| > 1.
@inline function _wedge_weight(s::T, idx::Int, nw::Int, transition::T, poly::Int) where {T}
    nw == 1 && return one(T)
    delta = T(2) / nw
    width = transition * delta
    lo = idx == 1 ? one(T) : _wedge_edge(s, idx - 1, delta, width, poly)
    hi = idx == nw ? zero(T) : _wedge_edge(s, idx, delta, width, poly)
    return lo - hi
end

@inline function _wedge_edge(s::T, k::Int, delta::T, width::T, poly::Int) where {T}
    return _step((s - (-one(T) + k * delta)) / width, poly)
end

# W = radial_j · normalised hourglass · wedge_p · wedge_q, then V = √W.
function _fill_surfacelet_mask!(
        out::AbstractArray{T, 3}, plan::Surfacelet3DPlan{T}, j::Int, k::Int
    ) where {T}
    p = plan.params
    nw = 1 << p.L_array[j]
    na, nb = _wedge_counts(p.L_array[j])
    hg = (k - 1) ÷ nw + 1
    rest = (k - 1) % nw
    ip = rest ÷ nb + 1
    iq = rest % nb + 1
    ax, bx = _transverse_axes(hg)
    w1, w2, w3 = plan.wfreq
    s1, s2, s3 = plan.signed
    g1, g2, g3 = plan.negidx
    delta = p.angular_transition
    @inbounds for k3 in eachindex(w3), k2 in eachindex(w2)
        for k1 in eachindex(w1)
            # Evaluate at the canonical representative of {ω, -ω}.  The wedge
            # ratios are sign-sensitive, and on the DFT grid ω → -ω is *not*
            # simply a sign flip of the folded frequency: at the Nyquist bin of
            # an even axis, π ≡ -π maps to itself, so `ω₂/ω₁` there would come
            # out with opposite signs at two bins that are each other's
            # negation.  That asymmetry makes the masked spectrum non-Hermitian
            # and the subband complex — and since the subbands are stored real,
            # it silently destroys perfect reconstruction.  Picking one member
            # of each ± pair by a fixed rule removes the ambiguity, evaluates
            # each direction once, and keeps the mask exactly even.
            q = (s1[k1], s2[k2], s3[k3])
            qn = (s1[g1[k1]], s2[g2[k2]], s3[g3[k3]])
            qq = qn > q ? qn : q
            w = (
                2 * T(π) * qq[1] / plan.dims[1],
                2 * T(π) * qq[2] / plan.dims[2],
                2 * T(π) * qq[3] / plan.dims[3],
            )
            a = (abs(w[1]), abs(w[2]), abs(w[3]))
            r = max(a[1], a[2], a[3])
            rad = _radial_weight(r, plan, j)
            if rad <= 0
                out[k1, k2, k3] = zero(T)
                continue
            end
            hw = _hourglass_weight(a, hg, delta, p.poly)
            if hw <= 0
                out[k1, k2, k3] = zero(T)
                continue
            end
            norm = _hourglass_weight(a, 1, delta, p.poly) +
                _hourglass_weight(a, 2, delta, p.poly) +
                _hourglass_weight(a, 3, delta, p.poly)
            # Signed ratios: the wedge slabs span [-1, 1], so using |ω_a|/|ω_i|
            # here would fold the two halves together and starve half the
            # wedges.
            ref = w[hg]
            s = ref == 0 ? zero(T) : w[ax] / ref
            t = ref == 0 ? zero(T) : w[bx] / ref
            wv = _wedge_weight(s, ip, na, p.transition, p.poly) *
                _wedge_weight(t, iq, nb, p.transition, p.poly)
            out[k1, k2, k3] = sqrt(rad * (hw / norm) * wv)
        end
    end
    return out
end

function _fill_coarse_mask!(out::AbstractArray{T, 3}, plan::Surfacelet3DPlan{T}) where {T}
    w1, w2, w3 = plan.wfreq
    @inbounds for k3 in eachindex(w3), k2 in eachindex(w2)
        for k1 in eachindex(w1)
            r = max(abs(w1[k1]), abs(w2[k2]), abs(w3[k3]))
            out[k1, k2, k3] = sqrt(_coarse_weight(r, plan))
        end
    end
    return out
end

_wedge_counts(L::Int) = (1 << cld(L, 2), 1 << fld(L, 2))

# ── Support boxes ────────────────────────────────────────────────────────────

# Smallest box symmetric about the origin containing the mask's support, as
# per-axis index maps.  Symmetry matters: the masks are even in ω, so a
# symmetric box keeps the restricted spectrum Hermitian and the subband real.
function _support_box(mask::AbstractArray{T, 3}, plan::Surfacelet3DPlan{T}) where {T}
    dims = plan.dims
    reach = zeros(Int, 3)
    found = false
    @inbounds for k3 in axes(mask, 3), k2 in axes(mask, 2), k1 in axes(mask, 1)
        mask[k1, k2, k3] > 0 || continue
        found = true
        q = (plan.signed[1][k1], plan.signed[2][k2], plan.signed[3][k3])
        for d in 1:3
            reach[d] = max(reach[d], abs(q[d]))
        end
    end
    found || return SubbandBox((1, 1, 1), ntuple(d -> zeros(Int, dims[d]), 3))
    sizes = ntuple(d -> min(2 * reach[d] + 1, dims[d]), 3)
    maps = ntuple(d -> _axis_map(plan.signed[d], reach[d], sizes[d], dims[d]), 3)
    return SubbandBox(sizes, maps)
end

# Full-grid index → box index (FFT order on the box), 0 when outside the box.
function _axis_map(signed::Vector{Int}, reach::Int, b::Int, n::Int)
    m = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        if b == n
            m[k] = k                              # no decimation on this axis
        else
            q = signed[k]
            m[k] = abs(q) <= reach ? (q >= 0 ? q + 1 : q + b + 1) : 0
        end
    end
    return m
end

# ── Transform ────────────────────────────────────────────────────────────────

"""
    surfacelet3d_forward(volume, params::SurfaceletParams3D; plan=nothing) -> Surfacelet3DCoefficients

Decimated 3-D surfacelet transform.  Each subband is returned at its own reduced
resolution, so the total coefficient count is a small multiple of the input —
see [`surfacelet3d_redundancy`](@ref).

Not shift-invariant: that is what the decimation costs, and it is why
[`nsct3d_forward`](@ref) still exists.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(7), 32, 32, 32);

julia> p = SurfaceletParams3D(J = 2, L_array = [1, 1]);

julia> c = surfacelet3d_forward(x, p);

julia> maximum(abs, surfacelet3d_inverse(c, p) .- x) < 1e-10
true
```
"""
function surfacelet3d_forward(
        volume::AbstractArray{<:Real, 3}, params::SurfaceletParams3D;
        plan::Union{Surfacelet3DPlan, Nothing} = nothing
    )
    T = _filter_eltype(_data_eltype(volume))
    pl = _surfacelet3d_plan(plan, params, size(volume), T)
    p = pl.params

    copyto!(pl.spec, volume)
    mul!(pl.work, pl.fwd, pl.spec)
    copyto!(pl.spec, pl.work)                      # keep the spectrum

    mask = Array{T, 3}(undef, pl.dims)
    subbands = Vector{Vector{Array{T, 3}}}(undef, p.J)
    for j in 1:p.J
        subbands[j] = Vector{Array{T, 3}}(undef, surfacelet3d_nsubbands(p, j))
        for k in eachindex(subbands[j])
            _fill_surfacelet_mask!(mask, pl, j, k)
            subbands[j][k] = _extract_subband(pl, mask, pl.boxes[j][k])
        end
    end
    _fill_coarse_mask!(mask, pl)
    coarse = _extract_subband(pl, mask, pl.coarse_box)
    return Surfacelet3DCoefficients(pl.dims, coarse, subbands)
end

"""
    surfacelet3d_inverse(coeffs::Surfacelet3DCoefficients, params::SurfaceletParams3D; plan=nothing) -> volume

Invert [`surfacelet3d_forward`](@ref).  Exact: the masks satisfy `Σ V² = 1` and
each vanishes outside its box, so restriction loses nothing.
"""
function surfacelet3d_inverse(
        coeffs::Surfacelet3DCoefficients{T}, params::SurfaceletParams3D;
        plan::Union{Surfacelet3DPlan, Nothing} = nothing
    ) where {T}
    pl = _surfacelet3d_plan(plan, params, coeffs.dims, T)
    p = pl.params
    length(coeffs.subbands) == p.J ||
        throw(ArgumentError("coefficients were produced for a different number of scales"))

    fill!(pl.spec, zero(Complex{T}))
    mask = Array{T, 3}(undef, pl.dims)
    for j in 1:p.J
        length(coeffs.subbands[j]) == surfacelet3d_nsubbands(p, j) ||
            throw(DimensionMismatch("scale $(j) has the wrong number of subbands"))
        for k in eachindex(coeffs.subbands[j])
            _fill_surfacelet_mask!(mask, pl, j, k)
            _accumulate_subband!(pl, coeffs.subbands[j][k], mask, pl.boxes[j][k])
        end
    end
    _fill_coarse_mask!(mask, pl)
    _accumulate_subband!(pl, coeffs.coarse, mask, pl.coarse_box)

    ifft!(pl.spec)
    out = Array{T, 3}(undef, pl.dims)
    @inbounds @simd for i in eachindex(out, pl.spec)
        out[i] = real(pl.spec[i])
    end
    return out
end

function _surfacelet3d_plan(
        plan::Union{Surfacelet3DPlan, Nothing}, params::SurfaceletParams3D,
        dims::NTuple{3, Int}, ::Type{T}
    ) where {T}
    plan === nothing && return make_surfacelet3d(params, dims, T)
    plan.dims == dims || throw(DimensionMismatch("plan size does not match the volume"))
    return plan
end

# Restrict the masked spectrum to the subband's box and transform back at box
# size.  The mask is even in ω and the box is symmetric, so the restricted data
# is Hermitian on the box grid and the result is real up to rounding.
function _extract_subband(
        pl::Surfacelet3DPlan{T}, mask::AbstractArray{T, 3}, box::SubbandBox
    ) where {T}
    buf = zeros(Complex{T}, box.size)
    m1, m2, m3 = box.maps
    spec = pl.spec
    @inbounds for k3 in axes(spec, 3)
        b3 = m3[k3]
        b3 == 0 && continue
        for k2 in axes(spec, 2)
            b2 = m2[k2]
            b2 == 0 && continue
            for k1 in axes(spec, 1)
                b1 = m1[k1]
                b1 == 0 && continue
                buf[b1, b2, b3] += spec[k1, k2, k3] * mask[k1, k2, k3]
            end
        end
    end
    ifft!(buf)
    out = Array{T, 3}(undef, box.size)
    @inbounds @simd for i in eachindex(out, buf)
        out[i] = real(buf[i])
    end
    return out
end

# Transform the subband back up to its box spectrum and accumulate it, masked,
# into the full spectrum.
function _accumulate_subband!(
        pl::Surfacelet3DPlan{T}, band::AbstractArray{T, 3},
        mask::AbstractArray{T, 3}, box::SubbandBox
    ) where {T}
    size(band) == box.size ||
        throw(DimensionMismatch("subband is $(size(band)), expected $(box.size)"))
    buf = Array{Complex{T}, 3}(undef, box.size)
    copyto!(buf, band)
    fft!(buf)
    m1, m2, m3 = box.maps
    spec = pl.spec
    @inbounds for k3 in axes(spec, 3)
        b3 = m3[k3]
        b3 == 0 && continue
        for k2 in axes(spec, 2)
            b2 = m2[k2]
            b2 == 0 && continue
            for k1 in axes(spec, 1)
                b1 = m1[k1]
                b1 == 0 && continue
                spec[k1, k2, k3] += buf[b1, b2, b3] * mask[k1, k2, k3]
            end
        end
    end
    return spec
end
