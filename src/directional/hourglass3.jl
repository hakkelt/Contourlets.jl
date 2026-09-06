# 3-D nonsubsampled hourglass filter bank.
#
# Reference: Y. M. Lu and M. N. Do, "A Mapping-Based Design for Nonsubsampled
# Hourglass Filter Banks in Arbitrary Dimensions", IEEE TSP 56(4), 2008.
#
# The bank splits the spectrum into three hourglass-shaped subbands, each aligned
# with one frequency axis:
#
#     P_i = {ω : |ω_i| ≥ |ω_j| for all j ≠ i}
#
# i.e. subband `i` keeps the frequencies closer to axis `i` than to any other.
# This is the directional stage that replaces the 2-D quincunx/shear filter bank
# — neither the quincunx lattice nor the Q2345 ladder carries over to 3-D, so
# none of `directional/quincunx_fb.jl` is reused here.
#
# Structure (Lu & Do, Fig. 3a): a cascade of two two-channel mapped filter banks.
#
#     spectrum ──┬─ P₁                                    (3-D mapping kernel)
#                └─ P₂ ∪ P₃ ──┬─ P₂                       (2-D kernel on ω₂,ω₃)
#                             └─ P₃
#
# The second kernel needs only the two remaining frequency variables — Lu & Do's
# Proposition 1 shows the optimal passband support for a node with index set `J`
# and child set `K` depends on `|J|` variables only.  In 3-D that makes the
# second stage a 2-D design, which is why `m2` below is a matrix.
#
# Implementation: frequency domain, on the DFT grid.  Lu & Do deliberately chose
# FIR spatial filters because their FFT-domain predecessor needed too much memory
# for streaming video.  That trade-off inverts here — this package's 3-D stages
# are nonsubsampled with a periodic boundary, so the FFT is already the default
# backend (see `conv3d_fft.jl`), and evaluating the mapping kernel directly on
# the grid is *exact*: no FIR truncation of a 37³ filter, no convolution.
#
# Perfect reconstruction is structural.  The 1-D polynomials satisfy
# `f₀(x) + f₁(x) = 1` identically, so the three masks sum to
# `f₀(M₁) + f₁(M₁)·(f₀(M₂) + f₁(M₂)) = f₀(M₁) + f₁(M₁) = 1` at every frequency,
# and reconstruction is a plain sum of the subbands.

"""
    HourglassFB3{T}

Precomputed 3-D nonsubsampled hourglass filter bank for one volume size and real
precision `T`.  Build with [`make_hourglass3`](@ref).

Holds the two mapping-kernel frequency responses (`m1` on the 3-D rfft grid,
`m2` on the 2-D `(ω₂, ω₃)` grid) rather than the three subband masks, which are
derived pointwise on the fly — a third of the memory, and the polynomial
evaluation is negligible next to the transforms.
"""
mutable struct HourglassFB3{T <: AbstractFloat, Pf, Pi}
    const dims::NTuple{3, Int}
    const order::Int              # Bernstein order N of the mapping kernels
    const poly::Int               # order m of the 1-D halfband split polynomial
    const m1::Array{T, 3}         # 3-D mapping response, rfft grid
    const m2::Matrix{T}           # 2-D mapping response on (ω₂, ω₃)
    const fwd::Pf
    const inv::Pi
    const spec::Array{Complex{T}, 3}   # spectrum of the input, preserved across subbands
    const work::Array{Complex{T}, 3}   # scratch consumed by the inverse plan
    const rbuf::Array{T, 3}
    # Second spectrum and output buffer, for the imaginary half of a complex
    # volume.  Allocated on first complex use so real-only workloads never pay
    # for them.  Two spectra cost about two real volumes — far less than holding
    # every subband's real part, which is what a two-pass scheme would need.
    spec2::Union{Nothing, Array{Complex{T}, 3}}
    rbuf2::Union{Nothing, Array{T, 3}}
end

"""
    make_hourglass3(dims::NTuple{3,Int}, ::Type{T}=Float64;
                    order::Int=12, poly::Int=1,
                    threading=Auto(), flags=FFTW.MEASURE) -> HourglassFB3

Build the 3-D hourglass filter bank for `dims`-sized volumes.

`order` is the Bernstein order `N` of the mapping kernels: larger `N` sharpens
the transition between hourglass subbands (Lu & Do use `N = 12`, which in their
FIR realisation corresponds to filters of support up to 61³ — here it costs
nothing but a slightly longer design step).  `poly` is the order `m` of the 1-D
halfband split polynomial; larger `m` flattens the response near the passband
centres.  Neither affects perfect reconstruction, which is exact for any choice.

A complex element type is accepted and produces a plan of the corresponding real
precision; complex volumes are filtered as two real passes, keeping
`T(x + i·y) == T(x) + i·T(y)` exact.
"""
function make_hourglass3(
        dims::NTuple{3, Int}, ::Type{T} = Float64;
        order::Int = 12, poly::Int = 1,
        threading::ThreadingPolicy = Auto(),
        flags::Integer = FFTW.MEASURE
    ) where {T <: AbstractFloat}
    n1, n2, n3 = dims
    all(>(0), dims) || throw(ArgumentError("volume dimensions must be positive"))
    order >= 1 || throw(ArgumentError("Bernstein order must be ≥ 1"))

    # Stage 1 — split P₁ out of the whole spectrum with a 3-D kernel.  Axis 1 is
    # the rfft axis, so only its non-negative half of the grid is needed.
    b1 = bernstein_table(order, dft_frequencies(T, n1; half = true))
    b2 = bernstein_table(order, dft_frequencies(T, n2))
    b3 = bernstein_table(order, dft_frequencies(T, n3))
    m1 = mapping_response(mapping_coefficients(order, 3, (1,)), (b1, b2, b3))

    # Stage 2 — split P₂ from P₃ with a 2-D kernel on the (ω₂, ω₃) plane.
    m2 = mapping_response(mapping_coefficients(order, 2, (1,)), (b2, b3))

    FFTW.set_num_threads(_fftw_threads(threading))
    rbuf = zeros(T, n1, n2, n3)
    fwd = plan_rfft(rbuf; flags = flags)
    spec = fwd * rbuf
    inv = plan_irfft(spec, n1; flags = flags)
    return HourglassFB3{T, typeof(fwd), typeof(inv)}(
        dims, order, poly, m1, m2, fwd, inv, spec, similar(spec), rbuf, nothing, nothing
    )
end

function make_hourglass3(
        dims::NTuple{3, Int}, ::Type{Complex{T}}; kwargs...
    ) where {T <: AbstractFloat}
    fb = make_hourglass3(dims, T; kwargs...)
    _complex_scratch!(fb)
    return fb
end

# Ensure the imaginary-half buffers exist; returns them.
function _complex_scratch!(fb::HourglassFB3{T}) where {T}
    s2 = fb.spec2
    if s2 === nothing
        s2 = similar(fb.spec)
        fb.spec2 = s2
    end
    r2 = fb.rbuf2
    if r2 === nothing
        r2 = Array{T, 3}(undef, fb.dims)
        fb.rbuf2 = r2
    end
    return s2, r2
end

# Load the real and imaginary parts of `volume` into the two spectra.
function _load_complex_spectra!(fb::HourglassFB3{T}, volume::AbstractArray{Complex{T}, 3}) where {T}
    spec2, _ = _complex_scratch!(fb)
    @inbounds @simd for i in eachindex(fb.rbuf, volume)
        fb.rbuf[i] = real(volume[i])
    end
    mul!(fb.spec, fb.fwd, fb.rbuf)
    @inbounds @simd for i in eachindex(fb.rbuf, volume)
        fb.rbuf[i] = imag(volume[i])
    end
    mul!(spec2, fb.fwd, fb.rbuf)
    return fb
end

make_hourglass3(n1::Int, n2::Int, n3::Int, ::Type{Td} = Float64; kwargs...) where {Td} =
    make_hourglass3((n1, n2, n3), Td; kwargs...)

"""
    hourglass3_mask(fb::HourglassFB3, i::Int) -> Array{T,3}

The frequency-domain mask of hourglass subband `i` (1, 2 or 3) on the rfft grid.
Materialising it is only useful for inspection and tests — the transform derives
mask values on the fly.  The three masks sum to 1 at every frequency.
"""
function hourglass3_mask(fb::HourglassFB3{T}, i::Int) where {T}
    1 <= i <= 3 || throw(ArgumentError("hourglass subband index must be 1, 2 or 3"))
    out = similar(fb.m1)
    @inbounds for k in axes(out, 3), j in axes(out, 2)
        w2 = _stage2_weight(fb.m2[j, k], fb.poly, i)
        @simd for p in axes(out, 1)
            out[p, j, k] = _stage1_weight(fb.m1[p, j, k], fb.poly, i) * w2
        end
    end
    return out
end

# Stage-1 weight for subband `i`: f₀(M₁) for i == 1, f₁(M₁) otherwise.
@inline function _stage1_weight(m::T, poly::Int, i::Int) where {T}
    f0 = halfband_split(m, poly)
    return i == 1 ? f0 : one(T) - f0
end

# Stage-2 weight for subband `i`: 1 for i == 1 (stage 2 does not touch P₁),
# f₀(M₂) for i == 2 and f₁(M₂) for i == 3.
@inline function _stage2_weight(m::T, poly::Int, i::Int) where {T}
    i == 1 && return one(T)
    f0 = halfband_split(m, poly)
    return i == 2 ? f0 : one(T) - f0
end

"""
    hourglass3_decompose(volume, fb::HourglassFB3) -> (s₁, s₂, s₃)

Split `volume` into the three hourglass subbands.  Every subband keeps the full
size of the input — the bank is nonsubsampled, so the decomposition is exactly
equivariant under circular shifts and the redundancy is 3 (`N` in `N`-D, which
Lu & Do show is unavoidable: hourglass partitioning cannot be critically sampled
above 2-D).

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(2), 16, 16, 16);

julia> fb = make_hourglass3(size(x), Float64; order = 6);

julia> s = hourglass3_decompose(x, fb);

julia> maximum(abs, hourglass3_reconstruct(s, fb) .- x) < 1e-10
true
```
"""
function hourglass3_decompose(volume::AbstractArray{T, 3}, fb::HourglassFB3{T}) where {T <: AbstractFloat}
    out = ntuple(_ -> Array{T, 3}(undef, fb.dims), 3)
    return hourglass3_decompose!(out, volume, fb)
end

function hourglass3_decompose(
        volume::AbstractArray{Complex{T}, 3}, fb::HourglassFB3{T}
    ) where {T <: AbstractFloat}
    out = ntuple(_ -> Array{Complex{T}, 3}(undef, fb.dims), 3)
    return hourglass3_decompose!(out, volume, fb)
end

"""
    hourglass3_decompose!(dst, volume, fb) -> dst

In-place [`hourglass3_decompose`](@ref); `dst` is a 3-tuple of arrays matching
the plan size and the element type of `volume`.
"""
function hourglass3_decompose!(
        dst::NTuple{3, <:AbstractArray{T, 3}}, volume::AbstractArray{T, 3},
        fb::HourglassFB3{T}
    ) where {T <: AbstractFloat}
    size(volume) == fb.dims || throw(DimensionMismatch("volume size does not match the plan"))
    copyto!(fb.rbuf, volume)
    mul!(fb.spec, fb.fwd, fb.rbuf)
    for i in 1:3
        _apply_hourglass_mask!(fb, fb.spec, i)
        mul!(fb.rbuf, fb.inv, fb.work)
        copyto!(dst[i], fb.rbuf)
    end
    return dst
end

function hourglass3_decompose!(
        dst::NTuple{3, <:AbstractArray{Complex{T}, 3}}, volume::AbstractArray{Complex{T}, 3},
        fb::HourglassFB3{T}
    ) where {T <: AbstractFloat}
    size(volume) == fb.dims || throw(DimensionMismatch("volume size does not match the plan"))
    # Real and imaginary parts go through the identical real kernel — the same
    # construction `conv3d_fft.jl` uses, and for the same reason: it keeps
    # `T(x + i·y) == T(x) + i·T(y)` exact.  Both spectra are held at once so each
    # subband is finished before moving to the next.
    _load_complex_spectra!(fb, volume)
    spec2, rbuf2 = _complex_scratch!(fb)
    for i in 1:3
        _apply_hourglass_mask!(fb, fb.spec, i)
        mul!(fb.rbuf, fb.inv, fb.work)
        _apply_hourglass_mask!(fb, spec2, i)
        mul!(rbuf2, fb.inv, fb.work)
        @inbounds @simd for p in eachindex(dst[i], fb.rbuf)
            dst[i][p] = Complex{T}(fb.rbuf[p], rbuf2[p])
        end
    end
    return dst
end

# work .= spec .* mask_i, with the mask derived from the two mapping responses.
function _apply_hourglass_mask!(fb::HourglassFB3{T}, spec::AbstractArray, i::Int) where {T}
    work, m1, m2 = fb.work, fb.m1, fb.m2
    poly = fb.poly
    @inbounds for k in axes(work, 3), j in axes(work, 2)
        w2 = _stage2_weight(m2[j, k], poly, i)
        @simd for p in axes(work, 1)
            work[p, j, k] = spec[p, j, k] * (_stage1_weight(m1[p, j, k], poly, i) * w2)
        end
    end
    return work
end

"""
    hourglass3_reconstruct(subbands, fb::HourglassFB3) -> volume

Invert [`hourglass3_decompose`](@ref).  The three analysis masks form a partition
of unity, so the synthesis filters are all 1 and reconstruction is a plain sum —
perfect reconstruction holds exactly, for any `order`/`poly`, and is not an
approximation that degrades with filter sharpness.
"""
function hourglass3_reconstruct(subbands::NTuple{3, <:AbstractArray{Td, 3}}, fb::HourglassFB3) where {Td}
    out = similar(subbands[1])
    return hourglass3_reconstruct!(out, subbands, fb)
end

"""
    hourglass3_reconstruct!(volume, subbands, fb) -> volume

In-place [`hourglass3_reconstruct`](@ref).
"""
function hourglass3_reconstruct!(
        volume::AbstractArray{Td, 3}, subbands::NTuple{3, <:AbstractArray{Td, 3}},
        fb::HourglassFB3
    ) where {Td}
    size(volume) == fb.dims || throw(DimensionMismatch("volume size does not match the plan"))
    s1, s2, s3 = subbands
    @. volume = s1 + s2 + s3
    return volume
end
