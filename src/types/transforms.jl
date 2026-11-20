# The transform objects CT, NSCT, TICT and STICT: every parameter of a
# decomposition (pyramid depth, per-level DFB depth, filters), with their
# validation, value equality and precision conversion, and the
# `parabolic_levels` helper that builds a parabolic-scaling `L_array`.

# Supertype of the four transforms, which hold the same parameters.
abstract type _ContourletTransform{T <: AbstractFloat} <: AbstractMultiscaleTransform end

# The fields of a transform at filter precision `T`, validated.
function _contourlet_fields(::Type{T}, J, L_array, lp_filters, dfb_filters) where {T <: AbstractFloat}
    J >= 1 || throw(ArgumentError("J must be ≥ 1, got $(J)"))
    length(L_array) == J || throw(ArgumentError("length(L_array) must equal J = $(J), got $(length(L_array))"))
    all(>=(0), L_array) || throw(ArgumentError("L_array entries must be ≥ 0, got $(L_array)"))
    lp = FilterPair{T}(T.(lp_filters.h), T.(lp_filters.g))
    return Int(J), collect(Int, L_array), lp, _convert_qfp(T, dfb_filters)
end

for (name, what) in (
        (:CT, "the discrete contourlet transform"),
        (:NSCT, "the nonsubsampled contourlet transform"),
        (:TICT, "the translation-invariant contourlet transform"),
        (:STICT, "the semi-translation-invariant contourlet transform"),
    )
    @eval begin
        struct $name{T} <: _ContourletTransform{T}
            J::Int
            L_array::Vector{Int}
            lp_filters::FilterPair{T}
            dfb_filters::QuincunxFilterPair{T}
            function $name{T}(J, L_array, lp_filters, dfb_filters) where {T <: AbstractFloat}
                J, L_array, lp_filters, dfb_filters = _contourlet_fields(T, J, L_array, lp_filters, dfb_filters)
                return new{T}(J, L_array, lp_filters, dfb_filters)
            end
        end

        function $name(; J::Integer, L_array::AbstractVector{<:Integer}, lp_filters::FilterPair = CDF97, dfb_filters::QuincunxFilterPair = Q2345)
            T = promote_type(eltype(lp_filters), eltype(dfb_filters))
            return $name{T}(J, L_array, lp_filters, dfb_filters)
        end

        _family(::$name) = $name
    end
end

const _TRANSFORM_KEYWORDS = """
- `J`: the number of pyramid levels (≥ 1);
- `L_array`: the directional depth of each level, `J` entries ≥ 0, finest level
  first (see [`parabolic_levels`](@ref)); level `j` has `2^L_array[j]`
  directions;
- `lp_filters`: the pyramid's biorthogonal filter pair ([`CDF97`](@ref) by
  default);
- `dfb_filters`: the ladder pair of the directional filter banks
  ([`Q2345`](@ref) by default).

The filters are stored at the precision of the wider of the two pairs; a
transform runs at the precision of its data (`Float32` data uses `Float32`
filters), whatever the precision it holds.  Transforms with equal parameters
are `==` and hash alike."""

"""
    CT(; J, L_array, lp_filters = CDF97, dfb_filters = Q2345)

The discrete contourlet transform of Do & Vetterli (2005): a `J`-level Laplacian
pyramid whose bandpass image at level `j` is split by a decimated directional
filter bank of depth `L_array[j]`.  It is nearly critically sampled (about
4/3 coefficients per pixel); its subbands have different sizes, so the
coefficients are a [`SubbandVector`](@ref).

$(_TRANSFORM_KEYWORDS)

Subbands ([`subband_layout`](@ref)): the lowpass first, as
`(scale = J + 1, direction = 0)`, then for `scale = 1` (the finest), 2, …, `J`
its directional subbands `(scale = j, direction = k)`, `k = 1, …, 2^L_array[j]`.
The transform runs serially; `threaded` is accepted and ignored.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 64, 64);

julia> t = CT(J = 2, L_array = [1, 2])
CT(J = 2, L_array = [1, 2])

julia> c = analyze(t, x);

julia> nsubbands(t, size(x)), size(subband(t, c, 1))
(7, (16, 16))

julia> maximum(abs, synthesize(t, c) .- x) < 1e-12
true
```

# References
1. M. N. Do and M. Vetterli, "The Contourlet Transform: An Efficient Directional
   Multiresolution Image Representation", *IEEE Transactions on Image Processing*,
   14(12):2091–2106, 2005.
   [doi:10.1109/TIP.2005.859376](https://doi.org/10.1109/TIP.2005.859376)
2. P. J. Burt and E. H. Adelson, "The Laplacian Pyramid as a Compact Image Code",
   *IEEE Transactions on Communications*, 31(4):532–540, 1983.
   [doi:10.1109/TCOM.1983.1095851](https://doi.org/10.1109/TCOM.1983.1095851)
3. R. H. Bamberger and M. J. T. Smith, "A Filter Bank for the Directional
   Decomposition of Images: Theory and Design", *IEEE Transactions on Signal
   Processing*, 40(4):882–893, 1992.
   [doi:10.1109/78.127960](https://doi.org/10.1109/78.127960)
"""
CT

"""
    NSCT(; J, L_array, lp_filters = CDF97, dfb_filters = Q2345)

The nonsubsampled contourlet transform of da Cunha, Zhou & Do (2006): the
contourlet transform without its samplers (a nonsubsampled pyramid and
nonsubsampled directional filter banks), so every subband has the image's size
and the transform commutes with circular shifts of the image.  The coefficients
are one array of size `(size(x)..., n)`, subband `i` being `c[:, :, i]`.

$(_TRANSFORM_KEYWORDS)

Subbands ([`subband_layout`](@ref)): the lowpass first, as
`(scale = J + 1, direction = 0)`, then for `scale = 1` (the finest), 2, …, `J`
its directional subbands `(scale = j, direction = k)`, `k = 1, …, 2^L_array[j]`.

With a workspace and `threaded = true` (the default `threaded = nothing` means
`true` for complex data, `false` for real data) the independent subband FFTs of
a host workspace run in one task per Julia thread.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(2), 32, 32);

julia> t = NSCT(J = 2, L_array = [1, 2]);

julia> c = analyze(t, x);

julia> size(c)
(32, 32, 7)

julia> maximum(abs, synthesize(t, c) .- x) < 1e-12
true
```

# References
1. A. L. da Cunha, J. Zhou, and M. N. Do, "The Nonsubsampled Contourlet Transform:
   Theory, Design, and Applications", *IEEE Transactions on Image Processing*,
   15(10):3089–3101, 2006.
   [doi:10.1109/TIP.2006.877507](https://doi.org/10.1109/TIP.2006.877507)
"""
NSCT

"""
    TICT(; J, L_array, lp_filters = CDF97, dfb_filters = Q2345)

The translation-invariant contourlet transform of Eslami & Radha (2006).  A
`J`-level translation-invariant Laplacian pyramid gives, at every level `i`,
four full-size highpass channels; each is split by a translation-invariant
directional filter bank of depth `L_array[i]`: the full tree for channels 1
and 4 (`2^L_array[i]` subbands), a half tree for channels 2 and 3
(`2^(L_array[i] - 1) + 1` subbands).  Every subband has the image's size, and
the transform commutes with circular shifts of the image.  The coefficients are
one array of size `(size(x)..., n)`, subband `i` being `c[:, :, i]`.

The redundancy is `3 Σᵢ 2^L_array[i] + 2J + 1` when every `L_array[i] ≥ 1` (a
level with `L_array[i] = 0` contributes its 4 channels).  [`STICT`](@ref) trades
the directional stage's translation invariance for a redundancy of `4J + 1`.
`lp_filters` must be biorthogonal (as [`CDF97`](@ref) is).

$(_TRANSFORM_KEYWORDS)

Subbands ([`subband_layout`](@ref)): the lowpass first, as
`(level = J + 1, channel = 0, direction = 0)`, then for `level = 1` (the
finest), 2, …, `J`, for `channel = 1, …, 4`, the directional subbands
`(level = i, channel = c, direction = k)`.

Threading as for [`NSCT`](@ref).

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 32, 32);

julia> t = TICT(J = 2, L_array = [2, 1]);

julia> c = analyze(t, x);

julia> size(c, 3) == 1 + (4 + 3 + 3 + 4) + (2 + 2 + 2 + 2)
true

julia> maximum(abs, synthesize(t, c) .- x) < 1e-12
true
```

# References
1. R. Eslami and H. Radha, "Translation-Invariant Contourlet Transform and Its
   Application to Image Denoising", *IEEE Transactions on Image Processing*,
   15(11):3362–3374, 2006.
   [doi:10.1109/TIP.2006.881992](https://doi.org/10.1109/TIP.2006.881992)
"""
TICT

"""
    STICT(; J, L_array, lp_filters = CDF97, dfb_filters = Q2345)

The semi-translation-invariant contourlet transform of Eslami & Radha (2006):
the translation-invariant Laplacian pyramid of [`TICT`](@ref) followed by
critically sampled directional filter banks (full tree on channels 1 and 4, half
trees on 2 and 3).  The pyramid commutes with circular shifts; the directional
subbands are those of the decimated DFB.  Redundancy `4J + 1`; the subbands
have different sizes, so the coefficients are a [`SubbandVector`](@ref).

Both image dimensions must be divisible by `2^max(L - 1, 1)` for the deepest
directional level `L`.

$(_TRANSFORM_KEYWORDS)

Subbands ([`subband_layout`](@ref)) are ordered as for [`TICT`](@ref).  The
transform runs serially; `threaded` is accepted and ignored.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 32, 32);

julia> t = STICT(J = 2, L_array = [3, 2]);

julia> c = analyze(t, x);

julia> length(c) == 9 * length(x)
true

julia> maximum(abs, synthesize(t, c) .- x) < 1e-12
true
```

# References
1. R. Eslami and H. Radha, "Translation-Invariant Contourlet Transform and Its
   Application to Image Denoising", *IEEE Transactions on Image Processing*,
   15(11):3362–3374, 2006.
   [doi:10.1109/TIP.2006.881992](https://doi.org/10.1109/TIP.2006.881992)
"""
STICT

function Base.:(==)(a::_ContourletTransform, b::_ContourletTransform)
    return _family(a) === _family(b) && a.J == b.J && a.L_array == b.L_array &&
        a.lp_filters == b.lp_filters && a.dfb_filters == b.dfb_filters
end

function Base.hash(t::_ContourletTransform, h::UInt)
    h = hash(nameof(_family(t)), h)
    return hash(t.dfb_filters, hash(t.lp_filters, hash(t.L_array, hash(t.J, h))))
end

function Base.show(io::IO, t::_ContourletTransform{T}) where {T}
    print(io, nameof(_family(t)), "(J = ", t.J, ", L_array = ", t.L_array)
    t.lp_filters == _convert_pair(T, CDF97) || print(io, ", lp_filters = ", t.lp_filters)
    t.dfb_filters == _convert_qfp(T, Q2345) || print(io, ", dfb_filters = ", t.dfb_filters)
    return print(io, ")")
end

_convert_pair(::Type{T}, fp::FilterPair{T}) where {T <: AbstractFloat} = fp
_convert_pair(::Type{T}, fp::FilterPair) where {T <: AbstractFloat} = FilterPair{T}(T.(fp.h), T.(fp.g))

# The transform `t` at filter precision `Tf` (`t` itself when it already is).
_convert_transform(::Type{Tf}, t::_ContourletTransform{Tf}) where {Tf <: AbstractFloat} = t
_convert_transform(::Type{Tf}, t::_ContourletTransform) where {Tf <: AbstractFloat} =
    _family(t){Tf}(t.J, t.L_array, t.lp_filters, t.dfb_filters)

# Images only: the contourlet transforms are planar.
_check_image(dims::Dims{2}) = dims
_check_image(dims::Dims) = throw(ArgumentError("the contourlet transforms take images (two axes), got data of size $(dims)"))

"""
    parabolic_levels(J, l_j0=1) -> Vector{Int}

Compute the per-scale DFB depth array following the parabolic scaling law

    l_j = l_j0 + ⌊(j0 - j) / 2⌋,    j0 = J,

so that the number of directions doubles every other scale — the discrete
counterpart of parabolic scaling, width ∝ length², in Do & Vetterli (2005).  Index `j` matches
the scale ordering of the transforms' `L_array`: `j = 1`
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

# References
1. M. N. Do and M. Vetterli, "The Contourlet Transform: An Efficient Directional
   Multiresolution Image Representation", *IEEE Transactions on Image Processing*,
   14(12):2091–2106, 2005.
   [doi:10.1109/TIP.2005.859376](https://doi.org/10.1109/TIP.2005.859376)
"""
function parabolic_levels(J::Int, l_j0::Int = 1)::Vector{Int}
    j0 = J
    return [max(0, l_j0 + (j0 - j) ÷ 2) for j in 1:J]
end
