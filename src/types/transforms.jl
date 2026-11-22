# The transform objects CT, NSCT, TICT and STICT: every parameter of a
# decomposition (number of scales, DFB depth per scale, filters), with their
# validation, value equality and precision conversion, and the
# `parabolic_levels` helper that builds parabolic-scaling `direction_levels`.

# Supertype of the four transforms, which hold the same parameters.
abstract type _ContourletTransform{T <: AbstractFloat} <: AbstractMultiscaleTransform end

# The fields of a transform at filter precision `T`, validated.
function _contourlet_fields(::Type{T}, scales, direction_levels, lp_filters, dfb_filters) where {T <: AbstractFloat}
    scales >= 1 || throw(ArgumentError("scales must be ≥ 1, got $(scales)"))
    length(direction_levels) == scales || throw(
        ArgumentError("length(direction_levels) must equal scales = $(scales), got $(length(direction_levels))")
    )
    all(>=(0), direction_levels) || throw(ArgumentError("direction_levels entries must be ≥ 0, got $(direction_levels)"))
    lp = FilterPair{T}(T.(lp_filters.h), T.(lp_filters.g))
    return Int(scales), ReadOnlyTaps{Int}(collect(Int, direction_levels)), lp, _convert_qfp(T, dfb_filters)
end

# The number of scales and the direction levels from the constructor keywords,
# either of which may be `nothing`: `scales` alone takes the parabolic-scaling
# levels, `direction_levels` alone has one scale per entry.
function _resolve_levels(scales, direction_levels)
    if direction_levels === nothing
        scales === nothing && throw(ArgumentError("give scales, direction_levels, or both"))
        return scales, parabolic_levels(scales)
    end
    scales === nothing && return length(direction_levels), direction_levels
    length(direction_levels) == scales || throw(
        ArgumentError("scales = $(scales) disagrees with direction_levels = $(direction_levels), which has $(length(direction_levels)) entries")
    )
    return scales, direction_levels
end

for (name, what) in (
        (:CT, "the discrete contourlet transform"),
        (:NSCT, "the nonsubsampled contourlet transform"),
        (:TICT, "the translation-invariant contourlet transform"),
        (:STICT, "the semi-translation-invariant contourlet transform"),
    )
    @eval begin
        struct $name{T} <: _ContourletTransform{T}
            scales::Int
            direction_levels::ReadOnlyTaps{Int}   # read-only: a workspace may share the transform
            lp_filters::FilterPair{T}
            dfb_filters::QuincunxFilterPair{T}
            function $name{T}(scales, direction_levels, lp_filters, dfb_filters) where {T <: AbstractFloat}
                scales, direction_levels, lp_filters, dfb_filters =
                    _contourlet_fields(T, scales, direction_levels, lp_filters, dfb_filters)
                return new{T}(scales, direction_levels, lp_filters, dfb_filters)
            end
        end

        function $name{T}(;
                scales = nothing, direction_levels = nothing,
                lp_filters::FilterPair = CDF97, dfb_filters::QuincunxFilterPair = Q2345
            ) where {T <: AbstractFloat}
            return $name{T}(_resolve_levels(scales, direction_levels)..., lp_filters, dfb_filters)
        end

        function $name(;
                scales = nothing, direction_levels = nothing,
                lp_filters::FilterPair = CDF97, dfb_filters::QuincunxFilterPair = Q2345
            )
            T = promote_type(eltype(lp_filters), eltype(dfb_filters))
            return $name{T}(; scales, direction_levels, lp_filters, dfb_filters)
        end

        _family(::$name) = $name
    end
end

const _TRANSFORM_KEYWORDS = """
- `scales`: the number of pyramid levels (≥ 1), `J` in the papers;
- `direction_levels`: the depth of the directional filter bank at each scale,
  `scales` entries ≥ 0, coarsest scale first; scale `j` has
  `2^direction_levels[j]` directions.  These are the `l_j` of the papers and
  the `nlevs` of the MATLAB toolboxes;
- `lp_filters`: the pyramid's biorthogonal filter pair ([`CDF97`](@ref) by
  default);
- `dfb_filters`: the ladder pair of the directional filter banks
  ([`Q2345`](@ref) by default).

At least one of `scales` and `direction_levels` must be given: `scales` alone
takes `direction_levels = parabolic_levels(scales)` (see
[`parabolic_levels`](@ref)), `direction_levels` alone has
`scales = length(direction_levels)`, and when both are given they must agree.
The filters are stored at the precision of the wider of the two pairs, or at
precision `T` with the `{T}` form; a transform runs at the precision of its data (`Float32` data uses `Float32`
filters), whatever the precision it holds.  Transforms with equal parameters
are `==` and hash alike.

The parameters are public properties: `t.scales`, `t.direction_levels` (a
read-only vector), `t.lp_filters` and `t.dfb_filters`."""

"""
    CT(; scales, direction_levels, lp_filters = CDF97, dfb_filters = Q2345)
    CT{T}(; scales, direction_levels, lp_filters = CDF97, dfb_filters = Q2345)

The discrete contourlet transform of Do & Vetterli (2005): a Laplacian
pyramid of `scales` levels whose bandpass image at scale `j` is split by a
decimated directional filter bank of depth `direction_levels[j]`.  It is nearly
critically sampled (about 4/3 coefficients per pixel); its subbands have different sizes, so the
coefficients are a [`SubbandVector`](@ref).

Image sizes: any size whose bandpass at every scale suits that scale's
directional filter bank.  The bandpass at the finest scale has the image's size
and each coarser one half of it, rounded up; a depth `l ≥ 1` needs both of its
sides divisible by `2^max(l - 1, 1)` (depth 0 takes any size).  For example
`CT(scales = 3, direction_levels = [0, 2, 3])` takes 100 × 100, and
`direction_levels = [0, 0]` takes odd sizes.  A size that does not fit throws an
`ArgumentError` naming the scale, the axis and the nearest length that fits.

$(_TRANSFORM_KEYWORDS)

Subbands ([`subband_layout`](@ref)): the lowpass first, as
`(scale = 0, direction = 0)`, then for `scale = 1` (the coarsest), 2, …,
`scales` (the finest) its directional subbands `(scale = j, direction = k)`,
`k = 1, …, 2^direction_levels[j]`.
The transform runs serially; `threaded` is accepted and ignored.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 64, 64);

julia> t = CT(scales = 2, direction_levels = [1, 2])
CT(scales = 2, direction_levels = [1, 2])

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
    NSCT(; scales, direction_levels, lp_filters = CDF97, dfb_filters = Q2345)
    NSCT{T}(; scales, direction_levels, lp_filters = CDF97, dfb_filters = Q2345)

The nonsubsampled contourlet transform of da Cunha, Zhou & Do (2006): the
contourlet transform without its samplers (a nonsubsampled pyramid and
nonsubsampled directional filter banks), so every subband has the image's size
and the transform commutes with circular shifts of the image.  The coefficients
are one array of size `(size(x)..., n)`, subband `i` being `c[:, :, i]`.

$(_TRANSFORM_KEYWORDS)

Subbands ([`subband_layout`](@ref)): the lowpass first, as
`(scale = 0, direction = 0)`, then for `scale = 1` (the coarsest), 2, …,
`scales` (the finest) its directional subbands `(scale = j, direction = k)`,
`k = 1, …, 2^direction_levels[j]`.

With a workspace and `threaded = true` (the default `threaded = nothing` means
`true` for complex data, `false` for real data) the independent subband FFTs of
a host workspace run in one task per Julia thread, or in at most the `tasks`
the workspace was built with ([`make_workspace`](@ref)).

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(2), 32, 32);

julia> t = NSCT(scales = 2, direction_levels = [1, 2]);

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
    TICT(; scales, direction_levels, lp_filters = CDF97, dfb_filters = Q2345)
    TICT{T}(; scales, direction_levels, lp_filters = CDF97, dfb_filters = Q2345)

The translation-invariant contourlet transform of Eslami & Radha (2006).  A
translation-invariant Laplacian pyramid of `scales` levels gives, at every
level `i`, four full-size highpass channels; each is split by a translation-invariant
directional filter bank of depth `direction_levels[i]`: the full tree for channels 1
and 4 (`2^direction_levels[i]` subbands), a half tree for channels 2 and 3
(`2^(direction_levels[i] - 1) + 1` subbands).  Every subband has the image's size, and
the transform commutes with circular shifts of the image.  The coefficients are
one array of size `(size(x)..., n)`, subband `i` being `c[:, :, i]`.

The redundancy is `3 Σᵢ 2^direction_levels[i] + 2 scales + 1` when every
`direction_levels[i] ≥ 1` (a
level with `direction_levels[i] = 0` contributes its 4 channels).  [`STICT`](@ref) trades
the directional stage's translation invariance for a redundancy of `4 scales + 1`.
`lp_filters` must be biorthogonal (as [`CDF97`](@ref) is).

$(_TRANSFORM_KEYWORDS)

Subbands ([`subband_layout`](@ref)): the lowpass first, as
`(level = 0, channel = 0, direction = 0)`, then for `level = 1` (the
coarsest), 2, …, `scales` (the finest), for `channel = 1, …, 4`, the directional subbands
`(level = i, channel = c, direction = k)`.

Threading as for [`NSCT`](@ref).

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 32, 32);

julia> t = TICT(scales = 2, direction_levels = [1, 2]);

julia> c = analyze(t, x);

julia> size(c, 3) == 1 + (2 + 2 + 2 + 2) + (4 + 3 + 3 + 4)
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
    STICT(; scales, direction_levels, lp_filters = CDF97, dfb_filters = Q2345)
    STICT{T}(; scales, direction_levels, lp_filters = CDF97, dfb_filters = Q2345)

The semi-translation-invariant contourlet transform of Eslami & Radha (2006):
the translation-invariant Laplacian pyramid of [`TICT`](@ref) followed by
critically sampled directional filter banks (full tree on channels 1 and 4, half
trees on 2 and 3).  The pyramid commutes with circular shifts; the directional
subbands are those of the decimated DFB.  Redundancy `4 scales + 1`; the subbands
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

julia> t = STICT(scales = 2, direction_levels = [2, 3]);

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
    return _family(a) === _family(b) && a.scales == b.scales && a.direction_levels == b.direction_levels &&
        a.lp_filters == b.lp_filters && a.dfb_filters == b.dfb_filters
end

function Base.hash(t::_ContourletTransform, h::UInt)
    h = hash(nameof(_family(t)), h)
    return hash(t.dfb_filters, hash(t.lp_filters, hash(t.direction_levels, hash(t.scales, h))))
end

function Base.show(io::IO, t::_ContourletTransform{T}) where {T}
    print(io, nameof(_family(t)))
    T === Float64 || print(io, "{", T, "}")
    print(io, "(scales = ", t.scales, ", direction_levels = ", t.direction_levels)
    t.lp_filters == _convert_pair(T, CDF97) || print(io, ", lp_filters = ", t.lp_filters)
    t.dfb_filters == _convert_qfp(T, Q2345) || print(io, ", dfb_filters = ", t.dfb_filters)
    return print(io, ")")
end

_convert_pair(::Type{T}, fp::FilterPair{T}) where {T <: AbstractFloat} = fp
_convert_pair(::Type{T}, fp::FilterPair) where {T <: AbstractFloat} = FilterPair{T}(T.(fp.h), T.(fp.g))

# The transform `t` at filter precision `Tf` (`t` itself when it already is).
_convert_transform(::Type{Tf}, t::_ContourletTransform{Tf}) where {Tf <: AbstractFloat} = t
_convert_transform(::Type{Tf}, t::_ContourletTransform) where {Tf <: AbstractFloat} =
    _family(t){Tf}(t.scales, t.direction_levels, t.lp_filters, t.dfb_filters)

# Images only: the contourlet transforms are planar.
_check_image(dims::Dims{2}) = dims
_check_image(dims::Dims) = throw(ArgumentError("the contourlet transforms take images (two axes), got data of size $(dims)"))

"""
    parabolic_levels(scales, coarsest = 1) -> Vector{Int}

The directional depths of `scales` scales, coarsest first, following the
parabolic scaling law

    l_j = coarsest + ⌊(j - 1) / 2⌋,    j = 1, …, scales,

so that the number of directions doubles every other scale from the coarsest
(`l_1 = coarsest`) to the finest (`j = scales`) — the discrete counterpart of
parabolic scaling, width ∝ length², in Do & Vetterli (2005).  It is the default
`direction_levels` of the transforms built with `scales` alone.

# Examples
```jldoctest
julia> using Contourlets

julia> parabolic_levels(4)
4-element Vector{Int64}:
 1
 1
 2
 2

julia> parabolic_levels(5, 2)
5-element Vector{Int64}:
 2
 2
 3
 3
 4
```

# References
1. M. N. Do and M. Vetterli, "The Contourlet Transform: An Efficient Directional
   Multiresolution Image Representation", *IEEE Transactions on Image Processing*,
   14(12):2091–2106, 2005.
   [doi:10.1109/TIP.2005.859376](https://doi.org/10.1109/TIP.2005.859376)
"""
function parabolic_levels(scales::Integer, coarsest::Integer = 1)::Vector{Int}
    scales, coarsest = Int(scales), Int(coarsest)
    return [max(0, coarsest + (j - 1) ÷ 2) for j in 1:scales]
end
