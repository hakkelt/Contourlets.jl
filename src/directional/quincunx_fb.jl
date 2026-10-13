# Two-channel quincunx filter bank (QFB).
#
# Quincunx sampling factors into a shear (a unimodular resampling) and a
# rectangular downsampling by 2 along one axis (Do 2001, thesis §3.2.1), so after
# the shear the quincunx FB (§3.2.2) reduces to a 1-D two-channel filter bank in
# one spatial direction:
#
#   dir=:col  →  filter along columns, keep even columns  (used after :h shear)
#   dir=:row  →  filter along rows,    keep even rows     (used after :v shear)
#                (implemented via transpose trick: pass image' to the :col path)
#
# The 1-D bank is the two-step ladder of the pair's lifting filter (see
# q2345.jl), so perfect reconstruction is exact for any ladder filter.

"""
    qfb_decompose(image, qfp; dir=:col) -> (sb0, sb1)

Two-channel quincunx filter bank analysis.

* `dir=:col` (default): filter in column direction, keep even columns.
* `dir=:row`: filter in row direction, keep even rows (via transpose).

Returns `(sb0, sb1)` where `sb0` is the LP subband and `sb1` the HP subband.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(7), 16, 16);

julia> sb0, sb1 = qfb_decompose(x, Q2345);

julia> sb0_r, sb1_r = qfb_decompose(x, Q2345; dir = :row);

julia> size(sb0), size(sb0_r)
((16, 8), (8, 16))
```
"""
function qfb_decompose(image::AbstractMatrix, qfp::QuincunxFilterPair; dir::Symbol = :col)
    Td = _data_eltype(image)         # data type (real or complex)
    Tf = _filter_eltype(Td)          # real filter precision
    img = Td === eltype(image) ? image : Td.(image)
    qfpT = _convert_qfp(Tf, qfp)
    if dir === :col
        return _qfb_col_decompose(img, qfpT)
    else
        sb0_t, sb1_t = _qfb_col_decompose(img', qfpT)
        return collect(sb0_t'), collect(sb1_t')
    end
end

"""
    qfb_decompose!(sb0, sb1, image, qfp; dir=:col) -> (sb0, sb1)

In-place two-channel quincunx filter bank analysis.
"""
function qfb_decompose!(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        image::AbstractMatrix, qfp::QuincunxFilterPair;
        dir::Symbol = :col
    )
    if dir === :col
        return _qfb_col_decompose!(sb0, sb1, image, qfp)
    else
        _qfb_col_decompose!(
            PermutedDimsArray(sb0, (2, 1)),
            PermutedDimsArray(sb1, (2, 1)),
            transpose(image), qfp
        )
        return sb0, sb1
    end
end

"""
    qfb_reconstruct(sb0, sb1, qfp; dir=:col) -> image

Two-channel quincunx filter bank synthesis.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(7), 16, 16);

julia> sb0, sb1 = qfb_decompose(x, Q2345);

julia> rec = qfb_reconstruct(sb0, sb1, Q2345);

julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function qfb_reconstruct(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        qfp::QuincunxFilterPair; dir::Symbol = :col
    )
    Td = _data_eltype(promote_type(eltype(sb0), eltype(sb1)))  # data type
    Tf = _filter_eltype(Td)                                    # real filter precision
    sb0T = Td === eltype(sb0) ? sb0 : Td.(sb0)
    sb1T = Td === eltype(sb1) ? sb1 : Td.(sb1)
    qfpT = _convert_qfp(Tf, qfp)
    if dir === :col
        return _qfb_col_reconstruct(sb0T, sb1T, qfpT)
    else
        rec_t = _qfb_col_reconstruct(collect(sb0T'), collect(sb1T'), qfpT)
        return collect(rec_t')
    end
end

"""
    qfb_reconstruct!(out, sb0, sb1, qfp; dir=:col) -> out

In-place two-channel quincunx filter bank synthesis.
"""
function qfb_reconstruct!(
        out::AbstractMatrix, sb0::AbstractMatrix,
        sb1::AbstractMatrix, qfp::QuincunxFilterPair;
        dir::Symbol = :col
    )
    if dir === :col
        return _qfb_col_reconstruct!(out, sb0, sb1, qfp)
    else
        _qfb_col_reconstruct!(
            PermutedDimsArray(out, (2, 1)),
            PermutedDimsArray(sb0, (2, 1)),
            PermutedDimsArray(sb1, (2, 1)),
            qfp
        )
        return out
    end
end

# ── Internal column-direction implementation ──────────────────────────────────

# Internal: callers (the public `qfb_decompose`) pass `img` at the data type `Td`
# and a real `qfp` at the filter precision `Tf`.
function _qfb_col_decompose(img::AbstractMatrix{Td}, qfp::QuincunxFilterPair{Tf}) where {Td, Tf}
    n1, n2 = size(img)
    d2 = n2 ÷ 2
    sb0 = zeros(Td, n1, d2)
    sb1 = zeros(Td, n1, d2)
    return _qfb_col_decompose!(sb0, sb1, img, qfp)
end

function _qfb_col_decompose!(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        image::AbstractMatrix, qfp::QuincunxFilterPair
    )
    iseven(size(image, 2)) || throw(ArgumentError("the QFB requires an even number of columns"))
    return _qfb_col_decompose_ladder!(sb0, sb1, image, qfp)
end

# Internal: callers (the public `qfb_reconstruct`) pass `sb0`/`sb1`/`qfp` already
# promoted to a common element type `T`.
function _qfb_col_reconstruct(
        sb0::AbstractMatrix{Td}, sb1::AbstractMatrix{Td},
        qfp::QuincunxFilterPair{Tf}
    ) where {Td, Tf}
    n1 = size(sb0, 1)
    n2 = size(sb0, 2) * 2
    out = zeros(Td, n1, n2)
    return _qfb_col_reconstruct!(out, sb0, sb1, qfp)
end

function _qfb_col_reconstruct!(
        out::AbstractMatrix, sb0::AbstractMatrix,
        sb1::AbstractMatrix, qfp::QuincunxFilterPair
    )
    iseven(size(out, 2)) || throw(ArgumentError("the QFB requires an even number of columns"))
    fill!(out, zero(eltype(out)))
    return _qfb_col_reconstruct_ladder!(out, sb0, sb1, qfp)
end

# ── Ladder (lifting) realisation — structural PR, periodic extension ─────────
#
# Channel 0 lives on the odd image columns (1-based), channel 1 on the even
# columns.  See src/filters/q2345.jl for the ladder derivation.

function _qfb_col_decompose_ladder!(
        sb0::AbstractMatrix, sb1::AbstractMatrix,
        img::AbstractMatrix, qfp::QuincunxFilterPair
    )
    T = eltype(sb0)
    f = _ladder_modulate(T.(qfp.f_ladder))
    c = length(f) ÷ 2
    n1 = size(img, 1)
    d2 = size(sb0, 2)
    s2 = sqrt(T(2))
    # y0 = (1/√2)·(p0 − B₁(p1)),  B₁ lags m − c + 1
    @inbounds for k in 1:d2
        for i in 1:n1
            acc = zero(T)
            for m in eachindex(f)
                kk = mod1(k - (m - c + 1), d2)
                acc += f[m] * img[i, 2kk]
            end
            sb0[i, k] = (img[i, 2k - 1] - acc) / s2
        end
    end
    # y1 = −√2·p1 − B₀(y0),  B₀ lags m − c
    @inbounds for k in 1:d2
        for i in 1:n1
            acc = zero(T)
            for m in eachindex(f)
                kk = mod1(k - (m - c), d2)
                acc += f[m] * sb0[i, kk]
            end
            sb1[i, k] = -s2 * img[i, 2k] - acc
        end
    end
    return sb0, sb1
end

function _qfb_col_reconstruct_ladder!(
        out::AbstractMatrix, sb0::AbstractMatrix,
        sb1::AbstractMatrix, qfp::QuincunxFilterPair
    )
    T = eltype(out)
    f = _ladder_modulate(T.(qfp.f_ladder))
    c = length(f) ÷ 2
    n1 = size(out, 1)
    d2 = size(sb0, 2)
    s2 = sqrt(T(2))
    # p1 = (−1/√2)·(y1 + B₀(y0)) → even columns
    @inbounds for k in 1:d2
        for i in 1:n1
            acc = zero(T)
            for m in eachindex(f)
                kk = mod1(k - (m - c), d2)
                acc += f[m] * sb0[i, kk]
            end
            out[i, 2k] = -(sb1[i, k] + acc) / s2
        end
    end
    # p0 = √2·y0 + B₁(p1) → odd columns
    @inbounds for k in 1:d2
        for i in 1:n1
            acc = zero(T)
            for m in eachindex(f)
                kk = mod1(k - (m - c + 1), d2)
                acc += f[m] * out[i, 2kk]
            end
            out[i, 2k - 1] = s2 * sb0[i, k] + acc
        end
    end
    return out
end
