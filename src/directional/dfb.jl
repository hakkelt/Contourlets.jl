# l-level binary tree Directional Filter Bank (DFB).
#
# Two construction paths are provided:
#
#  • Ladder path (default, used for ladder-mode filter pairs such as Q2345) —
#    a faithful port of the Do & Vetterli (2005) directional filter bank using
#    the Phoong et al. (1995) lifting network.  It composes quincunx (levels
#    1–2) and parallelogram (levels ≥3) polyphase decompositions with
#    resampling/backsampling so that the 2ᴸ subbands tile the frequency plane
#    into genuine directional wedges.  Perfect reconstruction is structural.
#
#  • Modulation-mode shear path (used for modulation-mode pairs, e.g. a
#    user-supplied Haar pair) — the simpler shear + 1-D split tree.  It is
#    perfect-reconstructing but only weakly directional; it is the only decimated
#    DFB available for custom modulation-mode filter pairs.
#
# Subband sizes after l levels (input n₁ × n₂):
#   l=1: 2 subbands  n₁ × n₂/2
#   l=2: 4 subbands  n₁/2 × n₂/2
#   l=3: 8 subbands  n₁/2 × n₂/4  (4 "mostly-horizontal" + 4 "mostly-vertical")
#   l=4: 16 subbands n₁/4 × n₂/4
#
# Reference: M. N. Do & M. Vetterli, "The contourlet transform", IEEE TIP 2005;
#            the MATLAB Contourlet Toolbox functions dfbdec_l / dfbrec_l.

# ── Resampling (periodic shear) ───────────────────────────────────────────────
# Resampling matrices (see the toolbox `resamp`/`resampc`):
#   R1 = [1 shift; 0 1]   R2 = [1 -shift; 0 1]   R3 = [1 0; shift 1]   R4 = [1 0; -shift 1]

function _resamp(x::AbstractMatrix{T}, type::Int, shift::Int = 1) where {T}
    m, n = size(x)
    y = similar(x)
    if type == 1
        @inbounds for j in 1:n, i in 1:m
            y[i, j] = x[mod1(i + shift * (j - 1), m), j]
        end
    elseif type == 2
        @inbounds for j in 1:n, i in 1:m
            y[i, j] = x[mod1(i - shift * (j - 1), m), j]
        end
    elseif type == 3
        @inbounds for j in 1:n, i in 1:m
            y[i, j] = x[i, mod1(j + shift * (i - 1), n)]
        end
    elseif type == 4
        @inbounds for j in 1:n, i in 1:m
            y[i, j] = x[i, mod1(j - shift * (i - 1), n)]
        end
    else
        throw(ArgumentError("resamp type must be 1..4"))
    end
    return y
end

# Cyclic column/row roll: forward = x[:, [2:end; 1]]; back = x[:, [end; 1:end-1]].
# Expressed via `circshift` so the roll is a single vectorised op (and avoids
# index-vector gathers that would force scalar indexing on the GPU).
_roll_cols(x; back::Bool = false) = circshift(x, (0, back ? 1 : -1))
_roll_rows(x; back::Bool = false) = circshift(x, (back ? 1 : -1, 0))

# Zero matrix on the same device/array type as `ref` (so the polyphase
# reconstructors stay on the GPU when given device arrays).
_zeros_like(ref::AbstractArray, dims::Vararg{Int}) =
    fill!(similar(ref, dims...), zero(eltype(ref)))

# ── Quincunx polyphase (types '1r', '2c') ─────────────────────────────────────

function _qpdec(x::AbstractMatrix, type::Symbol)
    if type === :q1r            # Q1 = R2 * D1 * R3
        y = _resamp(x, 2)
        p0 = _resamp(y[1:2:end, :], 3)
        p1 = _resamp(_roll_cols(y[2:2:end, :]), 3)
    elseif type === :q2c        # Q2 = R4 * D2 * R1
        y = _resamp(x, 4)
        p0 = _resamp(y[:, 1:2:end], 1)
        p1 = _resamp(_roll_rows(y[:, 2:2:end]), 1)
    else
        throw(ArgumentError("unsupported quincunx type $type"))
    end
    return p0, p1
end

function _qprec(p0::AbstractMatrix{T}, p1::AbstractMatrix{T}, type::Symbol) where {T}
    if type === :q1r
        m, n = size(p0)
        y = _zeros_like(p0, 2m, n)
        y[1:2:end, :] = _resamp(p0, 4)
        y[2:2:end, :] = _roll_cols(_resamp(p1, 4); back = true)
        return _resamp(y, 1)
    elseif type === :q2c
        m, n = size(p0)
        y = _zeros_like(p0, m, 2n)
        y[:, 1:2:end] = _resamp(p0, 2)
        y[:, 2:2:end] = _roll_rows(_resamp(p1, 2); back = true)
        return _resamp(y, 3)
    else
        throw(ArgumentError("unsupported quincunx type $type"))
    end
end

# ── Parallelogram polyphase (types 1..4) ──────────────────────────────────────

function _ppdec(x::AbstractMatrix, type::Int)
    if type == 1
        p0 = _resamp(x[1:2:end, :], 3)
        p1 = _resamp(_roll_cols(x[2:2:end, :]), 3)
    elseif type == 2
        p0 = _resamp(x[1:2:end, :], 4)
        p1 = _resamp(x[2:2:end, :], 4)
    elseif type == 3
        p0 = _resamp(x[:, 1:2:end], 1)
        p1 = _resamp(_roll_rows(x[:, 2:2:end]), 1)
    elseif type == 4
        p0 = _resamp(x[:, 1:2:end], 2)
        p1 = _resamp(x[:, 2:2:end], 2)
    else
        throw(ArgumentError("parallelogram type must be 1..4"))
    end
    return p0, p1
end

function _pprec(p0::AbstractMatrix{T}, p1::AbstractMatrix{T}, type::Int) where {T}
    m, n = size(p0)
    if type == 1
        x = _zeros_like(p0, 2m, n)
        x[1:2:end, :] = _resamp(p0, 4)
        x[2:2:end, :] = _roll_cols(_resamp(p1, 4); back = true)
    elseif type == 2
        x = _zeros_like(p0, 2m, n)
        x[1:2:end, :] = _resamp(p0, 3)
        x[2:2:end, :] = _resamp(p1, 3)
    elseif type == 3
        x = _zeros_like(p0, m, 2n)
        x[:, 1:2:end] = _resamp(p0, 2)
        x[:, 2:2:end] = _roll_rows(_resamp(p1, 2); back = true)
    elseif type == 4
        x = _zeros_like(p0, m, 2n)
        x[:, 1:2:end] = _resamp(p0, 1)
        x[:, 2:2:end] = _resamp(p1, 1)
    else
        throw(ArgumentError("parallelogram type must be 1..4"))
    end
    return x
end

# ── 2-D separable periodic filtering (sefilter2) ──────────────────────────────

function _extend2(x::AbstractMatrix{T}, ru::Int, rd::Int, cl::Int, cr::Int, extmod::Symbol) where {T}
    m, n = size(x)
    if extmod === :per
        out = Matrix{T}(undef, m + ru + rd, n + cl + cr)
        @inbounds for j in 1:(n + cl + cr), i in 1:(m + ru + rd)
            out[i, j] = x[mod1(i - ru, m), mod1(j - cl, n)]
        end
        return out
    elseif extmod === :qper_row
        m2 = round(Int, m / 2)
        out = Matrix{T}(undef, m + ru + rd, n + cl + cr)
        @inbounds for j in 1:(n + cl + cr), i in 1:(m + ru + rd)
            ii = mod1(i - ru, m)
            jj = j - cl
            if jj < 1 || jj > n
                # shift row index by m2 for column extension
                out[i, j] = x[mod1(ii + m2, m), mod1(jj, n)]
            else
                out[i, j] = x[ii, jj]
            end
        end
        return out
    elseif extmod === :qper_col
        n2 = round(Int, n / 2)
        out = Matrix{T}(undef, m + ru + rd, n + cl + cr)
        @inbounds for j in 1:(n + cl + cr), i in 1:(m + ru + rd)
            ii = i - ru
            jj = mod1(j - cl, n)
            if ii < 1 || ii > m
                # shift col index by n2 for row extension
                out[i, j] = x[mod1(ii, m), mod1(jj + n2, n)]
            else
                out[i, j] = x[ii, jj]
            end
        end
        return out
    else
        throw(ArgumentError("unsupported extmod: $extmod"))
    end
end

# y = X * F1(z1) * F2(z2) * z1^shift1 * z2^shift2, same size as x
function _sefilter2(x::AbstractMatrix{T}, f::Vector{T}, shift1::Int, shift2::Int, extmod::Symbol) where {T}
    L = length(f)
    lf = (L - 1) / 2
    ru = floor(Int, lf) + shift1
    rd = ceil(Int, lf) - shift1
    cl = floor(Int, lf) + shift2
    cr = ceil(Int, lf) - shift2
    ext = _extend2(x, ru, rd, cl, cr, extmod)
    m, n = size(x)
    # The lifting filter is symmetric (f[a] == f[L+1-a]); fold the taps so each
    # output uses ⌈L/2⌉ multiplies instead of L.  Cuts the FLOP count of this
    # (dominant) kernel roughly in half for the default Q2345 pair.
    sym = _is_symmetric(f, L)
    half = (L + 1) ÷ 2
    # Separable valid convolution with f along both dims (conv flips the filter).
    tmp = Matrix{T}(undef, m, size(ext, 2))
    if sym
        @inbounds for j in axes(ext, 2), i in 1:m
            acc = zero(T)
            for a in 1:half
                acc += f[a] * (ext[i + L - a, j] + ext[i + a - 1, j])
            end
            isodd(L) && (acc -= f[half] * ext[i + L - half, j])
            tmp[i, j] = acc
        end
    else
        @inbounds for j in axes(ext, 2), i in 1:m
            acc = zero(T)
            for a in 1:L
                acc += f[a] * ext[i + L - a, j]
            end
            tmp[i, j] = acc
        end
    end
    out = Matrix{T}(undef, m, n)
    if sym
        @inbounds for j in 1:n, i in 1:m
            acc = zero(T)
            for b in 1:half
                acc += f[b] * (tmp[i, j + L - b] + tmp[i, j + b - 1])
            end
            isodd(L) && (acc -= f[half] * tmp[i, j + L - half])
            out[i, j] = acc
        end
    else
        @inbounds for j in 1:n, i in 1:m
            acc = zero(T)
            for b in 1:L
                acc += f[b] * tmp[i, j + L - b]
            end
            out[i, j] = acc
        end
    end
    return out
end

# True when f reads the same forwards and backwards (symmetric impulse response).
@inline function _is_symmetric(f::Vector{T}, L::Int) where {T}
    @inbounds for a in 1:(L ÷ 2)
        f[a] == f[L + 1 - a] || return false
    end
    return true
end

# ── Two-channel ladder filter bank ────────────────────────────────────────────

# Polyphase decomposition selector.
_poly_dec(x, kind::Symbol, t) = kind === :q ? _qpdec(x, t) : _ppdec(x, t)
_poly_rec(p0, p1, kind::Symbol, t) = kind === :q ? _qprec(p0, p1, t) : _pprec(p0, p1, t)

function _fbdec_l(x::AbstractMatrix{T}, f::Vector{T}, kind::Symbol, t, extmod::Symbol = :per) where {T}
    p0, p1 = _poly_dec(x, kind, t)
    s2 = sqrt(T(2))
    y0 = (p0 .- _sefilter2(p1, f, 1, 1, extmod)) ./ s2
    y1 = (-s2) .* p1 .- _sefilter2(y0, f, 0, 0, extmod)
    return y0, y1
end

function _fbrec_l(y0::AbstractMatrix{T}, y1::AbstractMatrix{T}, f::Vector{T}, kind::Symbol, t, extmod::Symbol = :per) where {T}
    s2 = sqrt(T(2))
    p1 = (-(one(T) / s2)) .* (y1 .+ _sefilter2(y0, f, 0, 0, extmod))
    p0 = s2 .* y0 .+ _sefilter2(p1, f, 1, 1, extmod)
    return _poly_rec(p0, p1, kind, t)
end

# ── Backsampling (diagonal overall sampling) ──────────────────────────────────

function _backsamp!(y::Vector{<:AbstractMatrix})
    n = round(Int, log2(length(y)))
    if n == 1
        for k in 1:2
            yk = _resamp(y[k], 4)
            yk[:, 1:2:end] = _resamp(yk[:, 1:2:end], 1)
            yk[:, 2:2:end] = _resamp(yk[:, 2:2:end], 1)
            y[k] = yk
        end
    elseif n > 2
        N = 2^(n - 1)
        for k in 1:(2^(n - 2))
            shift = 2k - (2^(n - 2) + 1)
            y[2k - 1] = _resamp(y[2k - 1], 3, shift)
            y[2k] = _resamp(y[2k], 3, shift)
            y[2k - 1 + N] = _resamp(y[2k - 1 + N], 1, shift)
            y[2k + N] = _resamp(y[2k + N], 1, shift)
        end
    end
    return y
end

function _rebacksamp!(y::Vector{<:AbstractMatrix})
    n = round(Int, log2(length(y)))
    if n == 1
        for k in 1:2
            yk = copy(y[k])
            yk[:, 1:2:end] = _resamp(yk[:, 1:2:end], 2)
            yk[:, 2:2:end] = _resamp(yk[:, 2:2:end], 2)
            y[k] = _resamp(yk, 3)
        end
    elseif n > 2
        N = 2^(n - 1)
        for k in 1:(2^(n - 2))
            shift = 2k - (2^(n - 2) + 1)
            y[2k - 1] = _resamp(y[2k - 1], 3, -shift)
            y[2k] = _resamp(y[2k], 3, -shift)
            y[2k - 1 + N] = _resamp(y[2k - 1 + N], 1, -shift)
            y[2k + N] = _resamp(y[2k + N], 1, -shift)
        end
    end
    return y
end

# ── Ladder DFB decomposition / reconstruction ─────────────────────────────────

function _dfbdec_l(x::AbstractMatrix{T}, f::Vector{T}, n::Int) where {T}
    n == 0 && return [copy(x)]
    if n == 1
        y0, y1 = _fbdec_l(x, f, :q, :q1r, :qper_col)
        y = [y0, y1]
    else
        x0, x1 = _fbdec_l(x, f, :q, :q1r, :qper_col)
        M = typeof(x0)
        y = Vector{M}(undef, 4)
        y[2], y[1] = _fbdec_l(x0, f, :q, :q2c)
        y[4], y[3] = _fbdec_l(x1, f, :q, :q2c)
        for l in 3:n
            y_old = y
            y = Vector{M}(undef, 2^l)
            for k in 1:(2^(l - 2))
                i = mod(k - 1, 2) + 1
                y[2k], y[2k - 1] = _fbdec_l(y_old[k], f, :p, i)
            end
            for k in (2^(l - 2) + 1):(2^(l - 1))
                i = mod(k - 1, 2) + 3
                y[2k], y[2k - 1] = _fbdec_l(y_old[k], f, :p, i)
            end
        end
    end
    _backsamp!(y)
    # Flip the order of the second half channels.
    half = 2^(n - 1)
    y[(half + 1):end] = reverse(y[(half + 1):end])
    return y
end

function _dfbrec_l(y::Vector{<:AbstractMatrix{T}}, f::Vector{T}) where {T}
    n = round(Int, log2(length(y)))
    n == 0 && return copy(y[1])
    y = copy(y)
    M = typeof(y[1])
    half = 2^(n - 1)
    y[(half + 1):end] = reverse(y[(half + 1):end])
    _rebacksamp!(y)
    if n == 1
        return _fbrec_l(y[1], y[2], f, :q, :q1r, :qper_col)
    end
    for l in n:-1:3
        y_old = y
        y = Vector{M}(undef, 2^(l - 1))
        for k in 1:(2^(l - 2))
            i = mod(k - 1, 2) + 1
            y[k] = _fbrec_l(y_old[2k], y_old[2k - 1], f, :p, i)
        end
        for k in (2^(l - 2) + 1):(2^(l - 1))
            i = mod(k - 1, 2) + 3
            y[k] = _fbrec_l(y_old[2k], y_old[2k - 1], f, :p, i)
        end
    end
    x0 = _fbrec_l(y[2], y[1], f, :q, :q2c)
    x1 = _fbrec_l(y[4], y[3], f, :q, :q2c)
    return _fbrec_l(x0, x1, f, :q, :q1r, :qper_col)
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    dfb_decompose(bandpass, l_levels, qfp::QuincunxFilterPair) -> Vector{Matrix}

Decompose `bandpass` into `2^l_levels` directional subbands using an
`l_levels`-deep binary-tree DFB.

For the default ladder filter pair [`Q2345`](@ref) this is the Do & Vetterli
(2005) directional filter bank: the subbands tile the frequency plane into
`2^l_levels` directional wedges.  For modulation-mode pairs a simpler shear-based
tree is used (perfect-reconstructing but only weakly directional).

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
        qfp::QuincunxFilterPair
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    l_levels == 0 && return [copy(bandpass)]
    T = promote_type(eltype(bandpass), eltype(qfp))
    img = T === eltype(bandpass) ? bandpass : T.(bandpass)
    if is_ladder(qfp)
        return _dfbdec_l(img, T.(qfp.f_ladder), l_levels)
    end
    return _dfb_split(img, l_levels, 1, _convert_qfp(T, qfp))
end

"""
    dfb_reconstruct(subbands, qfp::QuincunxFilterPair) -> bandpass

Reconstruct a bandpass image from its `2^l` directional subbands.

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
        qfp::QuincunxFilterPair
    )
    n = length(subbands)
    n >= 1   || throw(ArgumentError("subbands must be non-empty"))
    ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
    n == 1 && return copy(subbands[1])
    T = promote_type(eltype(subbands[1]), eltype(qfp))
    if is_ladder(qfp)
        sbs = eltype(subbands[1]) === T ? subbands : [T.(s) for s in subbands]
        return _dfbrec_l(sbs, T.(qfp.f_ladder))
    end
    l_levels = round(Int, log2(n))
    return _dfb_merge(subbands, l_levels, 1, _convert_qfp(T, qfp))
end

# ── Modulation-mode shear path (modulation-mode pairs only) ───────────────────

function _dfb_split(
        img::AbstractMatrix, remaining::Int, depth::Int,
        qfp::QuincunxFilterPair
    )
    shear_dir = isodd(depth) ? :h : :v
    qfb_dir = isodd(depth) ? :col : :row
    sh = shear(img, shear_dir)
    sb0, sb1 = qfb_decompose(sh, qfp; dir = qfb_dir)
    sb0 = inv_shear(sb0, shear_dir)
    sb1 = inv_shear(sb1, shear_dir)
    remaining == 1 && return [sb0, sb1]
    return vcat(
        _dfb_split(sb0, remaining - 1, depth + 1, qfp),
        _dfb_split(sb1, remaining - 1, depth + 1, qfp)
    )
end

function _dfb_merge(
        sbs::Vector{<:AbstractMatrix}, l::Int, depth::Int,
        qfp::QuincunxFilterPair
    )
    shear_dir = isodd(depth) ? :h : :v
    qfb_dir = isodd(depth) ? :col : :row
    half = length(sbs) ÷ 2
    if l == 1
        sb0, sb1 = sbs[1], sbs[2]
    else
        sb0 = _dfb_merge(sbs[1:half], l - 1, depth + 1, qfp)
        sb1 = _dfb_merge(sbs[(half + 1):end], l - 1, depth + 1, qfp)
    end
    sh0 = shear(sb0, shear_dir)
    sh1 = shear(sb1, shear_dir)
    rec = qfb_reconstruct(sh0, sh1, qfp; dir = qfb_dir)
    return inv_shear(rec, shear_dir)
end
