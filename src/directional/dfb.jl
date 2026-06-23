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
    y = _scratch_like(x, m, n)
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
function _roll_cols(x; back::Bool = false)
    y = _scratch_like(x, size(x, 1), size(x, 2))
    return circshift!(y, x, (0, back ? 1 : -1))
end

function _roll_rows(x; back::Bool = false)
    y = _scratch_like(x, size(x, 1), size(x, 2))
    return circshift!(y, x, (back ? 1 : -1, 0))
end

# Zero matrix on the same device/array type as `ref` (so the polyphase
# reconstructors stay on the GPU when given device arrays).
_zeros_like(ref::AbstractArray, dims::Vararg{Int}) =
    fill!(_scratch_like(ref, dims...), zero(eltype(ref)))

# ── Quincunx polyphase (types '1r', '2c') ─────────────────────────────────────

function _qpdec(x::AbstractMatrix, type::Symbol)
    if type === :q1r            # Q1 = R2 * D1 * R3
        y = _resamp(x, 2)
        p0 = _resamp(@view(y[1:2:end, :]), 3)
        p1 = _resamp(_roll_cols(@view(y[2:2:end, :])), 3)
    elseif type === :q2c        # Q2 = R4 * D2 * R1
        y = _resamp(x, 4)
        p0 = _resamp(@view(y[:, 1:2:end]), 1)
        p1 = _resamp(_roll_rows(@view(y[:, 2:2:end])), 1)
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
        p0 = _resamp(@view(x[1:2:end, :]), 3)
        p1 = _resamp(_roll_cols(@view(x[2:2:end, :])), 3)
    elseif type == 2
        p0 = _resamp(@view(x[1:2:end, :]), 4)
        p1 = _resamp(@view(x[2:2:end, :]), 4)
    elseif type == 3
        p0 = _resamp(@view(x[:, 1:2:end]), 1)
        p1 = _resamp(_roll_rows(@view(x[:, 2:2:end])), 1)
    elseif type == 4
        p0 = _resamp(@view(x[:, 1:2:end]), 2)
        p1 = _resamp(@view(x[:, 2:2:end]), 2)
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
        out = _scratch_like(x, m + ru + rd, n + cl + cr)
        @inbounds for j in 1:(n + cl + cr), i in 1:(m + ru + rd)
            out[i, j] = x[mod1(i - ru, m), mod1(j - cl, n)]
        end
        return out
    elseif extmod === :qper_row
        m2 = round(Int, m / 2)
        out = _scratch_like(x, m + ru + rd, n + cl + cr)
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
        out = _scratch_like(x, m + ru + rd, n + cl + cr)
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

# y = X * F1(z1) * F2(z2) * z1^shift1 * z2^shift2, same size as x.
# Real filter `f` (Tf) applied to data `x` (Td, real or complex); `acc` is Td.
function _sefilter2(x::AbstractMatrix{Td}, f::Vector{Tf}, shift1::Int, shift2::Int, extmod::Symbol, threaded::Bool = false) where {Td, Tf}
    L = length(f)
    lf = (L - 1) / 2
    ru = floor(Int, lf) + shift1
    rd = ceil(Int, lf) - shift1
    cl = floor(Int, lf) + shift2
    cr = ceil(Int, lf) - shift2
    ext = _extend2(x, ru, rd, cl, cr, extmod)
    m, n = size(x)
    sym = _is_symmetric(f, L)
    half = (L + 1) ÷ 2
    # Separable valid convolution with f along both dims (conv flips the filter).
    tmp = _scratch_like(x, m, size(ext, 2))
    out = _scratch_like(x, m, n)
    return _sefilter2_kernel!(out, tmp, ext, f, L, half, sym, threaded)
end

function _sefilter2_kernel!(out::AbstractMatrix{Td}, tmp::AbstractMatrix{Td}, ext::AbstractMatrix{Td}, f::Vector{Tf}, L::Int, half::Int, sym::Bool, threaded::Bool) where {Td, Tf}
    m, n = size(out)
    if sym
        if threaded
            @batch for j in axes(ext, 2)
                @inbounds for i in 1:m
                    acc = zero(Td)
                    for a in 1:half
                        acc += f[a] * (ext[i + L - a, j] + ext[i + a - 1, j])
                    end
                    isodd(L) && (acc -= f[half] * ext[i + L - half, j])
                    tmp[i, j] = acc
                end
            end
            @batch for j in 1:n
                @inbounds for i in 1:m
                    acc = zero(Td)
                    for b in 1:half
                        acc += f[b] * (tmp[i, j + L - b] + tmp[i, j + b - 1])
                    end
                    isodd(L) && (acc -= f[half] * tmp[i, j + L - half])
                    out[i, j] = acc
                end
            end
        else
            for j in axes(ext, 2)
                @inbounds for i in 1:m
                    acc = zero(Td)
                    for a in 1:half
                        acc += f[a] * (ext[i + L - a, j] + ext[i + a - 1, j])
                    end
                    isodd(L) && (acc -= f[half] * ext[i + L - half, j])
                    tmp[i, j] = acc
                end
            end
            for j in 1:n
                @inbounds for i in 1:m
                    acc = zero(Td)
                    for b in 1:half
                        acc += f[b] * (tmp[i, j + L - b] + tmp[i, j + b - 1])
                    end
                    isodd(L) && (acc -= f[half] * tmp[i, j + L - half])
                    out[i, j] = acc
                end
            end
        end
    else
        if threaded
            @batch for j in axes(ext, 2)
                @inbounds for i in 1:m
                    acc = zero(Td)
                    for a in 1:L
                        acc += f[a] * ext[i + L - a, j]
                    end
                    tmp[i, j] = acc
                end
            end
            @batch for j in 1:n
                @inbounds for i in 1:m
                    acc = zero(Td)
                    for b in 1:L
                        acc += f[b] * tmp[i, j + L - b]
                    end
                    out[i, j] = acc
                end
            end
        else
            for j in axes(ext, 2)
                @inbounds for i in 1:m
                    acc = zero(Td)
                    for a in 1:L
                        acc += f[a] * ext[i + L - a, j]
                    end
                    tmp[i, j] = acc
                end
            end
            for j in 1:n
                @inbounds for i in 1:m
                    acc = zero(Td)
                    for b in 1:L
                        acc += f[b] * tmp[i, j + L - b]
                    end
                    out[i, j] = acc
                end
            end
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

# Polyphase decomposition selector — dispatch on t's type (Symbol→quincunx, Int→parallelogram).
_poly_dec(x, t::Symbol) = _qpdec(x, t)
_poly_dec(x, t::Int) = _ppdec(x, t)
_poly_rec(p0, p1, t::Symbol) = _qprec(p0, p1, t)
_poly_rec(p0, p1, t::Int) = _pprec(p0, p1, t)

function _fbdec_l(x::AbstractMatrix{Td}, f::Vector{Tf}, t, extmod::Symbol = :per, threaded::Bool = false) where {Td, Tf}
    p0, p1 = _poly_dec(x, t)
    s2 = sqrt(Tf(2))
    sef = _sefilter2(p1, f, 1, 1, extmod, threaded)
    y0 = _scratch_like(p0, size(p0, 1), size(p0, 2))
    if Td <: Real && y0 isa Array
        @turbo @. y0 = (p0 - sef) / s2
    else
        @. y0 = (p0 - sef) / s2
    end
    sef = _sefilter2(y0, f, 0, 0, extmod, threaded)
    y1 = _scratch_like(p1, size(p1, 1), size(p1, 2))
    if Td <: Real && y1 isa Array
        @turbo @. y1 = (-s2) * p1 - sef
    else
        @. y1 = (-s2) * p1 - sef
    end
    return y0, y1
end

function _fbrec_l(y0::AbstractMatrix{Td}, y1::AbstractMatrix{Td}, f::Vector{Tf}, t, extmod::Symbol = :per, threaded::Bool = false) where {Td, Tf}
    s2 = sqrt(Tf(2))
    sef = _sefilter2(y0, f, 0, 0, extmod, threaded)
    p1 = _scratch_like(y1, size(y1, 1), size(y1, 2))
    if Td <: Real && p1 isa Array
        @turbo @. p1 = (-(one(Tf) / s2)) * (y1 + sef)
    else
        @. p1 = (-(one(Tf) / s2)) * (y1 + sef)
    end
    sef = _sefilter2(p1, f, 1, 1, extmod, threaded)
    p0 = _scratch_like(y0, size(y0, 1), size(y0, 2))
    if Td <: Real && p0 isa Array
        @turbo @. p0 = s2 * y0 + sef
    else
        @. p0 = s2 * y0 + sef
    end
    return _poly_rec(p0, p1, t)
end

# ── Backsampling (diagonal overall sampling) ──────────────────────────────────

function _backsamp!(y::Vector{<:AbstractMatrix})
    n = round(Int, log2(length(y)))
    if n == 1
        for k in 1:2
            yk = _resamp(y[k], 4)
            yk[:, 1:2:end] = _resamp(@view(yk[:, 1:2:end]), 1)
            yk[:, 2:2:end] = _resamp(@view(yk[:, 2:2:end]), 1)
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
            yk = copyto!(_scratch_like(y[k], size(y[k])...), y[k])
            yk[:, 1:2:end] = _resamp(@view(yk[:, 1:2:end]), 2)
            yk[:, 2:2:end] = _resamp(@view(yk[:, 2:2:end]), 2)
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

function _dfbdec_l(x::AbstractMatrix{Td}, f::Vector{Tf}, n::Int, threaded::Bool = false) where {Td, Tf}
    n == 0 && return [copy(x)]
    if n == 1
        y0, y1 = _fbdec_l(x, f, :q1r, :qper_col, threaded)
        y = [y0, y1]
    else
        x0, x1 = _fbdec_l(x, f, :q1r, :qper_col, threaded)
        M = typeof(x0)
        y = Vector{M}(undef, 4)
        y[2], y[1] = _fbdec_l(x0, f, :q2c, :per, threaded)
        y[4], y[3] = _fbdec_l(x1, f, :q2c, :per, threaded)

        for l in 3:n
            y_old = y
            y = Vector{M}(undef, 2^l)
            for k in 1:(2^(l - 2))
                i = mod(k - 1, 2) + 1
                y[2k], y[2k - 1] = _fbdec_l(y_old[k], f, i, :per, threaded)
            end
            for k in (2^(l - 2) + 1):(2^(l - 1))
                i = mod(k - 1, 2) + 3
                y[2k], y[2k - 1] = _fbdec_l(y_old[k], f, i, :per, threaded)
            end
        end
    end
    _backsamp!(y)
    # Flip the order of the second half channels.
    half = 2^(n - 1)
    reverse!(@view y[(half + 1):end])
    return y
end

function _dfbrec_l(y::Vector{<:AbstractMatrix{Td}}, f::Vector{Tf}, threaded::Bool = false) where {Td, Tf}
    n = round(Int, log2(length(y)))
    n == 0 && return copy(y[1])
    y = copy(y)
    M = typeof(y[1])
    half = 2^(n - 1)
    reverse!(@view y[(half + 1):end])
    _rebacksamp!(y)
    if n == 1
        return _fbrec_l(y[1], y[2], f, :q1r, :qper_col, threaded)
    end
    for l in n:-1:3
        y_old = y
        y = Vector{M}(undef, 2^(l - 1))
        for k in 1:(2^(l - 2))
            i = mod(k - 1, 2) + 1
            y[k] = _fbrec_l(y_old[2k], y_old[2k - 1], f, i, :per, threaded)
        end
        for k in (2^(l - 2) + 1):(2^(l - 1))
            i = mod(k - 1, 2) + 3
            y[k] = _fbrec_l(y_old[2k], y_old[2k - 1], f, i, :per, threaded)
        end
    end
    x0 = _fbrec_l(y[2], y[1], f, :q2c, :per, threaded)
    x1 = _fbrec_l(y[4], y[3], f, :q2c, :per, threaded)
    return _fbrec_l(x0, x1, f, :q1r, :qper_col, threaded)
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    dfb_decompose(bandpass, l_levels, qfp::QuincunxFilterPair; threading=Auto()) -> Vector{Matrix}

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
        qfp::QuincunxFilterPair;
        threading::ThreadingPolicy = Auto()
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    l_levels == 0 && return [copy(bandpass)]
    Td = _data_eltype(bandpass)      # data type (real or complex)
    Tf = _filter_eltype(Td)          # real filter precision
    img = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    threaded = _use_threading(threading, Td)

    if is_ladder(qfp)
        # `f_ladder` stores the unmodulated pkva filter; the fan modulation is
        # applied once here, the same convention used by the QFB and NSDFB paths.
        f = _ladder_modulate(Tf.(qfp.f_ladder))
        return _dfbdec_l(img, f, l_levels, threaded)
    end
    return _dfb_split(img, l_levels, 1, _convert_qfp(Tf, qfp), threaded)
end

"""
    dfb_decompose!(subbands::Vector{<:AbstractMatrix}, bandpass, l_levels, qfp::QuincunxFilterPair;
                   workspace=nothing, threading=Auto()) -> subbands

In-place DFB analysis. `subbands` must be a vector of `2^l_levels` preallocated matrices of the correct sizes.
If a `workspace` is provided, intermediate allocations are avoided.

**Keyword arguments:**
- `threading::ThreadingPolicy`: Sets the threading strategy. Defaults to `Auto()`, which uses SIMD (single-threaded) for Real arrays, and `Polyester.@batch` (multithreaded) for Complex arrays.
"""
function dfb_decompose!(
        subbands::Vector{<:AbstractMatrix}, bandpass::AbstractMatrix, l_levels::Int,
        qfp::QuincunxFilterPair;
        workspace = nothing,
        threading::ThreadingPolicy = Auto()
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    length(subbands) == 2^l_levels || throw(ArgumentError("subbands must have 2^l_levels matrices"))
    if workspace !== nothing
        _arena_reset!(workspace.fwd_scratch)
        _with_arena(workspace.fwd_scratch) do
            sbs = dfb_decompose(bandpass, l_levels, qfp; threading = threading)
            for k in eachindex(sbs)
                copyto!(subbands[k], sbs[k])
            end
        end
    else
        sbs = dfb_decompose(bandpass, l_levels, qfp; threading = threading)
        for k in eachindex(sbs)
            copyto!(subbands[k], sbs[k])
        end
    end
    return subbands
end

"""
    dfb_reconstruct(subbands, qfp::QuincunxFilterPair; threading=Auto()) -> bandpass

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
        qfp::QuincunxFilterPair;
        threading::ThreadingPolicy = Auto()
    )
    n = length(subbands)
    n >= 1   || throw(ArgumentError("subbands must be non-empty"))
    ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
    n == 1 && return copy(subbands[1])
    Td = _data_eltype(subbands[1])   # data type (real or complex)
    Tf = _filter_eltype(Td)          # real filter precision
    threaded = _use_threading(threading, Td)

    if is_ladder(qfp)
        sbs = eltype(subbands[1]) === Td ? subbands : [Td.(s) for s in subbands]
        # Modulate once at use (see `dfb_decompose`).
        f = _ladder_modulate(Tf.(qfp.f_ladder))
        return _dfbrec_l(sbs, f, threaded)
    end
    l_levels = round(Int, log2(n))
    return _dfb_merge(subbands, l_levels, 1, _convert_qfp(Tf, qfp), threaded)
end

"""
    dfb_reconstruct!(bandpass::AbstractMatrix, subbands::Vector{<:AbstractMatrix}, qfp::QuincunxFilterPair;
                     workspace=nothing) -> bandpass

In-place DFB synthesis. `bandpass` is modified to store the reconstructed image.
If a `workspace` is provided, intermediate allocations are avoided.

**Keyword arguments:**
- `threading::ThreadingPolicy`: Sets the threading strategy. Defaults to `Auto()`.
"""
function dfb_reconstruct!(
        bandpass::AbstractMatrix, subbands::Vector{<:AbstractMatrix}, qfp::QuincunxFilterPair;
        workspace = nothing,
        threading::ThreadingPolicy = Auto()
    )
    if workspace !== nothing
        _arena_reset!(workspace.inv_scratch)
        _with_arena(workspace.inv_scratch) do
            bp = dfb_reconstruct(subbands, qfp; threading = threading)
            copyto!(bandpass, bp)
        end
    else
        bp = dfb_reconstruct(subbands, qfp; threading = threading)
        copyto!(bandpass, bp)
    end
    return bandpass
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
