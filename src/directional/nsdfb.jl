# Nonsubsampled Directional Filter Bank (NSDFB).
#
# This is the resampling-matrix construction of da Cunha, Zhou & Do (2006) — the
# shift-invariant analog of the decimated DFB.  Unlike a 1-D filter applied along
# a lattice direction (which yields *stripe* passbands and hence parallelogram
# *blocks*), this bank uses genuine 2-D **fan** and **parallelogram** filters
# convolved with **upsampling matrices**, so each directional subband occupies an
# angular **wedge** through the frequency origin.
#
# Construction (mirrors the reference `nsdfbdec.m`/`nsdfbrec.m`):
#   • diamond filters (h0,h1) from the pkva ladder β  (`_ld2quin`)
#   • fan filters       k1,k2 = modulate2(h0,h1, :c)
#   • parallelogram     f1,f2 = parafilters(h0,h1)   (modulate2 + transpose + resampz)
#   • binary tree:
#       level 1:    split with k1,k2                 (no upsampling)
#       level 2:    split each branch with k1,k2 upsampled by Q1 = [1 -1; 1 1]
#       level ≥3:   split channel k with parallelogram f{i} upsampled by Mₖ (Do's
#                   thesis eq. 3.18); i = mod(k-1,2)+1 (first half) or +3 (second half)
#
# Every stage is a *periodic* (circular) convolution (`_efilter2` / `_zconv2`),
# so the NSDFB — and hence the NSCT — is exactly invariant under circular shifts.
# Perfect reconstruction is structural (the pkva ladder gives biorthogonal
# diamond filters); PR holds at every level to machine precision.
#
# Unlike the decimated DFB, the directional filters are *scale independent*: the
# NSCT applies the same bank at every pyramid level (the multiscale à trous
# upsampling lives in the nonsubsampled pyramid, not here).  The `tree_level`
# argument is accepted for API symmetry but does not upsample the directional
# filters (matching the reference NSCT).
#
# NOTE on modulation: the 2-D fan filters are produced by `modulate2(·, :c)` on
# the *diamond* filters here, which is independent of the decimated DFB's 1-D
# `_ladder_modulate` (see [[dfb-ladder-modulation-regression]]).  Do NOT apply
# `_ladder_modulate` on this path.

# ── MATLAB-equivalent filter primitives ──────────────────────────────────────

# modulate2(x, type): multiply by (-1)^(idx - origin), origin = floor(size/2)+1.
function _modulate2(x::AbstractMatrix{T}, type::Symbol) where {T}
    s1, s2 = size(x)
    o1 = fld(s1, 2) + 1
    o2 = fld(s2, 2) + 1
    y = copy(x)
    if type === :r || type === :b
        for i in 1:s1
            iszero((i - o1) & 1) || (@views y[i, :] .= .-y[i, :])
        end
    end
    if type === :c || type === :b
        for j in 1:s2
            iszero((j - o2) & 1) || (@views y[:, j] .= .-y[:, j])
        end
    end
    return y
end

# resampz(x, type, shift): zero-padded shear resampling; trims all-zero borders.
function _resampz(x::AbstractMatrix{T}, type::Int, shift::Int = 1) where {T}
    sx1, sx2 = size(x)
    if type in (1, 2)
        len = sx1 + abs(shift * (sx2 - 1))
        y = zeros(T, len, sx2)
        shift1 = [(n - 1) * (type == 1 ? -shift : shift) for n in 1:sx2]
        shift1[end] < 0 && (shift1 .-= shift1[end])
        for n in 1:sx2
            y[shift1[n] .+ (1:sx1), n] .= @view x[:, n]
        end
        start = 1
        while all(iszero, @view y[start, :])
            start += 1
        end
        finish = size(y, 1)
        while all(iszero, @view y[finish, :])
            finish -= 1
        end
        return y[start:finish, :]
    elseif type in (3, 4)
        len = sx2 + abs(shift * (sx1 - 1))
        y = zeros(T, sx1, len)
        shift2 = [(m - 1) * (type == 3 ? -shift : shift) for m in 1:sx1]
        shift2[end] < 0 && (shift2 .-= shift2[end])
        for m in 1:sx1
            y[m, shift2[m] .+ (1:sx2)] .= @view x[m, :]
        end
        start = 1
        while all(iszero, @view y[:, start])
            start += 1
        end
        finish = size(y, 2)
        while all(iszero, @view y[:, finish])
            finish -= 1
        end
        return y[:, start:finish]
    else
        throw(ArgumentError("resampz type must be one of {1,2,3,4}"))
    end
end

# Full 2-D convolution (small filters; used only at filter-construction time).
function _conv2_full(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where {T}
    m1, n1 = size(a); m2, n2 = size(b)
    out = zeros(T, m1 + m2 - 1, n1 + n2 - 1)
    @inbounds for j1 in 1:n1, i1 in 1:m1
        aij = a[i1, j1]
        iszero(aij) && continue
        for j2 in 1:n2, i2 in 1:m2
            out[i1 + i2 - 1, j1 + j2 - 1] += aij * b[i2, j2]
        end
    end
    return out
end

# qupz(x, 1): quincunx upsampling (zero-pad) with Q1.
function _qupz1(x::AbstractMatrix{T}) where {T}
    x1 = _resampz(x, 4)
    m, n = size(x1)
    x2 = zeros(T, 2m - 1, n)
    x2[1:2:end, :] .= x1
    return _resampz(x2, 1)
end

# ld2quin(beta): quincunx diamond filter pair from the ladder allpass β.
function _ld2quin(beta::AbstractVector{T}) where {T}
    lf = length(beta)
    n = lf ÷ 2
    sp = beta * beta'
    h = _qupz1(sp)
    h0 = copy(h)
    h0[2n, 2n] += one(T)
    h0 ./= 2
    h1 = .-_conv2_full(h, h0)
    h1[4n - 1, 4n - 1] += one(T)
    return h0, h1
end

# parafilters(f1,f2): four parallelogram filter pairs from the diamond pair.
function _parafilters(f1::AbstractMatrix{T}, f2::AbstractMatrix{T}) where {T}
    y1 = Vector{Matrix{T}}(undef, 4)
    y2 = Vector{Matrix{T}}(undef, 4)
    y1[1] = _modulate2(f1, :r); y2[1] = _modulate2(f2, :r)
    y1[2] = _modulate2(f1, :c); y2[2] = _modulate2(f2, :c)
    y1[3] = permutedims(y1[1]); y2[3] = permutedims(y2[1])
    y1[4] = permutedims(y1[2]); y2[4] = permutedims(y2[2])
    for i in 1:4
        y1[i] = _resampz(y1[i], i)
        y2[i] = _resampz(y2[i], i)
    end
    return y1, y2
end

# ── Sparse filter representation ──────────────────────────────────────────────
#
# The NSDFB filters (diamond, fan, parallelogram) are 27–33% dense; ~70% of
# entries are structural zeros.  `SparseFilter2D` stores only the non-zero
# `(di, dj, val)` triples (di = row offset from filter origin, dj = col offset),
# cutting inner-loop work by ~3× in `_efilter2!` and `_zconv2!`.
#
# It also records the l1/l2 indices (0-based) used by `_zconv2!` and the max
# radius used to split the interior (no-mod) from the boundary band in
# `_efilter2!`.

struct SparseFilter2D{Tf}
    dis::Vector{Int}   # row offsets from origin  (a - of1 for _efilter2!)
    djs::Vector{Int}   # col offsets from origin  (b - of2 for _efilter2!)
    l1s::Vector{Int}   # col-1 in filter matrix   (l1 for _zconv2!)
    l2s::Vector{Int}   # row-1 in filter matrix   (l2 for _zconv2!)
    vals::Vector{Tf}
    FRow::Int          # size(f, 2) — needed by _zconv2! for NewFRow/NewFCol
    FCol::Int          # size(f, 1) — needed by _zconv2!
    r1::Int            # max(abs.(dis)) — interior radius for _efilter2!
    r2::Int            # max(abs.(djs))
end

function SparseFilter2D(f::AbstractMatrix{Tf}) where {Tf}
    fm, fn = size(f)
    of1 = fld(fm, 2) + 1
    of2 = fld(fn, 2) + 1
    dis = Int[]; djs = Int[]
    l1s = Int[]; l2s = Int[]
    vals = Tf[]
    for b in 1:fn, a in 1:fm   # column-major scan
        v = f[a, b]
        iszero(v) && continue
        push!(dis, a - of1); push!(djs, b - of2)
        push!(l2s, a - 1);   push!(l1s, b - 1)   # 0-based l2=row-1, l1=col-1
        push!(vals, v)
    end
    r1 = isempty(dis) ? 0 : maximum(abs, dis)
    r2 = isempty(djs) ? 0 : maximum(abs, djs)
    return SparseFilter2D(dis, djs, l1s, l2s, vals, fn, fm, r1, r2)
end

Base.size(sf::SparseFilter2D) = (sf.FCol, sf.FRow)
Base.size(sf::SparseFilter2D, d::Integer) = d == 1 ? sf.FCol : d == 2 ? sf.FRow : 1

# Reconstruct the original dense matrix from a SparseFilter2D (used by GPU
# extension to copy filters to device).
function _sparse_to_dense(sf::SparseFilter2D{Tf}) where {Tf}
    f = zeros(Tf, sf.FCol, sf.FRow)
    of1 = fld(sf.FCol, 2) + 1
    of2 = fld(sf.FRow, 2) + 1
    for k in eachindex(sf.vals)
        f[sf.dis[k] + of1, sf.djs[k] + of2] = sf.vals[k]
    end
    return f
end

# ── 2-D fan/parallelogram filter bundle (analysis + synthesis) ────────────────
#
# A = SparseFilter2D{Tf} on the CPU; device-matrix type on the GPU.

struct _NSDFBFilters{A}
    k1::A            # analysis fan filters
    k2::A
    f1::Vector{A}    # analysis parallelogram filters (4 each)
    f2::Vector{A}
    gk1::A           # synthesis fan filters
    gk2::A
    gf1::Vector{A}   # synthesis parallelogram filters
    gf2::Vector{A}
end

# Move a filter bundle onto the array type `M` (identity on the CPU; the GPU
# extension overloads this to convert sparse→dense and copy to device).
_adapt_nsdfb_filters(::Type{<:AbstractMatrix}, F::_NSDFBFilters) = F

# Memoization cache: building the ladder diamond + fan + parallelogram filters
# is expensive; cache per (objectid(qfp), Tf) so repeated calls in the public
# allocating path don't rebuild.
const _NSDFB_FILTER_CACHE_LOCK = ReentrantLock()
const _NSDFB_FILTER_CACHE = Dict{Tuple{UInt, DataType}, Any}()

"""
Build the 2-D NSDFB filter bundle (fan + parallelogram, analysis + synthesis)
from the pkva ladder of `qfp`, at real filter precision `Tf`.  Equivalent to
`dfilters('pkva', ·)` + `parafilters` in the reference NSCT toolbox.
"""
function _nsdfb_filters(qfp::QuincunxFilterPair, ::Type{Tf}) where {Tf}
    is_ladder(qfp) || throw(
        ArgumentError(
            "the resampling-matrix NSDFB requires a ladder (pkva) QuincunxFilterPair"
        )
    )
    beta = Tf.(qfp.f_ladder)
    h0, h1 = _ld2quin(beta)                 # analysis diamond pair (the √2 of
    # dfilters cancels the 1/√2 of nsdfbdec)
    g0 = _modulate2(h1, :b)                 # synthesis diamond pair (dfilters 'r')
    g1 = _modulate2(h0, :b)
    k1 = _modulate2(h0, :c); k2 = _modulate2(h1, :c)
    f1, f2 = _parafilters(h0, h1)
    gk1 = _modulate2(g0, :c); gk2 = _modulate2(g1, :c)
    gf1, gf2 = _parafilters(g0, g1)
    return _NSDFBFilters(
        SparseFilter2D(k1), SparseFilter2D(k2),
        SparseFilter2D.(f1), SparseFilter2D.(f2),
        SparseFilter2D(gk1), SparseFilter2D(gk2),
        SparseFilter2D.(gf1), SparseFilter2D.(gf2)
    )
end

# Thread-safe memoized wrapper — used by the public allocating path.
# Use a local typeassert (not a function-level return annotation) so callers see a
# concrete return type without triggering a convert(T, ::Any) runtime dispatch on LTS.
function _nsdfb_filters_cached(qfp::QuincunxFilterPair, ::Type{Tf}) where {Tf}
    key = (objectid(qfp), Tf)
    result = lock(_NSDFB_FILTER_CACHE_LOCK) do
        get!(_NSDFB_FILTER_CACHE, key) do
            _nsdfb_filters(qfp, Tf)
        end
    end
    return result::_NSDFBFilters{SparseFilter2D{Tf}}
end

# ── Periodic convolution kernels (data type Td, filter type Tf) ───────────────
#
# Dense-matrix methods for `_efilter2!`/`_zconv2!` are kept for device arrays
# (overridden by `ContourletsGPUExt`).  CPU hot path dispatches to the
# `SparseFilter2D` methods below, which skip structural zeros and use
# `@turbo` (real data) or `@batch` (complex data).

# efilter2: periodic conv, filter origin floor(size/2)+1.  Dense fallback.
function _efilter2!(out::AbstractMatrix, x::AbstractMatrix, f::AbstractMatrix)
    n1, n2 = size(x)
    fm, fn = size(f)
    of1 = fld(fm, 2) + 1
    of2 = fld(fn, 2) + 1
    T = eltype(out)
    @inbounds for j in 1:n2, i in 1:n1
        s = zero(T)
        for b in 1:fn
            jj = mod1(j - (b - of2), n2)
            for a in 1:fm
                s += f[a, b] * x[mod1(i - (a - of1), n1), jj]
            end
        end
        out[i, j] = s
    end
    return out
end

# Sparse CPU method: interior uses @turbo (real) or @batch (complex).
function _efilter2!(out::AbstractMatrix, x::AbstractMatrix, sf::SparseFilter2D)
    n1, n2 = size(x)
    T = eltype(out)
    dis, djs, vals = sf.dis, sf.djs, sf.vals
    nnz = length(vals)
    r1, r2 = sf.r1, sf.r2
    i_lo = 1 + r1; i_hi = n1 - r1
    j_lo = 1 + r2; j_hi = n2 - r2

    if T <: Real && i_lo <= i_hi && j_lo <= j_hi
        # Interior: contiguous accesses — SIMD-vectorisable over i.
        # Each tap k contributes vals[k]*x[i-dis[k], j-djs[k]] with constant
        # offsets for fixed k, so the i-loop is stride-1 in x.
        @turbo for j in j_lo:j_hi, i in i_lo:i_hi
            s = zero(T)
            for k in 1:nnz
                s += vals[k] * x[i - dis[k], j - djs[k]]
            end
            out[i, j] = s
        end
        # Left column band
        @inbounds for j in 1:(j_lo - 1), i in 1:n1
            s = zero(T)
            for k in 1:nnz
                s += vals[k] * x[mod1(i - dis[k], n1), mod1(j - djs[k], n2)]
            end
            out[i, j] = s
        end
        # Right column band
        @inbounds for j in (j_hi + 1):n2, i in 1:n1
            s = zero(T)
            for k in 1:nnz
                s += vals[k] * x[mod1(i - dis[k], n1), mod1(j - djs[k], n2)]
            end
            out[i, j] = s
        end
        # Top row band (within interior column range)
        @inbounds for j in j_lo:j_hi, i in 1:(i_lo - 1)
            s = zero(T)
            for k in 1:nnz
                s += vals[k] * x[mod1(i - dis[k], n1), mod1(j - djs[k], n2)]
            end
            out[i, j] = s
        end
        # Bottom row band (within interior column range)
        @inbounds for j in j_lo:j_hi, i in (i_hi + 1):n1
            s = zero(T)
            for k in 1:nnz
                s += vals[k] * x[mod1(i - dis[k], n1), mod1(j - djs[k], n2)]
            end
            out[i, j] = s
        end
    else
        # Complex data or image too small for interior split: @batch over columns.
        @batch for j in 1:n2
            @inbounds for i in 1:n1
                s = zero(T)
                for k in 1:nnz
                    s += vals[k] * x[mod1(i - dis[k], n1), mod1(j - djs[k], n2)]
                end
                out[i, j] = s
            end
        end
    end
    return out
end

# zconv2: periodic conv with filter upsampled by integer matrix M.
# Direct port of zconv2.c (M stored column-major).  Dense fallback.
function _zconv2!(out::AbstractMatrix, x::AbstractMatrix, f::AbstractMatrix, M::NTuple{4, Int})
    SCol, SRow = size(x)
    FCol, FRow = size(f)
    M0, M1, M2, M3 = M
    NewFRow = (M0 - 1) * (FRow - 1) + M2 * (FCol - 1) + FRow - 1
    NewFCol = (M3 - 1) * (FCol - 1) + M1 * (FRow - 1) + FCol - 1
    T = eltype(out)
    mn1 = (NewFRow ÷ 2) % SRow
    mn2save = (NewFCol ÷ 2) % SCol
    @inbounds for n1 in 0:(SRow - 1)
        mn2 = mn2save
        for n2 in 0:(SCol - 1)
            outx = mn1; outy = mn2
            s = zero(T)
            for l1 in 0:(FRow - 1)
                ix = outx; iy = outy
                for l2 in 0:(FCol - 1)
                    s += x[iy + 1, ix + 1] * f[l2 + 1, l1 + 1]
                    ix -= M2
                    ix < 0 && (ix += SRow)
                    ix > SRow - 1 && (ix -= SRow)
                    iy -= M3
                    iy < 0 && (iy += SCol)
                end
                outx -= M0
                outx < 0 && (outx += SRow)
                outy -= M1
                outy < 0 && (outy += SCol)
                outy > SCol - 1 && (outy -= SCol)
            end
            out[n2 + 1, n1 + 1] = s
            mn2 += 1
            mn2 > SCol - 1 && (mn2 -= SCol)
        end
        mn1 += 1
        mn1 > SRow - 1 && (mn1 -= SRow)
    end
    return out
end

# Sparse CPU method: iterates only the nnz non-zero filter taps.
# Key optimisation over the dense path: precompute mod-reduced offsets once per
# (filter, M, image-size) call so the hot inner loop only needs one conditional
# add/subtract per tap rather than a full integer division.
#
# outx = (mn1_start + n1) in [0, 2*SRow-2] → one conditional subtraction
# outy = (mn2_start + n2) in [0, 2*SCol-2] → one conditional subtraction
# ix   = outx - mod_drow[k] ∈ [-(SRow-1), SRow-1] → one conditional add
# iy   = outy - mod_dcol[k] ∈ [-(SCol-1), SCol-1] → one conditional add
#
# @batch over n1 gives ~Ncores× speedup for complex data.
function _zconv2!(out::AbstractMatrix, x::AbstractMatrix, sf::SparseFilter2D, M::NTuple{4, Int})
    SCol, SRow = size(x)
    M0, M1, M2, M3 = M
    FRow = sf.FRow; FCol = sf.FCol
    NewFRow = (M0 - 1) * (FRow - 1) + M2 * (FCol - 1) + FRow - 1
    NewFCol = (M3 - 1) * (FCol - 1) + M1 * (FRow - 1) + FCol - 1
    T = eltype(out)
    mn1_start = (NewFRow ÷ 2) % SRow
    mn2_start = (NewFCol ÷ 2) % SCol
    l1s, l2s, vals = sf.l1s, sf.l2s, sf.vals
    nnz = length(vals)
    # Precompute mod-reduced offsets: only nnz divisions (not SRow*SCol*nnz).
    mod_drows = [mod(l1s[k] * M0 + l2s[k] * M2, SRow) for k in 1:nnz]
    mod_dcols = [mod(l1s[k] * M1 + l2s[k] * M3, SCol) for k in 1:nnz]
    @batch for n1 in 0:(SRow - 1)
        outx = mn1_start + n1
        outx >= SRow && (outx -= SRow)
        @inbounds for n2 in 0:(SCol - 1)
            outy = mn2_start + n2
            outy >= SCol && (outy -= SCol)
            s = zero(T)
            for k in 1:nnz
                ix = outx - mod_drows[k]
                ix < 0 && (ix += SRow)
                iy = outy - mod_dcols[k]
                iy < 0 && (iy += SCol)
                s += x[iy + 1, ix + 1] * vals[k]
            end
            out[n2 + 1, n1 + 1] = s
        end
    end
    return out
end

_efilter2(x::AbstractMatrix, f) =
    _efilter2!(similar(x, _data_eltype(x)), x, f)
_zconv2(x::AbstractMatrix, f, M::NTuple{4, Int}) =
    _zconv2!(similar(x, _data_eltype(x)), x, f, M)

# Two-channel nonsubsampled FB: split / merge along an upsampling matrix.
# `f1`/`f2` may be `SparseFilter2D` (CPU) or device matrices (GPU).
function _nssfb_split(x::AbstractMatrix, f1, f2, M)
    if M === nothing
        return _efilter2(x, f1), _efilter2(x, f2)
    else
        return _zconv2(x, f1, M), _zconv2(x, f2, M)
    end
end

function _nssfb_merge(x1::AbstractMatrix, x2::AbstractMatrix, f1, f2, M)
    if M === nothing
        return _efilter2(x1, f1) .+ _efilter2(x2, f2)
    else
        return _zconv2(x1, f1, M) .+ _zconv2(x2, f2, M)
    end
end

# Upsampling matrices Mₖ for level l (Do thesis eq. 3.18), as NTuple{4,Int}.
function _mkl_first(k::Int, l::Int)
    slk = 2 * fld(k - 1, 2) - 2^(l - 3) + 1
    # 2*[2^(l-3) 0; 0 1]*[1 0; -slk 1] = [2^(l-2) 0; -2 slk  2]
    p = 2^(l - 2)
    return (p, -2 * slk, 0, 2)                 # (M0, M1, M2, M3) col-major
end
function _mkl_second(k::Int, l::Int)
    slk = 2 * fld(k - 2^(l - 2) - 1, 2) - 2^(l - 3) + 1
    # 2*[1 0; 0 2^(l-3)]*[1 -slk; 0 1] = [2 0; -2 slk... ] handled below
    p = 2^(l - 2)
    return (2, 0, -2 * slk, p)
end

const _Q1 = (1, 1, -1, 1)   # Q1 = [1 -1; 1 1] col-major (M0=1,M1=1,M2=-1,M3=1)

# ── Binary-tree analysis / synthesis ─────────────────────────────────────────

function _nsdfb_tree_decompose(x::AbstractMatrix, F::_NSDFBFilters, clevels::Int)
    clevels == 0 && return [copy(x)]
    if clevels == 1
        a, b = _nssfb_split(x, F.k1, F.k2, nothing)
        return [a, b]
    end
    x1, x2 = _nssfb_split(x, F.k1, F.k2, nothing)
    y = Vector{typeof(x1)}(undef, 4)
    y[1], y[2] = _nssfb_split(x1, F.k1, F.k2, _Q1)
    y[3], y[4] = _nssfb_split(x2, F.k1, F.k2, _Q1)
    for l in 3:clevels
        yold = y
        y = Vector{typeof(x1)}(undef, 2^l)
        for k in 1:(2^(l - 2))
            M = _mkl_first(k, l)
            i = mod(k - 1, 2) + 1
            y[2k - 1], y[2k] = _nssfb_split(yold[k], F.f1[i], F.f2[i], M)
        end
        for k in (2^(l - 2) + 1):(2^(l - 1))
            M = _mkl_second(k, l)
            i = mod(k - 1, 2) + 3
            y[2k - 1], y[2k] = _nssfb_split(yold[k], F.f1[i], F.f2[i], M)
        end
    end
    return y
end

function _nsdfb_tree_reconstruct(subbands::Vector{<:AbstractMatrix}, F::_NSDFBFilters)
    n = length(subbands)
    clevels = round(Int, log2(n))
    clevels == 0 && return copy(subbands[1])
    clevels == 1 && return _nssfb_merge(subbands[1], subbands[2], F.gk1, F.gk2, nothing)
    x = collect(subbands)
    for l in clevels:-1:3
        for k in 1:(2^(l - 2))
            M = _mkl_first(k, l)
            i = mod(k - 1, 2) + 1
            x[k] = _nssfb_merge(x[2k - 1], x[2k], F.gf1[i], F.gf2[i], M)
        end
        for k in (2^(l - 2) + 1):(2^(l - 1))
            M = _mkl_second(k, l)
            i = mod(k - 1, 2) + 3
            x[k] = _nssfb_merge(x[2k - 1], x[2k], F.gf1[i], F.gf2[i], M)
        end
    end
    x[1] = _nssfb_merge(x[1], x[2], F.gk1, F.gk2, _Q1)
    x[2] = _nssfb_merge(x[3], x[4], F.gk1, F.gk2, _Q1)
    return _nssfb_merge(x[1], x[2], F.gk1, F.gk2, nothing)
end

# ── Public API ───────────────────────────────────────────────────────────────

"""
    nsdfb_decompose(bandpass, l_levels, qfp, tree_level) -> Vector{Matrix}

Nonsubsampled DFB analysis (resampling-matrix / fan-filter construction).  Returns
`2^l_levels` directional subbands, each the same size as `bandpass`, each occupying
an angular wedge of the frequency plane.  `tree_level` is accepted for API symmetry
but does not affect the (scale-independent) directional filters.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(5), 16, 16);

julia> sbs = nsdfb_decompose(x, 2, Q2345, 1);

julia> length(sbs), size(sbs[1]) == size(x)
(4, true)
```
"""
function nsdfb_decompose(
        bandpass::AbstractMatrix, l_levels::Int,
        qfp::QuincunxFilterPair, tree_level::Int = 1;
        threading::ThreadingPolicy = Auto()
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    tree_level >= 1 || throw(ArgumentError("tree_level must be ≥ 1"))
    l_levels == 0 && return [copy(bandpass)]
    Td = _data_eltype(bandpass)
    Tf = _filter_eltype(Td)
    img = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    F = _nsdfb_filters_cached(qfp, Tf)
    return _nsdfb_tree_decompose(img, F, l_levels)
end

"""
    nsdfb_reconstruct(subbands, qfp, tree_level) -> bandpass

Nonsubsampled DFB synthesis (inverse of [`nsdfb_decompose`](@ref)).

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(5), 16, 16);

julia> sbs = nsdfb_decompose(x, 2, Q2345, 1);

julia> rec = nsdfb_reconstruct(sbs, Q2345, 1);

julia> maximum(abs, rec .- x) < 1e-12
true
```
"""
function nsdfb_reconstruct(
        subbands::Vector{<:AbstractMatrix},
        qfp::QuincunxFilterPair, tree_level::Int = 1;
        threading::ThreadingPolicy = Auto()
    )
    n = length(subbands)
    ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
    n == 1 && return copy(subbands[1])
    Td = _data_eltype(subbands[1])
    Tf = _filter_eltype(Td)
    F = _nsdfb_filters_cached(qfp, Tf)
    return _nsdfb_tree_reconstruct(subbands, F)
end

# ── Allocation-free companions (used by the no-FFT workspace / GPU fallback) ───

function _nsdfb_decompose_into!(
        out::Vector{<:AbstractMatrix}, bandpass::AbstractMatrix,
        l_levels::Int, F::_NSDFBFilters, arena
    )
    l_levels == 0 && return (copyto!(out[1], bandpass); out)
    sbs = _nsdfb_tree_decompose(bandpass, F, l_levels)
    for k in eachindex(sbs)
        copyto!(out[k], sbs[k])
    end
    return out
end

function _nsdfb_reconstruct_into!(
        dest::AbstractMatrix, subbands::Vector{<:AbstractMatrix},
        F::_NSDFBFilters, arena
    )
    n = length(subbands)
    n == 1 && return (copyto!(dest, subbands[1]); dest)
    copyto!(dest, _nsdfb_tree_reconstruct(subbands, F))
    return dest
end

# ── FFT-domain equivalent filters (for the workspace fast path) ───────────────
#
# The whole NSDFB tree is linear and circular, so each leaf is a circular
# convolution with an equivalent filter = the tree's impulse response.  We build
# those impulse responses with the spatial tree, then FFT them.  This guarantees
# the FFT workspace path reproduces the spatial/MATLAB path bit-for-bit.

function _build_equivalent_filters(::Type{Tf}, n1::Int, n2::Int, F::_NSDFBFilters, L::Int, is_real::Bool) where {Tf}
    _fft(g) = is_real ? rfft(g) : fft(complex(g))

    # Analysis: one tree pass on a delta yields all 2^L equivalent filters.
    delta = zeros(Tf, n1, n2); delta[1, 1] = one(Tf)
    Hsp = _nsdfb_tree_decompose(delta, F, L)
    H_leaves = [Matrix{Complex{Tf}}(_fft(h)) for h in Hsp]

    # Synthesis: equivalent filter for leaf k = reconstruction of a delta in slot k.
    nleaves = 2^L
    G_leaves = Vector{Matrix{Complex{Tf}}}(undef, nleaves)
    for k in 1:nleaves
        ek = [zeros(Tf, n1, n2) for _ in 1:nleaves]
        ek[k][1, 1] = one(Tf)
        gsp = _nsdfb_tree_reconstruct(ek, F)
        G_leaves[k] = Matrix{Complex{Tf}}(_fft(gsp))
    end
    return H_leaves, G_leaves
end
