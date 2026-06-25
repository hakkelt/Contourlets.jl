# GPU port of the decimated DFB scalar-loop primitives.
#
# The decimated DFB tree (`_dfbdec_l` / `_dfbrec_l`, the polyphase splits, the
# lifting `_fbdec_l`, backsampling) is built from array slicing, circular rolls,
# broadcasts and just two scalar-loop kernels: the periodic resampling `_resamp`
# and the separable periodic filter `_sefilter2`.  Providing device methods for
# those two keeps the whole tree on the GPU (slicing / circshift / broadcast all
# run on device), so `dfb_decompose` / `dfb_reconstruct` work on GPU arrays.

# ── Periodic resampling (`_resamp`) ───────────────────────────────────────────
# Resampling matrices R1..R4 — periodic shear along rows (1,2) or columns (3,4).

@kernel function _resamp_kernel!(y, @Const(x), type::Int, shift::Int, m::Int, n::Int)
    i, j = @index(Global, NTuple)
    if type == 1
        @inbounds y[i, j] = x[mod1(i + shift * (j - 1), m), j]
    elseif type == 2
        @inbounds y[i, j] = x[mod1(i - shift * (j - 1), m), j]
    elseif type == 3
        @inbounds y[i, j] = x[i, mod1(j + shift * (i - 1), n)]
    else
        @inbounds y[i, j] = x[i, mod1(j - shift * (i - 1), n)]
    end
end

function Contourlets._resamp(x::_AbstractGPUMatrix{T}, type::Int, shift::Int = 1) where {T}
    (1 <= type <= 4) || throw(ArgumentError("resamp type must be 1..4"))
    m, n = size(x)
    backend = _gpu_backend(x)
    y = _scratch_like(x, m, n)
    _resamp_kernel!(backend, (16, 16))(y, x, type, shift, m, n; ndrange = (m, n))
    return y
end

# ── Separable periodic filtering (`_sefilter2`) ───────────────────────────────
# Mirrors the CPU two-pass valid convolution over the periodically extended
# array, with the boundary extension materialised by a kernel (modes :per = 1,
# :qper_col = 2).

# Fused pass 1: boundary extension is folded directly into the column filter so
# the (m+ru+rd, n+cl+cr) `ext` scratch array and its kernel launch are eliminated.
# Output `tmp` is (m, ncols); the column dimension is still extended (consumed by
# pass 2), but the row boundary is resolved per-tap from `x` (mirrors the inline
# indexing the NSDFB `_efilter2_kernel!` already uses).
@kernel function _sef_pass1_fused_kernel!(
        tmp, @Const(x), @Const(f),
        L::Int, ru::Int, cl::Int, m::Int, n::Int, mode::Int, n2::Int
    )
    i, j = @index(Global, NTuple)        # i ∈ 1:m, j ∈ 1:ncols
    T = eltype(tmp)
    acc = zero(T)
    if mode == 1                          # :per
        jj = mod1(j - cl, n)
        @inbounds for a in 1:L
            ii = mod1((i + L - a) - ru, m)
            acc += f[a] * x[ii, jj]
        end
    else                                  # :qper_col
        jjr = mod1(j - cl, n)
        @inbounds for a in 1:L
            iir = (i + L - a) - ru
            if iir < 1 || iir > m
                acc += f[a] * x[mod1(iir, m), mod1(jjr + n2, n)]
            else
                acc += f[a] * x[iir, jjr]
            end
        end
    end
    @inbounds tmp[i, j] = acc
end

@kernel function _sef_pass2_kernel!(out, @Const(tmp), @Const(f), L::Int)
    i, j = @index(Global, NTuple)
    T = eltype(out)
    acc = zero(T)
    @inbounds for b in 1:L
        acc += f[b] * tmp[i, j + L - b]
    end
    @inbounds out[i, j] = acc
end

# ── Public DFB entry points (GPU dispatch) ────────────────────────────────────
# The ladder path's filter must be device-resident before reaching `_sefilter2`;
# these overloads upload once at the API boundary when no cached filter is supplied.

function Contourlets.dfb_decompose(
        bandpass::_AbstractGPUMatrix, l_levels::Int, qfp::Contourlets.QuincunxFilterPair;
        filter = nothing, threading::Contourlets.ThreadingPolicy = Contourlets.Auto()
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    l_levels == 0 && return [copy(bandpass)]
    Td = Contourlets._data_eltype(bandpass)
    Tf = Contourlets._filter_eltype(Td)
    img = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    if Contourlets.is_ladder(qfp)
        f = filter !== nothing ? filter :
            _to_device(_gpu_backend(bandpass), Contourlets._ladder_modulate(Tf.(qfp.f_ladder)))
        return Contourlets._dfbdec_l(img, f, l_levels, false)
    end
    return Contourlets._dfb_split(img, l_levels, 1, Contourlets._convert_qfp(Tf, qfp), false)
end

function Contourlets.dfb_reconstruct(
        subbands::Vector{<:_AbstractGPUMatrix}, qfp::Contourlets.QuincunxFilterPair;
        filter = nothing, threading::Contourlets.ThreadingPolicy = Contourlets.Auto()
    )
    n = length(subbands)
    n >= 1   || throw(ArgumentError("subbands must be non-empty"))
    ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
    n == 1 && return copy(subbands[1])
    Td = Contourlets._data_eltype(subbands[1])
    Tf = Contourlets._filter_eltype(Td)
    if Contourlets.is_ladder(qfp)
        sbs = eltype(subbands[1]) === Td ? subbands : [Td.(s) for s in subbands]
        f = filter !== nothing ? filter :
            _to_device(_gpu_backend(subbands[1]), Contourlets._ladder_modulate(Tf.(qfp.f_ladder)))
        return Contourlets._dfbrec_l(sbs, f, false)
    end
    l_levels = round(Int, log2(n))
    return Contourlets._dfb_merge(subbands, l_levels, 1, Contourlets._convert_qfp(Tf, qfp), false)
end

# ── Separable periodic filtering (`_sefilter2`) ───────────────────────────────

function Contourlets._sefilter2(
        x::_AbstractGPUMatrix{Td}, f::AbstractVector{Tf}, shift1::Int, shift2::Int, extmod::Symbol, threaded::Bool = false;
        kwargs...
    ) where {Td, Tf}
    backend = _gpu_backend(x)
    L = length(f)
    lf = (L - 1) / 2
    ru = floor(Int, lf) + shift1
    cl = floor(Int, lf) + shift2
    cr = ceil(Int, lf) - shift2
    m, n = size(x)
    mode = if extmod === :per
        1
    elseif extmod === :qper_col
        2
    else
        throw(ArgumentError("unsupported extmod: $extmod"))
    end
    n2 = round(Int, n / 2)

    ncols = n + cl + cr
    tmp = _scratch_like(x, m, ncols)
    _sef_pass1_fused_kernel!(backend, (16, 16))(tmp, x, f, L, ru, cl, m, n, mode, n2; ndrange = (m, ncols))

    out = _scratch_like(x, m, n)
    _sef_pass2_kernel!(backend, (16, 16))(out, tmp, f, L; ndrange = (m, n))
    return out
end
