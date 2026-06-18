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
    y = similar(x)
    _resamp_kernel!(backend, (16, 16))(y, x, type, shift, m, n; ndrange = (m, n))
    return y
end

# ── Separable periodic filtering (`_sefilter2`) ───────────────────────────────
# Mirrors the CPU two-pass valid convolution over the periodically extended
# array, with the boundary extension materialised by a kernel (modes :per = 1,
# :qper_row = 2, :qper_col = 3).

@kernel function _extend2_kernel!(
        out, @Const(x), ru::Int, cl::Int, m::Int, n::Int, mode::Int, m2::Int, n2::Int
    )
    I, J = @index(Global, NTuple)
    if mode == 1                      # :per
        @inbounds out[I, J] = x[mod1(I - ru, m), mod1(J - cl, n)]
    elseif mode == 2                  # :qper_row
        ii = mod1(I - ru, m)
        jj = J - cl
        if jj < 1 || jj > n
            @inbounds out[I, J] = x[mod1(ii + m2, m), mod1(jj, n)]
        else
            @inbounds out[I, J] = x[ii, jj]
        end
    else                              # :qper_col
        ii = I - ru
        jj = mod1(J - cl, n)
        if ii < 1 || ii > m
            @inbounds out[I, J] = x[mod1(ii, m), mod1(jj + n2, n)]
        else
            @inbounds out[I, J] = x[ii, jj]
        end
    end
end

@kernel function _sef_pass1_kernel!(tmp, @Const(ext), @Const(f), L::Int)
    i, j = @index(Global, NTuple)
    T = eltype(tmp)
    acc = zero(T)
    @inbounds for a in 1:L
        acc += f[a] * ext[i + L - a, j]
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

function Contourlets._sefilter2(
        x::_AbstractGPUMatrix{Td}, f::Vector{Tf}, shift1::Int, shift2::Int, extmod::Symbol
    ) where {Td, Tf}
    backend = _gpu_backend(x)
    L = length(f)
    lf = (L - 1) / 2
    ru = floor(Int, lf) + shift1
    rd = ceil(Int, lf) - shift1
    cl = floor(Int, lf) + shift2
    cr = ceil(Int, lf) - shift2
    m, n = size(x)
    mode = if extmod === :per
        1
    elseif extmod === :qper_row
        2
    elseif extmod === :qper_col
        3
    else
        throw(ArgumentError("unsupported extmod: $extmod"))
    end
    m2 = round(Int, m / 2)
    n2 = round(Int, n / 2)
    f_d = _ensure_gpu(backend, f)

    ext = KernelAbstractions.allocate(backend, Td, m + ru + rd, n + cl + cr)
    _extend2_kernel!(backend, (16, 16))(ext, x, ru, cl, m, n, mode, m2, n2; ndrange = size(ext))

    ncols = n + cl + cr
    tmp = KernelAbstractions.allocate(backend, Td, m, ncols)
    _sef_pass1_kernel!(backend, (16, 16))(tmp, ext, f_d, L; ndrange = (m, ncols))

    out = KernelAbstractions.allocate(backend, Td, m, n)
    _sef_pass2_kernel!(backend, (16, 16))(out, tmp, f_d, L; ndrange = (m, n))
    return out
end
