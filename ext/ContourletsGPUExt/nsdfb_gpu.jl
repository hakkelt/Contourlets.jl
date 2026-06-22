# GPU port of the Nonsubsampled Directional Filter Bank (NSDFB).
#
# The NSDFB is the resampling-matrix (fan + parallelogram filter) construction.
# Its two periodic-convolution primitives — `_efilter2!` (filter centred at
# floor(size/2)+1) and `_zconv2!` (filter upsampled by an integer matrix) — are
# per-output-pixel reductions, so each maps onto a KernelAbstractions kernel.
# The host-side binary tree (`_nsdfb_tree_decompose` / `_nsdfb_tree_reconstruct`)
# is array-type generic, so providing device methods for these two primitives
# keeps the whole tree on the device.  The filter bundle is moved to the device
# once via `_adapt_nsdfb_filters`.  Each thread reproduces the CPU reduction, so
# the device result matches the host path.

# ── Kernels ───────────────────────────────────────────────────────────────────

@kernel function _efilter2_kernel!(out, @Const(x), @Const(f), of1::Int, of2::Int, fm::Int, fn::Int, n1::Int, n2::Int)
    i, j = @index(Global, NTuple)
    T = eltype(out)
    s = zero(T)
    @inbounds for b in 1:fn
        jj = mod1(j - (b - of2), n2)
        for a in 1:fm
            s += f[a, b] * x[mod1(i - (a - of1), n1), jj]
        end
    end
    @inbounds out[i, j] = s
end

@kernel function _zconv2_kernel!(
        out, @Const(x), @Const(f),
        M0::Int, M1::Int, M2::Int, M3::Int,
        Start1::Int, Start2::Int, FRow::Int, FCol::Int, SRow::Int, SCol::Int
    )
    # Julia index (row, col) = (n2+1, n1+1) in the C port's naming.
    r, c = @index(Global, NTuple)
    n2 = r - 1            # row index 0-based  (C: n2, SCol)
    n1 = c - 1            # col index 0-based  (C: n1, SRow)
    T = eltype(out)
    outx = mod(Start1 + n1, SRow)
    outy = mod(Start2 + n2, SCol)
    s = zero(T)
    @inbounds for l1 in 0:(FRow - 1)
        for l2 in 0:(FCol - 1)
            ix = mod(outx - l1 * M0 - l2 * M2, SRow)
            iy = mod(outy - l1 * M1 - l2 * M3, SCol)
            s += x[iy + 1, ix + 1] * f[l2 + 1, l1 + 1]
        end
    end
    @inbounds out[r, c] = s
end

# ── Device primitives (dispatched by the generic tree) ────────────────────────

function Contourlets._efilter2!(out::_AbstractGPUMatrix, x::_AbstractGPUMatrix, f::_AbstractGPUMatrix)
    backend = _gpu_backend(out)
    n1, n2 = size(x)
    fm, fn = size(f)
    kernel = _efilter2_kernel!(backend, (16, 16))
    kernel(out, x, f, fld(fm, 2) + 1, fld(fn, 2) + 1, fm, fn, n1, n2; ndrange = (n1, n2))
    return out
end

function Contourlets._zconv2!(out::_AbstractGPUMatrix, x::_AbstractGPUMatrix, f::_AbstractGPUMatrix, M::NTuple{4, Int})
    backend = _gpu_backend(out)
    SCol, SRow = size(x)
    FCol, FRow = size(f)
    M0, M1, M2, M3 = M
    NewFRow = (M0 - 1) * (FRow - 1) + M2 * (FCol - 1) + FRow - 1
    NewFCol = (M3 - 1) * (FCol - 1) + M1 * (FRow - 1) + FCol - 1
    kernel = _zconv2_kernel!(backend, (16, 16))
    kernel(out, x, f, M0, M1, M2, M3, NewFRow ÷ 2, NewFCol ÷ 2, FRow, FCol, SRow, SCol; ndrange = (SCol, SRow))
    return out
end

# Allocating wrappers must produce device output for device input.
Contourlets._efilter2(x::_AbstractGPUMatrix, f::_AbstractGPUMatrix) =
    Contourlets._efilter2!(similar(x, Contourlets._data_eltype(x)), x, f)
Contourlets._zconv2(x::_AbstractGPUMatrix, f::_AbstractGPUMatrix, M::NTuple{4, Int}) =
    Contourlets._zconv2!(similar(x, Contourlets._data_eltype(x)), x, f, M)

# ── Move the filter bundle onto the device ────────────────────────────────────

function Contourlets._adapt_nsdfb_filters(::Type{M}, F::Contourlets._NSDFBFilters) where {M <: AbstractGPUMatrix}
    backend = KernelAbstractions.get_backend(similar(M, (1, 1)))
    # Convert SparseFilter2D back to dense matrix before uploading to device.
    d(x::Contourlets.SparseFilter2D) = _to_device(backend, Contourlets._sparse_to_dense(x))
    d(x::AbstractMatrix) = _to_device(backend, x)
    dv(v) = [d(x) for x in v]
    return Contourlets._NSDFBFilters(d(F.k1), d(F.k2), dv(F.f1), dv(F.f2), d(F.gk1), d(F.gk2), dv(F.gf1), dv(F.gf2))
end

# The workspace caches one bundle per LP level; move each to the device.
function Contourlets._device_qup_cache(::Type{M}, qup_cache) where {M <: AbstractGPUMatrix}
    return [Contourlets._adapt_nsdfb_filters(M, F) for F in qup_cache]
end

function Contourlets._device_lp_cache(::Type{M}, lp_cache) where {M <: AbstractGPUMatrix}
    backend = KernelAbstractions.get_backend(similar(M, (1, 1)))
    return [_to_device(backend, v) for v in lp_cache]
end

# ── Public NSDFB entry points (device dispatch) ───────────────────────────────

function Contourlets.nsdfb_decompose(
        bandpass::_AbstractGPUMatrix, l_levels::Int,
        qfp::QuincunxFilterPair, tree_level::Int = 1;
        threading::Contourlets.ThreadingPolicy = Contourlets.Auto()
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    tree_level >= 1 || throw(ArgumentError("tree_level must be ≥ 1"))
    l_levels == 0 && return [copy(bandpass)]
    Td = Contourlets._data_eltype(bandpass)
    Tf = Contourlets._filter_eltype(Td)
    img = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    F = Contourlets._adapt_nsdfb_filters(typeof(bandpass), Contourlets._nsdfb_filters(qfp, Tf))
    return Contourlets._nsdfb_tree_decompose(img, F, l_levels)
end

function Contourlets.nsdfb_reconstruct(
        subbands::Vector{<:_AbstractGPUMatrix},
        qfp::QuincunxFilterPair, tree_level::Int = 1;
        threading::Contourlets.ThreadingPolicy = Contourlets.Auto()
    )
    n = length(subbands)
    ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
    n == 1 && return copy(subbands[1])
    Td = Contourlets._data_eltype(subbands[1])
    Tf = Contourlets._filter_eltype(Td)
    F = Contourlets._adapt_nsdfb_filters(typeof(subbands[1]), Contourlets._nsdfb_filters(qfp, Tf))
    return Contourlets._nsdfb_tree_reconstruct(subbands, F)
end
