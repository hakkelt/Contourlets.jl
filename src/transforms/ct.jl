# Discrete Contourlet Transform (CT) — forward and inverse.
#
# Forward (from the finest scale, J = scales):
#   coarse ← image
#   for j = J … 1:
#       coarse_j, bandpass_j = lp_decompose(coarse, lp_filters)
#       subbands of scale j = dfb_decompose(bandpass_j, dfb_filters, direction_levels[j])
#       coarse = coarse_j
#
# Inverse (runs j = 1 … J):
#       bandpass_j = dfb_reconstruct(subbands of scale j, dfb_filters)
#       coarse = lp_reconstruct(coarse, bandpass_j, lp_filters)
#
# The coefficients are a `SubbandVector`: the lowpass, then the subbands of
# scale 1 (the coarsest), 2, …, J (`subband_layout`).  The directional filter banks write
# straight into its subbands, which are reshaped views of its storage.
#
# Reference: M. N. Do & M. Vetterli, "The contourlet transform: an efficient
# directional multiresolution image representation", IEEE TIP 14(12), 2005 (the
# pyramidal directional filter bank).

# The subband layout of a CT on a `dims` image, and the index of the first
# subband of each scale.
# Whether an image axis of length n fits scale j of t: the bandpass there (n
# halved, rounded up, once per finer scale) must split into its decimated
# directional bank (`_check_bank_dims`).
function _ct_axis_fits(t::CT, n::Int, j::Int)
    for _ in (j + 1):t.scales
        n = cld(n, 2)
    end
    l = t.direction_levels[j]
    return l == 0 || n % 2^max(l - 1, 1) == 0
end

# Throws, naming the scale, the axis and the nearest length that fits, unless
# every scale's bandpass of an image of size `dims` suits its bank.
function _check_ct_dims(t::CT, dims::NTuple{2, Int})
    for a in 1:2, j in 1:t.scales
        n = dims[a]
        _ct_axis_fits(t, n, j) && continue
        fits(m) = all(k -> _ct_axis_fits(t, m, k), 1:t.scales)
        near = 0
        for d in 1:(4n + 64), m in (n - d, n + d)
            m >= 1 && fits(m) && (near = m; break)
        end
        l = t.direction_levels[j]
        bp = n
        for _ in (j + 1):t.scales
            bp = cld(bp, 2)
        end
        throw(ArgumentError("an image of size $(dims[1]) × $(dims[2]) does not fit: at scale $(j) of $(t.scales) (1 the coarsest) axis $(a) has bandpass length $(bp), which its $(l)-level directional filter bank needs divisible by $(2^max(l - 1, 1)); the nearest image length that fits is $(near)"))
    end
    return dims
end

function _layout(t::CT, dims::NTuple{2, Int})
    _check_ct_dims(t, dims)
    n1, n2 = dims
    levels = Vector{Vector{NTuple{2, Int}}}(undef, t.scales)
    for j in t.scales:-1:1          # the bandpass halves from the finest scale on
        levels[j] = dfb_subband_sizes((n1, n2), t.direction_levels[j])
        n1, n2 = cld(n1, 2), cld(n2, 2)
    end
    return _stack_levels((n1, n2), levels, dims)
end

# The layout of a lowpass of size `coarse` followed by the subbands of `groups`,
# and the index of each group's first subband.
function _stack_levels(coarse::NTuple{2, Int}, groups::Vector{Vector{NTuple{2, Int}}}, dims::NTuple{2, Int})
    sizes = [coarse]
    first = Vector{Int}(undef, length(groups))
    for (g, s) in enumerate(groups)
        first[g] = length(sizes) + 1
        append!(sizes, s)
    end
    return SubbandLayout(sizes, dims), first
end

# Without a workspace: the pyramid convolves, and each level builds its DFB.
function _analyze_direct!(c::SubbandVector, first::Vector{Int}, t::CT, image::AbstractMatrix)
    Td = eltype(c)
    Tf = filter_eltype(Td)
    p = _convert_transform(Tf, t)
    fp, qfp = p.lp_filters, p.dfb_filters
    coarse = Td === eltype(image) ? image : Td.(image)
    for j in p.scales:-1:1
        coarse_j, bp_j = lp_decompose(coarse, fp)
        l = p.direction_levels[j]
        dfb_decompose!(_Bands(c, first[j], 1 << l), bp_j, qfp, l)
        coarse = coarse_j
    end
    copyto!(subband(c, 1), coarse)
    return c
end

function _synthesize_direct!(x::AbstractMatrix, first::Vector{Int}, t::CT, c::SubbandVector)
    Tf = filter_eltype(eltype(c))
    p = _convert_transform(Tf, t)
    fp, qfp = p.lp_filters, p.dfb_filters
    coarse = copy(subband(c, 1))
    for j in 1:p.scales
        bp = dfb_reconstruct(_Bands(c, first[j], 1 << p.direction_levels[j]), qfp)
        coarse = lp_reconstruct(coarse, bp, fp)
    end
    return copyto!(x, coarse)
end

# Typed dispatch barrier: the workspace keyword is untyped.  `capture` selects the
# seam that `ContourletsCUDAExt` overloads to capture device graphs.
function _ragged_forward!(c::SubbandVector, image::AbstractMatrix, ws::CTWorkspace, capture::Bool)
    T = eltype(ws)
    img = T === eltype(image) ? image : T.(image)
    capture ? _ct_forward_ws!(c, img, ws) : _ct_forward_inner!(c, img, ws)
    return c
end

function _ragged_inverse!(x::AbstractMatrix, c::SubbandVector, ws::CTWorkspace, capture::Bool)
    capture ? _ct_inverse_ws!(x, c, ws) : _ct_inverse_inner!(x, c, ws)
    return x
end

# Internal dispatch seams — overloaded by ContourletsCUDAExt for graph capture.
function _ct_forward_ws!(c::SubbandVector{T}, image::AbstractMatrix, ws::CTWorkspace{T}) where {T}
    _ct_forward_inner!(c, image, ws)
    return c
end

function _ct_inverse_ws!(image::AbstractMatrix{T}, c::SubbandVector{T}, ws::CTWorkspace{T}) where {T}
    _ct_inverse_inner!(image, c, ws)
    return image
end

function _ct_forward_inner!(c::SubbandVector{T}, image::AbstractMatrix, ws::CTWorkspace{T}) where {T}
    p = ws.params
    L = p.direction_levels
    fp, qfp = p.lp_filters, p.dfb_filters
    img = T === eltype(image) ? image : T.(image)
    _arena_reset!(ws.fwd_scratch)
    return _with_arena(ws.fwd_scratch) do
        current = img
        for j in p.scales:-1:1
            n1, n2 = size(current)
            coarse_j = _scratch_like(current, cld(n1, 2), cld(n2, 2))
            bp_j = _scratch_like(current, n1, n2)
            tmp_j = _scratch_like(current, n1, n2)
            tmp2_j = _scratch_like(current, n1, n2)

            lp_decompose!(coarse_j, bp_j, current, fp; tmp = tmp_j, tmp2 = tmp2_j)
            # The workspace runner holds this level's filters, buffers and plans.
            dfb_decompose!(_Bands(c, ws.first[j], 1 << L[j]), bp_j, qfp, L[j]; workspace = ws)
            current = coarse_j
        end
        copyto!(subband(c, 1), current)
    end
end

function _ct_inverse_inner!(image::AbstractMatrix{T}, c::SubbandVector{T}, ws::CTWorkspace{T}) where {T}
    p = ws.params
    L = p.direction_levels
    fp, qfp = p.lp_filters, p.dfb_filters

    _arena_reset!(ws.inv_scratch)
    _with_arena(ws.inv_scratch) do
        lowpass = subband(c, 1)
        current = _scratch_like(image, size(lowpass, 1), size(lowpass, 2))
        copyto!(current, lowpass)

        for j in 1:p.scales
            n1, n2 = ws.bp_dims[j]
            bp = _scratch_like(current, n1, n2)
            dfb_reconstruct!(bp, _Bands(c, ws.first[j], 1 << L[j]), qfp; workspace = ws)
            tmp_j = _scratch_like(current, n1, n2)
            tmp2_j = _scratch_like(current, n1, n2)
            if j < p.scales
                next_coarse = _scratch_like(current, n1, n2)
                lp_reconstruct!(next_coarse, current, bp, fp; tmp = tmp_j, tmp2 = tmp2_j)
                current = next_coarse
            else
                lp_reconstruct!(image, current, bp, fp; tmp = tmp_j, tmp2 = tmp2_j)
            end
        end
    end
    return image
end

# Grow the workspace scratch arena to its steady-state size by running one forward
# + inverse pass on zeros, so the first real call allocates nothing.
function _prewarm_ct!(ws::CTWorkspace{Td, Tf, M}) where {Td, Tf, M}
    img = _allocate_zeros(M, Td, ws.image_size)
    c = SubbandVector(fill!(similar(img, Td, length(ws.layout)), zero(Td)), ws.layout)
    _ct_forward_inner!(c, img, ws)
    _ct_inverse_inner!(img, c, ws)
    return ws
end
