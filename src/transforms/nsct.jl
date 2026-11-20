# Nonsubsampled Contourlet Transform (NSCT) — forward and inverse.
#
# Same structure as the CT but with:
#   • lp_decompose → nsp_decompose (no downsampling, à trous upsampled filters)
#   • dfb_decompose → nsdfb_decompose (no quincunx decimation)
# All outputs have the same spatial size as the input, so the coefficients are
# one array of size (n1, n2, n): the lowpass, then the subbands of level 1, 2, …, J.
# A workspace runs the whole transform as masks on the image's spectrum
# (`NSCTWorkspace`, the mask-bank engine); without one the pyramid convolves.
#
# Reference: A. L. da Cunha, J. Zhou & M. N. Do, "The nonsubsampled contourlet
# transform: theory, design, and applications", IEEE TIP 15(10), 2006, §II-A.

# Subband (j, k) is slice `offsets[j] + k` of the coefficients.
function _nsct_offsets(t::NSCT)
    offsets = Vector{Int}(undef, t.J)
    s = 1
    for j in 1:t.J
        offsets[j] = s
        s += 1 << t.L_array[j]
    end
    return offsets
end

function _analyze_direct!(c::AbstractArray, t::NSCT, image::AbstractMatrix, threaded)
    Td = eltype(c)
    Tf = filter_eltype(Td)
    p = _convert_transform(Tf, t)
    fp, qfp = p.lp_filters, p.dfb_filters
    offsets = _nsct_offsets(p)
    coarse = Td === eltype(image) ? image : Td.(image)
    for j in 1:p.J
        coarse_j, bp_j = nsp_decompose(coarse, fp, j)
        for (k, sb) in enumerate(nsdfb_decompose(bp_j, p.L_array[j], qfp, j))
            copyto!(view(c, :, :, offsets[j] + k), sb)
        end
        coarse = coarse_j
    end
    copyto!(view(c, :, :, 1), coarse)
    return c
end

function _synthesize_direct!(x::AbstractMatrix, t::NSCT, c::AbstractArray, threaded)
    Tf = filter_eltype(eltype(c))
    p = _convert_transform(Tf, t)
    fp, qfp = p.lp_filters, p.dfb_filters
    offsets = _nsct_offsets(p)
    coarse = copy(view(c, :, :, 1))
    for j in p.J:-1:1
        bands = [copy(view(c, :, :, offsets[j] + k)) for k in 1:(1 << p.L_array[j])]
        bp = nsdfb_reconstruct(bands, qfp, j)
        coarse = nsp_reconstruct(coarse, bp, fp, j)
    end
    return copyto!(x, coarse)
end
