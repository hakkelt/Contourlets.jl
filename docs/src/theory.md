# [Theory](@id theory)

This page gives a brief mathematical description of the transforms implemented
in Contourlets.jl.  For in-depth derivations see the primary references listed
at the bottom.

## Laplacian Pyramid (LP)

The LP provides a multiscale decomposition without aliasing.

**Analysis** (one level, downsampling factor 2):
```
coarse   = ↓2 · (h ⊛ image)
pred     = g ⊛ (↑2 · coarse)
bandpass = image − pred
```

**Synthesis** (exact for any biorthogonal pair `(h, g)`):
```
rec = bandpass + g ⊛ (↑2 · coarse) = image
```

The default filter pair [`CDF97`](@ref) is the CDF 9/7 biorthogonal wavelet used
in JPEG 2000.

## Directional Filter Bank (DFB)

The DFB partitions the 2-D frequency plane into ``2^L`` directional wedges
using a binary tree of quincunx filter banks and shearing operations
(Bamberger & Smith 1992).

At each tree level ``d = 1, \ldots, L``:
1. **Shear** the input with alternating horizontal/vertical periodised index
   remapping.
2. Apply the **quincunx filter bank** (analysis pair ``h_q, g_q``) and
   quincunx downsampling.
3. **Inverse-shear** the high-pass branch for the next level.

The default quincunx pair [`Q2345`](@ref) is a PR 1×2 Haar-like pair with
perfect reconstruction verified to ``< 10^{-15}``.

## Pyramid Directional Filter Bank (PDFB) — Contourlet Transform

The CT couples `J` LP levels with one DFB per bandpass:

```
coarse ← image
for j = 1 … J:
    coarse_j, bandpass_j = LP(coarse, h, g)
    subbands[j]          = DFB(bandpass_j, L_array[j], h_q, g_q)
    coarse               = coarse_j
```

**Parabolic scaling** chooses the direction count so that the spatial support
of directional subbands satisfies the parabolic scaling law
``\text{width} \sim \text{length}^2``:

```math
L_j = L_{j_0} + \left\lfloor \frac{j_0 - j}{2} \right\rfloor, \quad j = 1, \ldots, J
```

Use [`parabolic_levels`](@ref) to generate ``L_\text{array}`` automatically.

## Nonsubsampled Contourlet Transform (NSCT)

The NSCT replaces every decimated filter bank with its non-subsampled (à trous)
counterpart:

- **NSP**: at LP level ``j``, the filters ``h, g`` are replaced by
  ``h_j = h`` upsampled by ``2^{j-1}`` (inserting ``2^{j-1}-1`` zeros between
  taps).  No downsampling is applied, so all outputs have the same size as the
  input.
- **NSDFB**: the quincunx filters at tree level ``d`` within pyramid level
  ``j`` are upsampled by ``2^{d + j - 2}`` (combined pyramid + directional
  upsampling factor).

The NSCT is **fully shift-invariant**: a circular shift of the input produces
the same shift in each subband.  The cost is redundancy proportional to
``1 + \sum_{j=1}^J 2^{L_j}``.

## References

1. M. N. Do and M. Vetterli, "The Contourlet Transform: An Efficient Directional
   Multiresolution Image Representation," *IEEE Trans. Image Process.*, 2005.
2. A. L. da Cunha, J. Zhou, and M. N. Do, "The Nonsubsampled Contourlet Transform:
   Theory, Design, and Applications," *IEEE Trans. Image Process.*, 2006.
3. R. H. Bamberger and M. J. T. Smith, "A Filter Bank for the Directional
   Decomposition of Images," *IEEE Trans. Signal Process.*, 1992.
4. I. Daubechies and W. Sweldens, "Factoring Wavelet Transforms into Lifting
   Steps," *J. Fourier Anal. Appl.*, 1998.
