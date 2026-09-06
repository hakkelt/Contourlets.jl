# [Theory — 3-D](@id theory3d)

```@meta
CurrentModule = Contourlets
```

This page covers the 3-D transforms.  For the 2-D constructions see
[Theory](@ref theory).

The 3-D transforms are not a dimensional generalisation of the 2-D ones.  The
quincunx lattice, the shear index remap and the Q2345 ladder all rely on
properties of the plane that do not survive, and the Laplacian pyramid's
critically sampled directional partner has no 3-D analogue at all: Lu and Do
prove that hourglass-shaped frequency partitioning *cannot* be achieved by a
critically sampled filter bank above two dimensions.  So the 3-D directional
stage is a different construction, with a redundancy of ``N`` in ``N``
dimensions that is unavoidable rather than a design choice.

## Nonsubsampled pyramid (NSP3)

The multiscale stage is the direct extension of the 2-D à trous pyramid.  At
level ``j`` the filters are upsampled by ``2^{j-1}`` along each axis and nothing
is decimated:

```
coarse   = h_j ⊛ volume
bandpass = volume − g_j ⊛ coarse
```

The synthesis rescale `_NSP_SYNTH_SCALE` is applied **per axis**, so the 2-D
constant carries over unchanged.  This is worth stating explicitly because the
obvious guess is wrong in an instructive way: the biorthogonal synthesis ``g``
has DC gain 2 per axis, giving ``2^3 = 8`` across a separable 3-D filtering, and
one might expect the correction to become ``1/8``.  It does not — the ``1/2``
applies once per axis, and ``(1/2)^3`` is exactly the ``1/8`` needed.  A wrong
normalisation here does not break perfect reconstruction, it merely fills the
"bandpass" with low-pass leakage, which is why the test suite checks that a
constant volume leaves the bandpass at zero.

All filtering is periodic, so the stage is exactly equivariant under circular
shifts and — since periodic separable filtering *is* circular convolution — an
FFT evaluation of it is exact rather than approximate.  See
[`conv3d_sep_fft!`](@ref).

## Hourglass filter bank

The directional stage starts by splitting the spectrum into three
hourglass-shaped subbands, each aligned with one frequency axis:

```math
P_i = \{\omega : |\omega_i| \ge |\omega_j| \ \text{for all } j \ne i\}
```

so subband ``i`` keeps the frequencies closer to axis ``i`` than to any other.

The design follows Lu and Do's mapping-based construction.  One-dimensional
polynomials ``f_0, f_1`` are composed with a multidimensional *mapping kernel*
``M(\omega)`` — a zero-phase FIR filter designed so that ``M \approx +1`` on the
desired passband and ``\approx -1`` outside it — giving multidimensional filters
``f_0(M(\omega))`` and ``f_1(M(\omega))`` that inherit both the FIR property and
the perfect-reconstruction relation between ``f_0`` and ``f_1``.

The kernels come from multivariate Bernstein polynomials under the change of
variables ``x_i = (1 - \cos\omega_i)/2``, which maps ``|\omega_i|`` monotonically
onto ``[0, 1]``.  That is the reason the construction is tractable: an hourglass
passband is a statement about the *ordering* of frequency magnitudes, and after
the change of variables it becomes the same ordering statement about the integer
Bernstein sample indices — see [`mapping_coefficients`](@ref).

Three subbands are produced by a cascade of two two-channel banks:

```
spectrum ──┬─ P₁                             (3-D mapping kernel)
           └─ P₂ ∪ P₃ ──┬─ P₂                (2-D kernel on ω₂, ω₃)
                        └─ P₃
```

The second stage needs only two frequency variables.  That is not an
optimisation but a theorem: Lu and Do's Proposition 1 shows the optimal passband
support for a tree node with index set ``J`` involves ``|J|`` frequency
variables, so in 3-D the second cut is a 2-D design.

### Two deviations from the reference construction

**Frequency domain rather than FIR.**  Lu and Do deliberately chose FIR spatial
filters because their earlier frequency-domain design needed too much memory for
streaming video.  That trade-off inverts here.  These stages are nonsubsampled
with a periodic boundary, so the FFT is already the default backend, and
evaluating the mapping kernel directly on the DFT grid is *exact* — there is no
truncation of the 37³ and 61³ FIR filters the paper reports.

**A partition of unity rather than a near-tight frame.**  The 1-D polynomials
used here satisfy ``f_0(x) + f_1(x) = 1`` identically.  Perfect reconstruction is
then structural — the masks sum to 1 at every frequency, the synthesis filters
are all 1, and reconstruction is a plain sum — and it is exact for every choice
of `order` and `poly`.  The cost is a frame-bound ratio near 2 instead of Lu and
Do's optimised lifting/ladder coefficients, which reach a near-tight frame.  For
analysis, thresholding and reconstruction this matters little; for applications
that need tight-frame noise behaviour it is a real limitation.

`order` (the Bernstein order ``N``) controls how sharply the subbands separate.
Larger ``N`` converges to the ideal indicator; the deviation from it falls
monotonically, from about 0.095 mean absolute error at ``N = 4`` to 0.024 at
``N = 20``.

## Directional filter bank (NSDFB3)

Each hourglass is then refined into wedges.  Inside ``P_i`` the two transverse
ratios

```math
s = \omega_a / \omega_i \in [-1, 1], \qquad t = \omega_b / \omega_i \in [-1, 1]
```

parameterise direction, and ``l`` levels of refinement cut each into ``2^l``
equal slabs.  With ``l_a + l_b = L`` this gives ``2^L`` wedges per hourglass and
``3 \cdot 2^L`` directional subbands in total, whose ideal passbands are
rectangular-based pyramids radiating from the origin.

Both ratios are invariant under ``\omega \to -\omega``, so the masks stay
symmetric and a real volume keeps real subbands.  Each ratio involves only two
frequency axes, so a wedge mask is a *matrix*: the whole bank stores
``2(2^{l_a} + 2^{l_b})`` planes no matter how large the volume is.

Lu and Do build these wedges from a chain of iteratively resampled checkerboard
filter banks, because their transform is critically sampled inside each
hourglass and the resampling is what keeps it so.  This bank is nonsubsampled,
so there is nothing to resample for and the wedge boundaries are placed directly
in the frequency domain.  The ideal partition is the same one.  Constructing it
directly also avoids the failure mode that a separable approximation invites:
filtering the transverse directions with 1-D filters produces stripe and block
tilings of the frequency plane, not wedges, and the resulting transform is still
perfectly reconstructing — so perfect reconstruction alone does not detect the
error.  The test suite therefore checks the wedge *index* a plane wave lands in,
not merely that some subband is large.

## 3-D NSCT

Composing the two stages gives the transform:

```
volume ─ NSP3 ─┬─ coarse ─→ next scale
               └─ bandpass ─ NSDFB3 ─→ 3·2^L directional subbands
```

Every coefficient array keeps the size of the input, so the transform is exactly
shift-invariant — and correspondingly redundant: ``1 + \sum_j 3 \cdot 2^{L_j}``
copies of the input.  For ``J = 3`` and ``L = [2, 2, 3]`` that is 49 volumes,
3.1 GB at 256³ `Float32`.  The streaming form of [`nsct3_forward`](@ref) exists
for this reason: it holds one subband at a time.

Because the directional bank depends only on the volume size and never on the
scale, a workspace plans the hourglass mapping response once and shares it
across every scale, with only the (matrix-sized) wedge masks differing.

## References

1. Y. M. Lu and M. N. Do, "Multidimensional Directional Filter Banks and
   Surfacelets", *IEEE Transactions on Image Processing*, 16(4):918–931, 2007.
2. Y. M. Lu and M. N. Do, "A Mapping-Based Design for Nonsubsampled Hourglass
   Filter Banks in Arbitrary Dimensions", *IEEE Transactions on Signal
   Processing*, 56(4):1466–1478, 2008.
3. G. G. Lorentz, *Bernstein Polynomials*, 2nd ed., Chelsea, 1986 — the
   multivariate Bernstein approximation behind the mapping kernels.
