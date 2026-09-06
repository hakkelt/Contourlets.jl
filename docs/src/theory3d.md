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
of `order` and `poly`.  The cost is measurable: the analysis masks are the
weights ``W_k`` themselves, so ``\sum_k W_k = 1`` holds exactly but
``\sum_k W_k^2`` ranges over ``[1/3, 1]`` — a frame-bound ratio of 3.  SurfBox
normalises pointwise instead, taking ``V_k = \sqrt{N W_k}`` for analysis, which
makes ``\sum_k V_k^2 = N`` identically: an exactly tight frame with redundancy
``N``, verified to machine precision in
`test/items/test_surfbox_reference.jl`.  For analysis, thresholding and
reconstruction the difference matters little; for applications that need
tight-frame noise behaviour — denoising with a single global threshold,
iterative reconstruction where the adjoint's conditioning matters — it is a real
limitation, and the fix is the pointwise normalisation above rather than the
optimised lifting coefficients the papers describe.

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

## Running on a GPU

The 3-D stack needs no device kernels.  Both transforms are FFT plus a pointwise
mask, and every mask is stored as broadcastable factors — the mapping responses
and wedge planes are 3-D arrays with singleton axes where they do not depend on
a frequency — so applying a mask, splitting a complex volume into real and
imaginary halves, and the pyramid's frequency-domain multiply are all plain
broadcasts.  Moving a plan to the device is therefore just moving its arrays and
replanning the transforms:

```julia
using CUDA, Contourlets
using CUDA: CuArray
import Adapt

params = ContourletParams3(J = 2, L_array = [2, 2])
ws = Adapt.adapt(CuArray, make_nsct3_workspace(params, (128, 128, 128), Float32))
c  = nsct3_forward(CuArray(vol), params; workspace = ws)
```

A workspace adapt moves the shared hourglass response once, not once per scale.
Verified against the CPU path on CUDA and OpenCL.  The decimated surfacelet
transform is CPU-only for now: its support-box extraction is an index gather,
not a broadcast.

## Surfacelet transform (decimated)

The NSCT3 buys exact shift invariance with redundancy.  The surfacelet transform
trades the invariance back for a smaller coefficient set, which is what makes it
usable as a sparsifying transform rather than only as an analysis tool.

The target frequency partitioning is Lu and Do's: pairs of concentric cubes
radiating from the origin, at different scales and directions.  Scale ``j``
covers the shell ``\|\omega\|_\infty \in [\pi/2^j, \pi/2^{j-1}]``.

Decimation is done by frequency-domain support extraction.  Every mask here has
*compact* support in frequency, so the masked spectrum is restricted to the
smallest symmetric box containing it and inverse-transformed at that reduced
size — the mechanism the wrapping-based discrete curvelet transform uses.  Two
consequences:

* The directional masks are **not** the Bernstein mapping kernels used by the
  nonsubsampled bank.  Those are the right choice there and the wrong one here:
  a mapping kernel is a polynomial in ``\cos\omega``, so ``f_0(M(\omega))`` is
  nonzero almost everywhere and its bounding box is the whole grid.  The
  requirement inverts, so the construction does: masks are built from soft steps
  that are *exactly* 0 and 1 outside a transition band.

* Perfect reconstruction comes from ``\sum V_i^2 = 1`` (the masks are the square
  roots of a partition of unity) rather than ``\sum W_i = 1``.  Synthesis is then
  the analysis adjoint, and the frame is tight in the weighted sense
  ``\sum_i N_i \|s_i\|^2 = N \|x\|^2``, where ``N_i`` is the subband's element
  count.

### One subtlety that silently breaks it

On the DFT grid, ``\omega \to -\omega`` is *not* a sign flip of the folded
frequency: at the Nyquist bin of an even-length axis ``\pi \equiv -\pi`` maps to
itself.  A sign-sensitive quantity like the wedge ratio ``\omega_2/\omega_1``
therefore comes out with opposite signs at two bins that are each other's
negation.  That makes the masked spectrum non-Hermitian, the subband complex,
and — because subbands are stored real — destroys perfect reconstruction, with
no error message.  The masks are evaluated at a canonical representative of each
``\pm`` pair to remove the ambiguity.

### How much it actually saves

Measured redundancy against the nonsubsampled transform at the same parameters:

| `J`, `L`     | surfacelet3 | NSCT3 |
|--------------|-------------|-------|
| 1, [1]       |  6.2        |   7   |
| 2, [1, 1]    |  7.7        |  13   |
| 2, [2, 2]    | 15.3        |  25   |
| 3, [2, 2, 3] | 16.0        |  49   |

So this is a *partially* decimated transform — about 3× leaner at useful
settings, not the redundancy of roughly 4 that Lu and Do report.  The reason is
worth knowing before trying to tune it away: an axis-aligned box recovers the
radial decimation in full (a coarser shell shrinks by 2 per axis) but almost
none of the directional decimation, because the bounding box of a pyramid
radiating from the origin is nearly as large as the pyramid's enclosing cube.
Closing that gap needs a box aligned with the wedge — sheared wrapping, or
equivalently Lu and Do's resampled checkerboard chain — and the shear has to be
an integer index remap on the sampling grid, which at dyadic wedge boundaries it
is not unless the decimation is interleaved with the refinement the way the
reference construction does it.  That is a missing piece, not a tuning problem.

That analysis is confirmed by the reference implementation.  Running SurfBox's
`Surfdec` on a 96³ volume at `Pyr_mode = 1.5` gives a redundancy of 3.9259 for
24 directional subbands and *the same* 3.9259 for 96 — flat in the direction
count, as sheared boxes predict.  This package's `surfacelet3` gives 15.4 and
46.4 at the matching settings, growing roughly linearly.  So the gap is not a
constant factor; it widens exactly where a directional transform earns its keep.

One further limit worth stating plainly: the reference multiscale pyramid
resamples by the non-integer factor 3/2 per axis, which this package's cube-shell
octave partition does not do.

### What has been checked against SurfBox

SurfBox is available again (MATLAB File Exchange submission 14485), so the
directional stages are now regression-tested against it in
`test/items/test_surfbox_reference.jl`, with `test/reference/` holding the golden
data and the MATLAB scripts that regenerate it.  What the comparison establishes:

  * The hourglass geometry agrees.  Over a 32³ grid the two constructions pick
    the same dominant hourglass at 94.6% of frequencies, with a mean absolute
    weight difference of 0.037; every disagreement is a near-tie in the
    reference (median gap 0.005 between the top two weights, against 0.887
    averaged over all frequencies), so the two differ only in how they round the
    transition bands.
  * The wedge partition agrees.  Across 80 plane waves at 12 directional
    subbands, the confusion matrix between SurfBox's subband labels and this
    package's is a bijection, 84% of directions match exactly under it, and for
    *every* one of the 80 the reference's subband is this package's first or
    second choice by energy.  This is the check that matters, because perfect
    reconstruction alone would pass for any structurally-PR bank, including one
    with the wrong tiling.

What it does not establish is coefficient-level agreement: SurfBox's hourglass
raises the magnitude response of a Kaiser-windowed diamond kernel to a power,
this package composes Bernstein mapping polynomials, and its NDFB is a decimated
resampled-checkerboard chain against directly-placed frequency boundaries here.
Same ideal partition, different discrete realisation.

## References

1. Y. M. Lu and M. N. Do, "Multidimensional Directional Filter Banks and
   Surfacelets", *IEEE Transactions on Image Processing*, 16(4):918–931, 2007.
2. Y. M. Lu and M. N. Do, "A Mapping-Based Design for Nonsubsampled Hourglass
   Filter Banks in Arbitrary Dimensions", *IEEE Transactions on Signal
   Processing*, 56(4):1466–1478, 2008.
3. G. G. Lorentz, *Bernstein Polynomials*, 2nd ed., Chelsea, 1986 — the
   multivariate Bernstein approximation behind the mapping kernels.
