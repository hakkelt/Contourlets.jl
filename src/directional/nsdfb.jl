# Nonsubsampled Directional Filter Bank (NSDFB).
#
# da Cunha, Zhou & Do (2006, §II-A) obtain the NSDFB "by switching off the
# downsamplers/upsamplers in each two-channel filter bank in the DFB tree structure
# and upsampling the filters accordingly".  By the noble identities each channel is
# then a single filtering with the DFB's equivalent filter, so the NSDFB is the bank
# of `bank_design.jl` without the decimation: subband k is hₖ ⊛ x, at the
# input's size.  Each subband occupies an angular wedge of the frequency plane.
#
# Without samplers the per-channel delays of the DFB serve no purpose, so the
# filters here are the zero-phase amplitudes: every subband stays registered with
# the input.  The amplitudes are scaled by 2^(-l/2) so that Σₖ GₖHₖ = 1 (each
# two-channel stage satisfies Bezout's identity G₀H₀ + G₁H₁ = 1, NSCT paper §II-B).
#
# All filtering is circular (on the DFT grid), so the NSDFB — and hence the NSCT —
# commutes with circular shifts of the input, exactly.
#
# Coarse scales (§II-A.3, Fig. 4).  The fan filters' transition bands meet at the
# origin, so the low-frequency part of every directional filter is its "bad" part,
# and the bandpass of a coarse pyramid level falls right into it.  da Cunha et al.
# therefore filter pyramid level j with Uₖ(z^(2^m I)) instead of Uₖ(z), choosing m
# so that the filters' good region overlaps the pyramid passband; their example
# upsamples the last two stages of a five-scale NSCT by 2I.  The paper gives no
# general formula.  Here `tree_level = j` (1 = finest) uses
#
#     m = max(0, j − 2),
#
# measured, not guessed: for the separable CDF 9/7 pyramid of this package and
# every depth l = 1 … 5, it minimises the share of each level's bandpass energy
# that falls outside the subband's ideal angular wedge (0.04–0.28 at every level
# and depth, against 0.40–0.83 at j = 5 without upsampling; 256² grid).  m = j − 1 would map the band
# onto the filters' full good region but folds the pyramid's leakage back across
# the replicas of the upsampled filter.  Hₖ(2^m ω) is still a function on the
# DFT grid, and Σₖ GₖHₖ = 1 holds unchanged.

# Upsampling exponent m of the directional filters at pyramid level `tree_level`.
_nsdfb_upsampling(tree_level::Int) = max(0, tree_level - 2)

# A spectrum buffer and FFT pair for the data type (the rfft half-plane for real
# data).  The spectra and outputs are allocated with `similar` and filled through
# plans, so they keep the input's array type even for an array type whose FFT runs
# on host memory.
#
# A workspace passes `measured = true`: host plans are then measured and
# single-threaded, because the workspace threads over its independent subband FFTs
# instead (MultiscaleFilterBanks' mask-bank engine), which at these sizes scales far better than FFTW's
# threads splitting one FFT.
function _ns_plans(x::AbstractMatrix; measured::Bool = false)
    X = _ns_spectrum(x)
    (measured && x isa Array) || return (X, _ns_plan_pair(x, X)...)
    return with_fftw_threads(1) do
        (X, _ns_plan_pair(x, X; flags = FFTW.MEASURE)...)
    end
end

_ns_spectrum(x::AbstractMatrix{T}) where {T <: Real} = similar(x, Complex{T}, (size(x, 1) ÷ 2 + 1, size(x, 2)))
_ns_spectrum(x::AbstractMatrix) = similar(x)

_ns_plan_pair(x::AbstractMatrix{<:Real}, X; kwargs...) = (plan_rfft(x; kwargs...), plan_irfft(X, size(x, 1); kwargs...))
_ns_plan_pair(x::AbstractMatrix, X; kwargs...) = (plan_fft(x; kwargs...), plan_ifft(X; kwargs...))

function _nsdfb_bank(
        img::AbstractMatrix, l::Int, qfp::QuincunxFilterPair, tree_level::Int, half::Symbol, ::Type{Tf}
    ) where {Tf}
    bank = _directional_bank(
        Tf, size(img), l, qfp.f_ladder;
        decimated = false, half = half, real_data = eltype(img) <: Real,
        upsample = _nsdfb_upsampling(tree_level)
    )
    return _adapt_bank(img, bank)
end

"""
    nsdfb_decompose(bandpass, l_levels, qfp, tree_level=1; half=:both)
        -> Vector{Matrix}

Nonsubsampled DFB analysis.  Returns `2^l_levels` directional subbands, each the
same size as `bandpass`, each occupying an angular wedge of the frequency plane.
Every subband commutes with circular shifts of `bandpass`.

`tree_level` is the pyramid level `j` the bandpass comes from (1 = finest), as in
[`nsct_forward`](@ref).  From `j = 3` on, each directional filter `Uₖ(z)` is
replaced by `Uₖ(z^(2^m I))` with `m = j − 2`, so that the filters' well-resolved
region covers the coarser band (da Cunha, Zhou & Do 2006, §II-A.3).  Pass the
same `tree_level` to [`nsdfb_reconstruct`](@ref).

`half` selects the half trees of Eslami & Radha (2006): `:first` keeps splitting
only the first level-1 fan and returns its `2^(l_levels-1)` wedges followed by the
unsplit second fan; `:second` returns the unsplit first fan followed by the
`2^(l_levels-1)` wedges of the second fan.  Either gives `2^(l_levels-1) + 1`
subbands, and the split fan's wedges equal the corresponding subbands of the full
tree.

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
        half::Symbol = :both
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    tree_level >= 1 || throw(ArgumentError("tree_level must be ≥ 1"))
    _check_half(half)
    l_levels == 0 && return [copy(bandpass)]
    Td = data_eltype(bandpass)
    Tf = filter_eltype(Td)
    img = Td === eltype(bandpass) ? bandpass : Td.(bandpass)
    bank = _nsdfb_bank(img, l_levels, qfp, tree_level, half, Tf)
    X, pfwd, pinv = _ns_plans(img)
    mul!(X, pfwd, img)
    Y = similar(X)
    return map(bank.ana) do a
        Y .= a .* X
        mul!(similar(img), pinv, Y)    # irfft plans may overwrite Y
    end
end

"""
    nsdfb_reconstruct(subbands, qfp, tree_level=1; half=:both) -> bandpass

Nonsubsampled DFB synthesis (inverse of [`nsdfb_decompose`](@ref)).  Pass the
same `tree_level` and `half` as the analysis; the tree depth is inferred from
`length(subbands)`.

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
        half::Symbol = :both
    )
    tree_level >= 1 || throw(ArgumentError("tree_level must be ≥ 1"))
    _check_half(half)
    n = length(subbands)
    n >= 1 || throw(ArgumentError("subbands must be non-empty"))
    n == 1 && return copy(subbands[1])
    l_levels = _dfb_levels_from_count(n, half)
    Td = data_eltype(subbands[1])
    Tf = filter_eltype(Td)
    bank = _nsdfb_bank(subbands[1], l_levels, qfp, tree_level, half, Tf)
    ref = Td === eltype(subbands[1]) ? subbands[1] : Td.(subbands[1])
    X, pfwd, pinv = _ns_plans(ref)
    acc = fill!(similar(X), zero(eltype(X)))
    for k in 1:n
        s = Td === eltype(subbands[k]) ? subbands[k] : Td.(subbands[k])
        mul!(X, pfwd, s)
        acc .+= bank.syn[k] .* X
    end
    return mul!(similar(ref), pinv, acc)
end
