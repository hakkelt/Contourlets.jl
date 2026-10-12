# Translation-invariant (TICT) and semi-translation-invariant (STICT) contourlet
# transforms of Eslami & Radha (2006, §III).
#
# Both start from the translation-invariant Laplacian pyramid (`tilp.jl`): at
# every level i, four full-size highpass channels K_c and, after the last level, a
# full-size lowpass.  A directional filter bank then splits every highpass
# channel (§III-C):
#
#   • TICT — the translation-invariant DFB, i.e. the DFB with its samplers
#     suppressed and the 2^(-l) scaling (§III-B).  That is this package's NSDFB.
#   • STICT — the critically sampled DFB.  The pyramid stays translation
#     invariant, the directional stage does not.  Redundancy 4L + 1.
#
# Following the channels' passbands (their Fig. 6), channels K₀ and K₃ get the full
# DFB and K₁ and K₂ the "vertical" and "horizontal" DFBs, half trees that stop
# splitting one of the two level-1 fans (Fig. 7(b)–(c)).  Which fan to split is
# fixed by the channel's energy: K₁ (coset shift (1,0) along the rows) carries most
# of its energy in the fan |ω_row| > |ω_col| and K₂ in the other, so each splits
# the fan its energy is in.  A level of depth lᵢ ≥ 1 then has 3·2^(lᵢ) + 2 TICT
# subbands (4 when lᵢ = 0), plus the lowpass.
#
# The DFB filters are the same at every pyramid level, as in the paper.  (The
# NSCT's coarse-scale upsampling, `nsdfb.jl`, does not help here: the coarse TILP
# channels carry the decimated pyramid's aliasing at high frequencies, and
# measured the same way the unmodified filters leak least at every level.)
#
# Everything is a pointwise product on the DFT grid, so both transforms are exact
# on the torus and the TICT commutes with every circular shift of the input.
#
# Reference: R. Eslami & H. Radha, "Translation-invariant contourlet transform and
# its application to image denoising", IEEE TIP 15(11), 2006, pp. 3362–3374.

# Half tree of each TILP channel (see above).
const _TILP_HALVES = (:both, :second, :first, :both)

function _check_tict_params(p::ContourletParams)
    any(>(0), p.L_array) && !is_ladder(p.dfb_filters) && throw(
        ArgumentError("the TICT and STICT need a ladder QuincunxFilterPair such as Q2345")
    )
    return p
end

# ── Workspaces ────────────────────────────────────────────────────────────────

"""
    TICTWorkspace

Precomputed TILP masks, directional filters, FFT buffers and plans of a TICT at
one image size and element type.  Build with [`make_tict_workspace`](@ref).
"""
struct TICTWorkspace{Td <: Number, Tf <: AbstractFloat, C <: AbstractMatrix{Complex{Tf}}, B, P, Q}
    params::ContourletParams{Tf}
    image_size::NTuple{2, Int}
    masks::_TILPMasks{C}
    banks::Vector{Vector{Union{Nothing, B}}}    # [level][channel]; nothing when L = 0
    items::Vector{NTuple{3, Int}}               # (level, channel, subband) of every output
    X::C                                        # input spectrum / synthesis accumulator
    Y::Vector{C}                                # per task: channel spectrum / channel sum
    Z::Vector{C}                                # per task: scratch (consumed by the inverse FFT)
    acc::Vector{C}                              # per task ≥ 2: partial synthesis sums
    pfwd::P
    pinv::Q
end

Base.eltype(::TICTWorkspace{Td}) where {Td} = Td

"""
    STICTWorkspace

Precomputed TILP masks, decimated directional filter banks with their buffers,
FFT buffers and plans of an STICT at one image size and element type.  Build with
[`make_stict_workspace`](@ref).
"""
struct STICTWorkspace{Td <: Number, Tf <: AbstractFloat, C <: AbstractMatrix{Complex{Tf}}, R, P, Q}
    params::ContourletParams{Tf}
    image_size::NTuple{2, Int}
    masks::_TILPMasks{C}
    runners::Vector{Vector{Union{Nothing, R}}}  # [level][channel]; nothing when L = 0
    X::C                                        # full spectrum / synthesis accumulator
    Z::C                                        # scratch
    pfwd::P                                     # in-place FFT plans on X and Z
    pinv::Q
end

Base.eltype(::STICTWorkspace{Td}) where {Td} = Td

_tict_types(params::ContourletParams{Tp}, T::Type) where {Tp} = (Td = promote_type(Tp, T); (Td, _filter_eltype(Td)))

"""
    make_tict_workspace(params, image_size; T=Float64, M=Matrix{T}) -> TICTWorkspace
    make_tict_workspace(image, params) -> TICTWorkspace

Workspace for [`tict_forward`](@ref) / [`tict_inverse`](@ref) at `image_size` for
data of element type `T` stored as `M` (pass a device matrix type, or use the
`image` form, to run on a GPU).  With it the `!` transforms allocate nothing.
"""
make_tict_workspace(image::AbstractMatrix, params::ContourletParams{Tp}) where {Tp} =
    make_tict_workspace(params, size(image); T = eltype(image), M = typeof(similar(image, promote_type(Tp, eltype(image)), 0, 0)))

function make_tict_workspace(
        params::ContourletParams{Tp}, image_size::Tuple{Int, Int};
        T::Type{<:Number} = Float64, M::Type{<:AbstractMatrix} = Matrix{promote_type(Tp, T)}
    ) where {Tp}
    Td, Tf = _tict_types(params, T)
    p = _check_tict_params(_convert_params(Tf, params))
    n1, n2 = image_size
    real_data = Td <: Real
    ref = _allocate_zeros(M, Td, image_size)
    rows = real_data ? (1:(n1 ÷ 2 + 1)) : (1:n1)
    masks = _adapt_masks(ref, _tilp_masks(Tf, p.lp_filters, image_size, p.J; rows = rows))
    built = Dict{Tuple{Int, Symbol}, Any}()
    bank(l, half) = get!(built, (l, half)) do
        _adapt_bank(
            ref, _directional_bank(
                Tf, image_size, l, p.dfb_filters.f_ladder;
                decimated = false, half = half, real_data = real_data
            )
        )
    end
    B = _bank_type(ref, Tf)
    banks = [Union{Nothing, B}[l == 0 ? nothing : bank(l, h) for h in _TILP_HALVES] for l in p.L_array]
    items = [(i, c, k) for (i, l) in enumerate(p.L_array) for c in 1:4 for k in 1:_tict_n_subbands(l, c)]
    X, pfwd, pinv = _ns_plans(ref)
    nt = _n_fft_tasks(X)
    Y = [similar(X) for _ in 1:nt]
    return TICTWorkspace{Td, Tf, typeof(X), B, typeof(pfwd), typeof(pinv)}(
        p, (n1, n2), masks, banks, items, X, Y, [similar(X) for _ in 1:nt], typeof(X)[similar(X) for _ in 2:nt], pfwd, pinv
    )
end

# Tasks the FFT loops of a workspace can split into: one per thread on the host,
# one on a device (whose FFTs are already parallel).
_n_fft_tasks(::Array) = Threads.nthreads()
_n_fft_tasks(::AbstractArray) = 1

# Items `t` of `nt` contiguous, balanced chunks of `1:n`.
_chunk(n::Int, nt::Int, t::Int) = (div((t - 1) * n, nt) + 1):div(t * n, nt)

"""
    make_stict_workspace(params, image_size; T=Float64, M=Matrix{T}) -> STICTWorkspace
    make_stict_workspace(image, params) -> STICTWorkspace

Workspace for [`stict_forward`](@ref) / [`stict_inverse`](@ref); see
[`make_tict_workspace`](@ref).  Both image dimensions must be divisible by
`2^max(L - 1, 1)` for the deepest directional level `L`.
"""
make_stict_workspace(image::AbstractMatrix, params::ContourletParams{Tp}) where {Tp} =
    make_stict_workspace(params, size(image); T = eltype(image), M = typeof(similar(image, promote_type(Tp, eltype(image)), 0, 0)))

function make_stict_workspace(
        params::ContourletParams{Tp}, image_size::Tuple{Int, Int};
        T::Type{<:Number} = Float64, M::Type{<:AbstractMatrix} = Matrix{promote_type(Tp, T)}
    ) where {Tp}
    Td, Tf = _tict_types(params, T)
    p = _check_tict_params(_convert_params(Tf, params))
    ref = _allocate_zeros(M, Td, image_size)
    masks = _adapt_masks(ref, _tilp_masks(Tf, p.lp_filters, image_size, p.J))
    built = Dict{Tuple{Int, Symbol}, Any}()
    runner(l, half) = get!(built, (l, half)) do
        b = _directional_bank(Tf, image_size, l, p.dfb_filters.f_ladder; decimated = true, half = half)
        _bank_runner(_adapt_bank(ref, b), ref)
    end
    R = _runner_type(ref, Tf)
    runners = [Union{Nothing, R}[l == 0 ? nothing : runner(l, h) for h in _TILP_HALVES] for l in p.L_array]
    X = fill!(similar(ref, Complex{Tf}, image_size), zero(Complex{Tf}))
    pfwd, pinv = _inplace_plans(X, FFTW.ESTIMATE)
    return STICTWorkspace{Td, Tf, typeof(X), R, typeof(pfwd), typeof(pinv)}(
        p, image_size, masks, runners, X, similar(X), pfwd, pinv
    )
end

# ── Coefficient containers ────────────────────────────────────────────────────

_tict_n_subbands(l::Int, c::Int) = _bank_n_subbands(l, _TILP_HALVES[c])

"""
    similar_tict_coefficients(params, image_size; Td=Float64, M=Matrix{Td}) -> TICTCoefficients

Zero-filled TICT coefficients for `image_size`: every subband has the image's size.
"""
function similar_tict_coefficients(
        params::ContourletParams{Tf}, image_size::Tuple{Int, Int};
        Td::Type = Tf, M::Type{<:AbstractMatrix} = Matrix{Td}
    ) where {Tf}
    z() = _allocate_zeros(M, Td, image_size)
    subbands = [[[z() for _ in 1:_tict_n_subbands(l, c)] for c in 1:4] for l in params.L_array]
    return TICTCoefficients(z(), subbands)
end

"""
    similar_stict_coefficients(params, image_size; Td=Float64, M=Matrix{Td}) -> TICTCoefficients

Zero-filled STICT coefficients for `image_size`: each channel's subbands have the
sizes of a critically sampled DFB of that channel.
"""
function similar_stict_coefficients(
        params::ContourletParams{Tf}, image_size::Tuple{Int, Int};
        Td::Type = Tf, M::Type{<:AbstractMatrix} = Matrix{Td}
    ) where {Tf}
    z(dims) = _allocate_zeros(M, Td, dims)
    n1, n2 = image_size
    subbands = [
        [[z(s) for s in dfb_subband_sizes(n1, n2, l; half = h)] for h in _TILP_HALVES]
            for l in params.L_array
    ]
    return TICTCoefficients(z(image_size), subbands)
end

# ── Transforms ────────────────────────────────────────────────────────────────

function _check_tict_input(ws, x::AbstractMatrix)
    size(x) == ws.image_size || throw(
        DimensionMismatch("image size $(size(x)) does not match the workspace's $(ws.image_size)")
    )
    return x
end

# The `!` forms write through the workspace's FFT plans, so the caller's arrays
# must have the workspace's element type.
function _check_tict_eltype(ws, A::AbstractMatrix, what::String)
    eltype(A) === eltype(ws) || throw(
        ArgumentError(
            string(
                "the $(what) has element type $(eltype(A)) but the workspace was built ",
                "for $(eltype(ws)); build both with the same element type"
            )
        )
    )
    return A
end

# The subbands `ws.items[range]` from the input spectrum `ws.X`, with the task's
# buffers Y (channel spectrum, refreshed when the channel changes) and Z.
function _tict_forward_items!(coeffs::TICTCoefficients, ws::TICTWorkspace, range, Y, Z)
    m = ws.masks
    channel = (0, 0)
    for n in range
        i, c, k = ws.items[n]
        if (i, c) != channel
            Y .= m.ana[i][c] .* ws.X
            channel = (i, c)
        end
        bank = ws.banks[i][c]
        if bank === nothing
            mul!(coeffs.subbands[i][c][k], ws.pinv, Y)    # consumes Y
            channel = (0, 0)
        else
            Z .= bank.ana[k] .* Y
            mul!(coeffs.subbands[i][c][k], ws.pinv, Z)
        end
    end
    return coeffs
end

# acc += the synthesis of the subbands `ws.items[range]`, with the task's FFT buffer
# F and channel buffer Y: a channel's directional syntheses are summed in Y and
# its TILP mask applied once.
function _tict_inverse_items!(acc, coeffs::TICTCoefficients, ws::TICTWorkspace, range, F, Y)
    m = ws.masks
    channel = (0, 0)
    for n in range
        i, c, k = ws.items[n]
        if (i, c) != channel
            channel == (0, 0) || (acc .+= m.syn[channel[1]][channel[2]] .* Y)
            fill!(Y, zero(eltype(Y)))
            channel = (i, c)
        end
        bank = ws.banks[i][c]
        mul!(F, ws.pfwd, coeffs.subbands[i][c][k])
        if bank === nothing
            Y .+= F
        else
            Y .+= bank.syn[k] .* F
        end
    end
    channel == (0, 0) || (acc .+= m.syn[channel[1]][channel[2]] .* Y)
    return acc
end

# The subbands are independent, so with `threaded` (host workspaces only) their
# FFTs run in `length(ws.Z)` tasks over contiguous chunks of `ws.items`; at these
# sizes that scales far better than FFTW's own threads, which split one FFT.
function _tict_forward_ws!(coeffs::TICTCoefficients, x::AbstractMatrix, ws::TICTWorkspace, threaded::Bool = false)
    mul!(ws.X, ws.pfwd, x)
    n, nt = length(ws.items), length(ws.Z)
    if threaded && nt > 1
        @batch for t in 1:nt
            _tict_forward_items!(coeffs, ws, _chunk(n, nt, t), ws.Y[t], ws.Z[t])
        end
    else
        _tict_forward_items!(coeffs, ws, 1:n, ws.Y[1], ws.Z[1])
    end
    ws.Z[1] .= ws.masks.coarse_ana .* ws.X
    mul!(coeffs.coarse, ws.pinv, ws.Z[1])
    return coeffs
end

function _tict_inverse_ws!(x::AbstractMatrix, coeffs::TICTCoefficients, ws::TICTWorkspace, threaded::Bool = false)
    fill!(ws.X, zero(eltype(ws.X)))
    n, nt = length(ws.items), length(ws.Z)
    if threaded && nt > 1
        @batch for t in 1:nt
            acc = t == 1 ? ws.X : ws.acc[t - 1]
            t == 1 || fill!(acc, zero(eltype(acc)))
            _tict_inverse_items!(acc, coeffs, ws, _chunk(n, nt, t), ws.Z[t], ws.Y[t])
        end
        for a in ws.acc
            ws.X .+= a
        end
    else
        _tict_inverse_items!(ws.X, coeffs, ws, 1:n, ws.Z[1], ws.Y[1])
    end
    F = ws.Z[1]
    mul!(F, ws.pfwd, coeffs.coarse)
    ws.X .+= ws.masks.coarse_syn .* F
    mul!(x, ws.pinv, ws.X)
    return x
end

function _stict_forward_ws!(coeffs::TICTCoefficients, x::AbstractMatrix, ws::STICTWorkspace)
    m = ws.masks
    ws.X .= x
    ws.pfwd * ws.X
    for i in eachindex(m.ana), c in 1:4
        out = coeffs.subbands[i][c]
        r = ws.runners[i][c]
        if r === nothing
            ws.Z .= m.ana[i][c] .* ws.X
            ws.pinv * ws.Z
            _store!(out[1], ws.Z)
        else
            r.X .= m.ana[i][c] .* ws.X
            _bank_analyze_spectrum!(out, r)
        end
    end
    ws.Z .= m.coarse_ana .* ws.X
    ws.pinv * ws.Z
    _store!(coeffs.coarse, ws.Z)
    return coeffs
end

function _stict_inverse_ws!(x::AbstractMatrix, coeffs::TICTCoefficients, ws::STICTWorkspace)
    m = ws.masks
    fill!(ws.X, zero(eltype(ws.X)))
    for i in eachindex(m.syn), c in 1:4
        sbs = coeffs.subbands[i][c]
        r = ws.runners[i][c]
        if r === nothing
            ws.Z .= sbs[1]
            ws.pfwd * ws.Z
            ws.X .+= m.syn[i][c] .* ws.Z
        else
            _bank_synthesize_spectrum!(sbs, r)
            ws.X .+= m.syn[i][c] .* r.X
        end
    end
    ws.Z .= coeffs.coarse
    ws.pfwd * ws.Z
    ws.X .+= m.coarse_syn .* ws.Z
    ws.pinv * ws.X
    return _store!(x, ws.X)
end

# Workspace at the data's own precision: like `nsct_forward`, the allocating forms
# never promote the data to the precision of `params`.
function _own_workspace(make, x::AbstractMatrix, params::ContourletParams)
    Td = _data_eltype(x)
    p = _convert_params(_filter_eltype(Td), params)
    return make(p, size(x); T = Td, M = typeof(similar(x, Td, 0, 0)))
end

# Input in the workspace's data type (a copy only when the types differ).
_ws_input(ws, x::AbstractMatrix) = (Td = eltype(ws); Td === eltype(x) ? x : Td.(x))

"""
    tict_forward(image, params::ContourletParams; workspace=nothing, threading=Auto())
        -> TICTCoefficients

Translation-invariant contourlet transform of Eslami & Radha (2006).  A
`params.J`-level translation-invariant Laplacian pyramid gives, at every level
`i`, four full-size highpass channels; each is split by a translation-invariant
directional filter bank of depth `params.L_array[i]` — the full tree for channels
1 and 4, a half tree for channels 2 and 3 (see [`TICTCoefficients`](@ref)).  Every
coefficient array has the input's size, and the transform commutes with circular
shifts of the input.

The redundancy is `3 Σᵢ 2^L_array[i] + 2J + 1` when every `L_array[i] ≥ 1` (a
level with `L_array[i] = 0` contributes its 4 channels).  [`stict_forward`](@ref) trades
the directional stage's translation invariance for a redundancy of `4J + 1`.

`params.lp_filters` must be biorthogonal (as [`CDF97`](@ref) is) and
`params.dfb_filters` a ladder pair such as [`Q2345`](@ref).  Pass a workspace from
[`make_tict_workspace`](@ref) to reuse the precomputed filters.  With threading
on (`Enabled()`, or `Auto()` for complex data) the independent subband FFTs run
in one task per Julia thread; device arrays ignore it.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 32, 32);

julia> p = ContourletParams(J = 2, L_array = [2, 1]);

julia> c = tict_forward(x, p);

julia> length.(c.subbands[1]), length.(c.subbands[2])
([4, 3, 3, 4], [2, 2, 2, 2])

julia> maximum(abs, tict_inverse(c, p) .- x) < 1e-12
true
```
"""
function tict_forward(
        image::AbstractMatrix, params::ContourletParams;
        workspace::Union{Nothing, TICTWorkspace} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    ws = workspace === nothing ? _own_workspace(make_tict_workspace, image, params) : workspace
    x = _check_tict_input(ws, _ws_input(ws, image))
    coeffs = similar_tict_coefficients(ws.params, ws.image_size; Td = eltype(ws), M = typeof(similar(x)))
    return _tict_forward_ws!(coeffs, x, ws, _use_threading(threading, eltype(ws)))
end

"""
    tict_forward!(coeffs, image, params; workspace=nothing, threading=Auto()) -> coeffs

In-place [`tict_forward`](@ref) into coefficients from
[`similar_tict_coefficients`](@ref).  With a workspace it allocates nothing
(a few hundred bytes for the tasks when threading is on).
"""
function tict_forward!(
        coeffs::TICTCoefficients, image::AbstractMatrix, params::ContourletParams;
        workspace::Union{Nothing, TICTWorkspace} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    ws = workspace === nothing ? _own_workspace(make_tict_workspace, image, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    return _tict_forward_ws!(coeffs, _check_tict_input(ws, _ws_input(ws, image)), ws, _use_threading(threading, eltype(ws)))
end

"""
    tict_inverse(coeffs::TICTCoefficients, params; workspace=nothing, threading=Auto()) -> image

Inverse of [`tict_forward`](@ref): exact reconstruction (to rounding error).
"""
function tict_inverse(
        coeffs::TICTCoefficients, params::ContourletParams;
        workspace::Union{Nothing, TICTWorkspace} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    ws = workspace === nothing ? _own_workspace(make_tict_workspace, coeffs.coarse, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    return _tict_inverse_ws!(similar(coeffs.coarse, eltype(ws)), coeffs, ws, _use_threading(threading, eltype(ws)))
end

"""
    tict_inverse!(image, coeffs, params; workspace=nothing, threading=Auto()) -> image

In-place [`tict_inverse`](@ref).  With a workspace it allocates nothing (a few
hundred bytes for the tasks when threading is on).
"""
function tict_inverse!(
        image::AbstractMatrix, coeffs::TICTCoefficients, params::ContourletParams;
        workspace::Union{Nothing, TICTWorkspace} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    ws = workspace === nothing ? _own_workspace(make_tict_workspace, image, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    _check_tict_eltype(ws, image, "output image")
    return _tict_inverse_ws!(_check_tict_input(ws, image), coeffs, ws, _use_threading(threading, eltype(ws)))
end

"""
    stict_forward(image, params::ContourletParams; workspace=nothing, threading=Auto())
        -> TICTCoefficients

Semi-translation-invariant contourlet transform of Eslami & Radha (2006): the
translation-invariant Laplacian pyramid of [`tict_forward`](@ref) followed by
critically sampled directional filter banks (full tree on channels 1 and 4, half
trees on 2 and 3).  The pyramid commutes with circular shifts; the directional
subbands are those of the decimated DFB.  Redundancy `4J + 1`.

Both image dimensions must be divisible by `2^max(L - 1, 1)` for the deepest
directional level `L`.  Pass a workspace from [`make_stict_workspace`](@ref) to
reuse the precomputed filters and buffers.

# Examples
```jldoctest
julia> using Contourlets, Random

julia> x = randn(Xoshiro(1), 32, 32);

julia> p = ContourletParams(J = 2, L_array = [3, 2]);

julia> c = stict_forward(x, p);

julia> sum(length, Iterators.flatten(Iterators.flatten(c.subbands))) + length(c.coarse) == 9 * length(x)
true

julia> maximum(abs, stict_inverse(c, p) .- x) < 1e-12
true
```
"""
function stict_forward(
        image::AbstractMatrix, params::ContourletParams;
        workspace::Union{Nothing, STICTWorkspace} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    ws = workspace === nothing ? _own_workspace(make_stict_workspace, image, params) : workspace
    x = _check_tict_input(ws, _ws_input(ws, image))
    coeffs = similar_stict_coefficients(ws.params, ws.image_size; Td = eltype(ws), M = typeof(similar(x)))
    return _stict_forward_ws!(coeffs, x, ws)
end

"""
    stict_forward!(coeffs, image, params; workspace=nothing, threading=Auto()) -> coeffs

In-place [`stict_forward`](@ref) into coefficients from
[`similar_stict_coefficients`](@ref).  With a workspace it allocates nothing.
"""
function stict_forward!(
        coeffs::TICTCoefficients, image::AbstractMatrix, params::ContourletParams;
        workspace::Union{Nothing, STICTWorkspace} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    ws = workspace === nothing ? _own_workspace(make_stict_workspace, image, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    return _stict_forward_ws!(coeffs, _check_tict_input(ws, _ws_input(ws, image)), ws)
end

"""
    stict_inverse(coeffs::TICTCoefficients, params; workspace=nothing, threading=Auto()) -> image

Inverse of [`stict_forward`](@ref): exact reconstruction (to rounding error).
"""
function stict_inverse(
        coeffs::TICTCoefficients, params::ContourletParams;
        workspace::Union{Nothing, STICTWorkspace} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    ws = workspace === nothing ? _own_workspace(make_stict_workspace, coeffs.coarse, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    return _stict_inverse_ws!(similar(coeffs.coarse, eltype(ws)), coeffs, ws)
end

"""
    stict_inverse!(image, coeffs, params; workspace=nothing, threading=Auto()) -> image

In-place [`stict_inverse`](@ref).  With a workspace it allocates nothing.
"""
function stict_inverse!(
        image::AbstractMatrix, coeffs::TICTCoefficients, params::ContourletParams;
        workspace::Union{Nothing, STICTWorkspace} = nothing,
        threading::ThreadingPolicy = Auto()
    )
    ws = workspace === nothing ? _own_workspace(make_stict_workspace, image, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    _check_tict_eltype(ws, image, "output image")
    return _stict_inverse_ws!(_check_tict_input(ws, image), coeffs, ws)
end
