# Translation-invariant contourlet transform (TICT) of Eslami & Radha (2006): the
# translation-invariant pyramid followed by nonsubsampled DFBs, on the DFT grid.
# TICTWorkspace, make_tict_workspace, the threaded subband loops and
# tict_forward / tict_inverse (and ! variants).  The overview and the parts shared
# with the STICT are in `tilp_transforms.jl`.

# ── Workspace ─────────────────────────────────────────────────────────────────

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
    p = _convert_params(Tf, params)
    n1, n2 = image_size
    real_data = Td <: Real
    ref = _allocate_zeros(M, Td, image_size)
    rows = real_data ? (1:(n1 ÷ 2 + 1)) : (1:n1)
    masks = _adapt_masks(ref, _tilp_masks(Tf, p.lp_filters, image_size, p.J; rows = rows))
    B = _bank_type(ref, Tf)
    # One full bank per depth; the half trees share its masks (`_half_bank`).
    built = Dict{Int, B}()
    full(l) = get!(built, l) do
        _adapt_bank(ref, _directional_bank(Tf, image_size, l, p.dfb_filters.f_ladder; decimated = false, real_data = real_data))
    end
    bank(l, half) = _half_bank(full(l), full(1), half)
    banks = [Union{Nothing, B}[l == 0 ? nothing : bank(l, h) for h in _TILP_HALVES] for l in p.L_array]
    items = [(i, c, k) for (i, l) in enumerate(p.L_array) for c in 1:4 for k in 1:_tict_n_subbands(l, c)]
    X, pfwd, pinv = _ns_plans(ref; measured = true)
    nt = _n_tasks(X, length(items))
    return TICTWorkspace{Td, Tf, typeof(X), B, typeof(pfwd), typeof(pinv)}(
        p, (n1, n2), masks, banks, items, X, [similar(X) for _ in 1:nt], [similar(X) for _ in 1:nt],
        typeof(X)[similar(X) for _ in 2:nt], pfwd, pinv
    )
end

# ── Transforms ────────────────────────────────────────────────────────────────

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
# FFTs run in `length(ws.Z)` tasks over contiguous chunks of `ws.items`
# (`fft_tasks.jl`).
function _tict_forward_ws!(coeffs::TICTCoefficients, x::AbstractMatrix, ws::TICTWorkspace, threaded::Bool = false)
    mul!(ws.X, ws.pfwd, x)
    _batch_chunks(length(ws.items), length(ws.Z), threaded) do range, t
        _tict_forward_items!(coeffs, ws, range, ws.Y[t], ws.Z[t])
    end
    ws.Z[1] .= ws.masks.coarse_ana .* ws.X
    mul!(coeffs.coarse, ws.pinv, ws.Z[1])
    return coeffs
end

function _tict_inverse_ws!(x::AbstractMatrix, coeffs::TICTCoefficients, ws::TICTWorkspace, threaded::Bool = false)
    fill!(ws.X, zero(eltype(ws.X)))
    _chunked_sum!(ws.X, ws.acc, length(ws.items), threaded; foreach = _batch_chunks) do acc, range, t
        _tict_inverse_items!(acc, coeffs, ws, range, ws.Z[t], ws.Y[t])
    end
    F = ws.Z[1]
    mul!(F, ws.pfwd, coeffs.coarse)
    ws.X .+= ws.masks.coarse_syn .* F
    mul!(x, ws.pinv, ws.X)
    return x
end

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
