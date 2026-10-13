# Translation-invariant contourlet transform (TICT) of Eslami & Radha (2006): the
# translation-invariant pyramid followed by nonsubsampled DFBs, on the DFT grid.
# TICTWorkspace, make_tict_workspace and tict_forward / tict_inverse (and !
# variants); the subband loops are the NSCT's (`mask_bank.jl`).  The overview and the parts shared
# with the STICT are in `tilp_transforms.jl`.

# ── Workspace ─────────────────────────────────────────────────────────────────

"""
    TICTWorkspace

Precomputed TILP masks, directional filters, FFT buffers and plans of a TICT at
one image size and element type.  Build with [`make_tict_workspace`](@ref).
"""
struct TICTWorkspace{Td <: Number, Tf <: AbstractFloat, M <: AbstractMatrix, E <: _MaskBankEngine}
    params::ContourletParams{Tf}
    image_size::NTuple{2, Int}
    # Staging buffers for an input or output that is not of the dense type `M`
    # the FFT plans run on (a view, a transpose; `_dense_input`).
    fwd_scratch::ScratchArena{M}
    inv_scratch::ScratchArena{M}
    # TILP channel masks (pyramid mask 4(i - 1) + c is channel c of level i) and
    # lowpass, the directional bank of every channel (bank index = pyramid index;
    # the identity bank where L = 0), the outputs (i, c, k), and the FFT plans and
    # per-task buffers (`mask_bank.jl`).
    engine::E
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
    # One full bank per depth (the identity bank at depth 0); the half trees share
    # its masks (`_half_bank`).
    built = Dict{Int, B}()
    full(l) = get!(built, l) do
        _adapt_bank(ref, _directional_bank(Tf, image_size, l, p.dfb_filters.f_ladder; decimated = false, real_data = real_data))
    end
    bank(l, half) = l == 0 ? full(0) : _half_bank(full(l), full(1), half)
    banks = B[bank(l, h) for l in p.L_array for h in _TILP_HALVES]
    g(i, c) = 4 * (i - 1) + c                   # pyramid mask and bank of channel c of level i
    items = [
        _MaskItem((i, c, k), g(i, c), g(i, c), k)
            for (i, l) in enumerate(p.L_array) for c in 1:4 for k in 1:_tict_n_subbands(l, c)
    ]
    X, pfwd, pinv = _ns_plans(ref; measured = true)
    engine = _MaskBankEngine(
        reduce(vcat, masks.ana), reduce(vcat, masks.syn), masks.coarse_ana, masks.coarse_syn,
        banks, items, X, pfwd, pinv
    )
    Mw = typeof(ref)
    return TICTWorkspace{Td, Tf, Mw, typeof(engine)}(p, (n1, n2), ScratchArena{Mw}(), ScratchArena{Mw}(), engine)
end

_ws_matrix_type(::TICTWorkspace{Td, Tf, M}) where {Td, Tf, M} = M

# ── Transforms ────────────────────────────────────────────────────────────────

# Every subband is ifft(Hₖ · K_{i,c} · X) with the TILP channel mask K_{i,c} and
# the channel's directional filter Hₖ: the mask-bank engine (`mask_bank.jl`), whose
# independent subband FFTs run in one task per FFT buffer with `threaded` (host
# workspaces only).
_tict_forward_ws!(coeffs::TICTCoefficients, x::AbstractMatrix, ws::TICTWorkspace, threaded::Bool = false) =
    _mask_bank_forward!(coeffs, x, ws.engine, threaded)

_tict_inverse_ws!(x::AbstractMatrix, coeffs::TICTCoefficients, ws::TICTWorkspace, threaded::Bool = false) =
    _dense_output!(x, ws, y -> _mask_bank_inverse!(y, coeffs, ws.engine, threaded))

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
    x = _dense_input(ws, _check_tict_input(ws, _ws_input(ws, image)))
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
    x = _dense_input(ws, _check_tict_input(ws, _ws_input(ws, image)))
    return _tict_forward_ws!(coeffs, x, ws, _use_threading(threading, eltype(ws)))
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
