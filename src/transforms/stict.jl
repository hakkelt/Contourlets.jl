# Semi-translation-invariant contourlet transform (STICT) of Eslami & Radha
# (2006): the translation-invariant pyramid followed by critically sampled DFBs.
# STICTWorkspace, make_stict_workspace and stict_forward / stict_inverse (and !
# variants).  The overview and the parts shared with the TICT are in
# `tilp_transforms.jl`.

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
    pbwd::Q                                     # (inverse unnormalised)
end

Base.eltype(::STICTWorkspace{Td}) where {Td} = Td

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
    p = _convert_params(Tf, params)
    ref = _allocate_zeros(M, Td, image_size)
    masks = _adapt_masks(ref, _tilp_masks(Tf, p.lp_filters, image_size, p.J))
    R = _runner_type(ref, Tf)
    # One full bank per depth, the half trees sharing its masks (`_half_bank`), and
    # one buffer pool for every runner: the channels run one after another.  The
    # runners' scratch `Z` is also the workspace's, which is never live meanwhile.
    flags = ref isa Array ? FFTW.MEASURE : FFTW.ESTIMATE
    banks = Dict{Int, _bank_type(ref, Tf)}()
    full(l) = get!(banks, l) do
        _adapt_bank(ref, _directional_bank(Tf, image_size, l, p.dfb_filters.f_ladder; decimated = true))
    end
    pool = _runner_pool(ref, Tf)
    built = Dict{Tuple{Int, Symbol}, R}()
    runner(l, half) = get!(built, (l, half)) do
        _bank_runner(_half_bank(full(l), full(1), half), ref; flags = flags, pool = pool)
    end
    runners = [Union{Nothing, R}[l == 0 ? nothing : runner(l, h) for h in _TILP_HALVES] for l in p.L_array]
    X = fill!(similar(ref, Complex{Tf}, image_size), zero(Complex{Tf}))
    pfwd, pbwd = _inplace_plans(X, flags)
    Z = isempty(pool) ? similar(X) : first(pool[(:Z, image_size)])
    return STICTWorkspace{Td, Tf, typeof(X), R, typeof(pfwd), typeof(pbwd)}(
        p, image_size, masks, runners, X, Z, pfwd, pbwd
    )
end

# ── Transforms ────────────────────────────────────────────────────────────────

function _stict_forward_ws!(coeffs::TICTCoefficients, x::AbstractMatrix, ws::STICTWorkspace)
    m = ws.masks
    ws.X .= x
    ws.pfwd * ws.X
    for i in eachindex(m.ana), c in 1:4
        out = coeffs.subbands[i][c]
        r = ws.runners[i][c]
        if r === nothing
            ws.Z .= m.ana[i][c] .* ws.X
            ws.pbwd * ws.Z
            _store!(out[1], ws.Z, _bwd_scale(ws.Z))
        else
            r.X .= m.ana[i][c] .* ws.X
            _bank_analyze_spectrum!(out, r)
        end
    end
    ws.Z .= m.coarse_ana .* ws.X
    ws.pbwd * ws.Z
    _store!(coeffs.coarse, ws.Z, _bwd_scale(ws.Z))
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
    ws.pbwd * ws.X
    return _store!(x, ws.X, _bwd_scale(ws.X))
end

"""
    stict_forward(image, params::ContourletParams; workspace=nothing)
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
        workspace::Union{Nothing, STICTWorkspace} = nothing
    )
    ws = workspace === nothing ? _own_workspace(make_stict_workspace, image, params) : workspace
    x = _check_tict_input(ws, _ws_input(ws, image))
    coeffs = similar_stict_coefficients(ws.params, ws.image_size; Td = eltype(ws), M = typeof(similar(x)))
    return _stict_forward_ws!(coeffs, x, ws)
end

"""
    stict_forward!(coeffs, image, params; workspace=nothing) -> coeffs

In-place [`stict_forward`](@ref) into coefficients from
[`similar_stict_coefficients`](@ref).  With a workspace it allocates nothing.
"""
function stict_forward!(
        coeffs::TICTCoefficients, image::AbstractMatrix, params::ContourletParams;
        workspace::Union{Nothing, STICTWorkspace} = nothing
    )
    ws = workspace === nothing ? _own_workspace(make_stict_workspace, image, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    return _stict_forward_ws!(coeffs, _check_tict_input(ws, _ws_input(ws, image)), ws)
end

"""
    stict_inverse(coeffs::TICTCoefficients, params; workspace=nothing) -> image

Inverse of [`stict_forward`](@ref): exact reconstruction (to rounding error).
"""
function stict_inverse(
        coeffs::TICTCoefficients, params::ContourletParams;
        workspace::Union{Nothing, STICTWorkspace} = nothing
    )
    ws = workspace === nothing ? _own_workspace(make_stict_workspace, coeffs.coarse, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    return _stict_inverse_ws!(similar(coeffs.coarse, eltype(ws)), coeffs, ws)
end

"""
    stict_inverse!(image, coeffs, params; workspace=nothing) -> image

In-place [`stict_inverse`](@ref).  With a workspace it allocates nothing.
"""
function stict_inverse!(
        image::AbstractMatrix, coeffs::TICTCoefficients, params::ContourletParams;
        workspace::Union{Nothing, STICTWorkspace} = nothing
    )
    ws = workspace === nothing ? _own_workspace(make_stict_workspace, image, params) : workspace
    _check_tict_eltype(ws, coeffs.coarse, "coefficient set")
    _check_tict_eltype(ws, image, "output image")
    return _stict_inverse_ws!(_check_tict_input(ws, image), coeffs, ws)
end
