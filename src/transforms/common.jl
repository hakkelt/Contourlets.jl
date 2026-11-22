# What the four transforms share: the subband layouts, the checks of the
# coefficients, the subbands of one level of a `SubbandVector` as a vector
# (`_Bands`), and the stacked-coefficient paths of the NSCT and the TICT, whose
# workspaces run MultiscaleFilterBanks' mask-bank engine.

const _StackedTransform = Union{NSCT, TICT}
const _RaggedTransform = Union{CT, STICT}

# ── Subband layouts ───────────────────────────────────────────────────────────

# CT / NSCT: the lowpass, then the directions of each scale, coarsest first.
_scale_layout(t::Union{CT, NSCT}) =
    [(scale = 0, direction = 0); [(scale = j, direction = k) for j in 1:t.scales for k in 1:(1 << t.direction_levels[j])]]

# TICT / STICT: the lowpass, then the directions of each channel of each level,
# coarsest level first.
_channel_layout(t::Union{TICT, STICT}) = [
    (level = 0, channel = 0, direction = 0);
    [(level = i, channel = c, direction = k) for (i, l) in enumerate(t.direction_levels) for c in 1:4 for k in 1:_tict_n_subbands(l, c)]
]

"""
    subband_layout(t::Union{CT, NSCT}, dims) -> Vector{@NamedTuple{scale::Int, direction::Int}}
    subband_layout(t::Union{TICT, STICT}, dims) -> Vector{@NamedTuple{level::Int, channel::Int, direction::Int}}

What each subband of the contourlet transform `t` is, in the order of
[`subband`](@ref)'s index.  The lowpass comes first, as `scale = 0` (or
`level = 0`) with `direction = 0` (and `channel = 0`).  Then, for each
scale from the coarsest (1) to the finest (`scales`), the directional
subbands `(scale = j, direction = k)` of the [`CT`](@ref) and [`NSCT`](@ref),
`k = 1, …, 2^direction_levels[j]`; for the [`TICT`](@ref) and [`STICT`](@ref), channel
`c = 1, …, 4` of each level in turn, `(level = i, channel = c, direction = k)`,
with `2^direction_levels[i]` directions on channels 1 and 4 and `2^(direction_levels[i] - 1) + 1`
on channels 2 and 3 (one direction on every channel where `direction_levels[i] = 0`).

The layout does not depend on `dims`, which must be the size of an image (two
axes).

# Examples
```jldoctest
julia> using Contourlets

julia> subband_layout(CT(scales = 2, direction_levels = [0, 1]), (32, 32))
4-element Vector{@NamedTuple{scale::Int64, direction::Int64}}:
 (scale = 0, direction = 0)
 (scale = 1, direction = 1)
 (scale = 2, direction = 1)
 (scale = 2, direction = 2)
```
"""
MultiscaleTransformsBase.subband_layout(t::Union{CT, NSCT}, dims::Dims) = (_check_image(dims); _scale_layout(t))
MultiscaleTransformsBase.subband_layout(t::Union{TICT, STICT}, dims::Dims) = (_check_image(dims); _channel_layout(t))

MultiscaleTransformsBase.nsubbands(t::Union{CT, NSCT}, dims::Dims) = (_check_image(dims); 1 + sum(l -> 1 << l, t.direction_levels))
MultiscaleTransformsBase.nsubbands(t::Union{TICT, STICT}, dims::Dims) = (_check_image(dims); 1 + sum(l -> sum(c -> _tict_n_subbands(l, c), 1:4), t.direction_levels))

"""
    subband(t::Union{CT, NSCT, TICT, STICT}, c, i) -> AbstractMatrix

Subband `i` of the coefficients `c` of `t`, as a view (writing to it changes
`c`); [`subband_layout`](@ref) says which subband `i` is.  For the
[`NSCT`](@ref) and [`TICT`](@ref) it is `view(c, :, :, i)`, for the
[`CT`](@ref) and [`STICT`](@ref) `subband(c, i)` of the [`SubbandVector`](@ref).
"""
MultiscaleTransformsBase.subband(::_StackedTransform, c::AbstractArray{<:Any, 3}, i::Integer) = view(c, :, :, i)
MultiscaleTransformsBase.subband(::_RaggedTransform, c::SubbandVector, i::Integer) = subband(c, i)

MultiscaleTransformsBase.signal_size(::_StackedTransform, c::AbstractArray{<:Any, 3}) = (size(c, 1), size(c, 2))

"""
    similar_coefficients(t::Union{CT, NSCT, TICT, STICT}, x::AbstractMatrix)

Zero coefficients of `t` for images like `x` (size, element type `float(eltype(x))`
and array type, so device), for [`analyze!`](@ref): an array of size
`(size(x)..., n)` for the [`NSCT`](@ref) and [`TICT`](@ref), a
[`SubbandVector`](@ref) for the [`CT`](@ref) and [`STICT`](@ref).
"""
function MultiscaleTransformsBase.similar_coefficients(t::_StackedTransform, x::AbstractArray)
    dims = _check_image(size(x))
    return _zero_stack(x, data_eltype(x), dims, nsubbands(t, dims))
end

function MultiscaleTransformsBase.similar_coefficients(t::_RaggedTransform, x::AbstractArray)
    Td = data_eltype(x)
    layout = first(_layout(t, _check_image(size(x))))
    return SubbandVector(fill!(similar(x, Td, length(layout)), zero(Td)), layout)
end

# ── Checks ────────────────────────────────────────────────────────────────────

function _check_eltype(what::String, A::AbstractArray, ::Type{Td}) where {Td}
    eltype(A) === Td ||
        throw(ArgumentError("the $(what) has element type $(eltype(A)), but the data needs $(Td)"))
    return A
end

function _check_stack(c::AbstractArray, dims::NTuple{2, Int}, n::Int)
    size(c) == (dims..., n) || throw(
        DimensionMismatch("coefficients of size $(size(c)) do not fit an image of size $(dims) with $(n) subbands, of size $((dims..., n))")
    )
    return c
end

function _check_layout(c::SubbandVector, layout::SubbandLayout)
    c.layout === layout || c.layout == layout || throw(
        DimensionMismatch("the coefficients' subband layout (data of size $(c.layout.signal_size)) does not fit this transform on data of size $(layout.signal_size)")
    )
    return c
end

# ── Subbands of one level of a SubbandVector ─────────────────────────────────

# Subbands `first`, …, `first + n - 1` of `c`, as the vector of matrices the
# directional filter banks write and read.  Built per call without allocating.
struct _Bands{A <: AbstractMatrix, C <: SubbandVector} <: AbstractVector{A}
    c::C
    first::Int
    n::Int
end

_Bands(c::SubbandVector, first::Int, n::Int) = _Bands{typeof(subband(c, first)), typeof(c)}(c, first, n)

Base.size(b::_Bands) = (b.n,)
Base.@propagate_inbounds function Base.getindex(b::_Bands, k::Int)
    @boundscheck checkbounds(b, k)
    return subband(b.c, b.first + k - 1)
end

# ── Images of the ragged transforms (CT, STICT) ──────────────────────────────

# An array like the workspace's own: a CT workspace's matrix type (empty, so it
# allocates nothing), an STICT workspace's spectrum buffer.
_ws_proto(::CTWorkspace{Td, Tf, M}) where {Td, Tf, M} = _allocate_zeros(M, Td, (0, 0))
_ws_proto(ws::STICTWorkspace) = ws.X

# Whether the data of `x` lives elsewhere than the workspace's: a host image for
# a device workspace, or the reverse.  Such images are copied through an array
# like the workspace's.
_foreign(ws, x::AbstractMatrix) = is_device_array(x) != is_device_array(_ws_proto(ws))

_stage_in(ws, x::AbstractMatrix) = _foreign(ws, x) ? copyto!(similar(_ws_proto(ws), eltype(ws), size(x)), x) : x

# Run `f(y)`, which writes the image into `y`, for the output `x`.
_stage_out!(f::F, ws, x::AbstractMatrix) where {F} =
    _foreign(ws, x) ? copyto!(x, f(similar(_ws_proto(ws), eltype(ws), size(x)))) : f(x)

# ── Stacked coefficients (NSCT, TICT) ─────────────────────────────────────────

const _StackedWorkspace = Union{NSCTWorkspace, TICTWorkspace}

# The workspace's FFT plans run on its dense matrix type; a view, another
# wrapper or a host array for a device workspace is staged through a grow-once
# scratch buffer of that type (`fwd_scratch` for inputs, `inv_scratch` for
# outputs).
function _staging(a::ScratchArena, ws, x::AbstractMatrix)
    _arena_reset!(a)
    return _arena_take!(a, ws.stack, size(x)...)
end
_dense_input(ws, x::AbstractMatrix) =
    x isa _ws_matrix_type(ws) ? x : copyto!(_staging(ws.fwd_scratch, ws, x), x)

# Run `f(y)`, which writes into a dense `y`, for the output `out`.
function _dense_output!(out::AbstractMatrix, ws, f::F) where {F}
    out isa _ws_matrix_type(ws) && return f(out)
    return copyto!(out, f(_staging(ws.inv_scratch, ws, out)))
end

# Analysis of `x` into the stacked coefficients `c` through the engine of `ws`.
# Coefficients of another array type than the workspace's (a view, host
# coefficients for a device workspace) go through a temporary copy.
function _stacked_forward!(c::AbstractArray, x::AbstractArray, ws::_StackedWorkspace, threaded::Union{Bool, Nothing})
    T = eltype(ws)
    img = _dense_input(ws, T === eltype(x) ? x : T.(x))
    th = use_threading(threaded, T)
    if c isa typeof(ws.stack)
        mask_bank_forward!(_Stacked(c, ws.offsets), img, ws.engine; threaded = th)
    else
        tmp = similar(ws.stack, size(c))
        mask_bank_forward!(_Stacked(tmp, ws.offsets), img, ws.engine; threaded = th)
        copyto!(c, tmp)
    end
    return c
end

function _stacked_inverse!(x::AbstractArray, c::AbstractArray, ws::_StackedWorkspace, threaded::Union{Bool, Nothing})
    th = use_threading(threaded, eltype(ws))
    s = c isa typeof(ws.stack) ? c : copyto!(similar(ws.stack, size(c)), c)
    _dense_output!(x, ws, y -> mask_bank_inverse!(y, _Stacked(s, ws.offsets), ws.engine; threaded = th))
    return x
end

# Run one analysis and synthesis on zeros, threaded as a call with the default
# `threaded = nothing`, so the first real call is compiled and allocates nothing.
function _prewarm_stacked!(ws::_StackedWorkspace)
    x = _allocate_zeros(_ws_matrix_type(ws), eltype(ws), ws.image_size)
    c = _zero_stack(x, eltype(ws), ws.image_size, ws.nsubbands)
    _stacked_forward!(c, x, ws, nothing)
    _stacked_inverse!(x, c, ws, nothing)
    return ws
end

"""
    analyze!(c, t::Union{CT, NSCT, TICT, STICT}, x::AbstractMatrix; workspace = nothing, threaded = nothing) -> c

Analysis of the image `x` by the contourlet transform `t` into the
coefficients `c` from [`similar_coefficients`](@ref) (or an earlier
[`analyze`](@ref)), whose element type must be `float(eltype(x))`.  With a
`workspace` from [`make_workspace`](@ref) for `t`, `eltype(c)` and `size(x)`
the call allocates nothing (a few hundred bytes for the tasks when threading is
on); without one, the filters and plans are built for the call.  A workspace
for another transform, element type or image size throws.

`threaded = true` splits the independent subband FFTs of an [`NSCT`](@ref) or
[`TICT`](@ref) host workspace between its tasks (one per Julia thread unless
the workspace was built with fewer `tasks`), `false` runs them
serially, and `nothing` (the default) threads complex data only.  The
[`CT`](@ref) and [`STICT`](@ref) run serially and ignore it.
"""
function MultiscaleTransformsBase.analyze!(
        c::AbstractArray, t::_StackedTransform, x::AbstractArray;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    dims = _check_image(size(x))
    Td = data_eltype(x)
    workspace === nothing || check_workspace(workspace, t, Td, dims)
    _check_eltype("coefficient array", _check_stack(c, dims, nsubbands(t, dims)), Td)
    workspace === nothing && return _analyze_direct!(c, t, x, threaded)
    return _stacked_forward!(c, x, workspace, threaded)
end

"""
    synthesize!(x, t::Union{CT, NSCT, TICT, STICT}, c; workspace = nothing, threaded = nothing) -> x

Synthesis of the coefficients `c` of `t` into the image `x`: the inverse of
[`analyze!`](@ref), exact up to rounding.  With a `workspace` for `t`,
`eltype(x)` and `size(x)` (and coefficients of that element type) it allocates
nothing.  `threaded` as for [`analyze!`](@ref).

The transforms are frames, not tight frames: the synthesis is the inverse of
the analysis, not its adjoint.
"""
function MultiscaleTransformsBase.synthesize!(
        x::AbstractArray, t::_StackedTransform, c::AbstractArray;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    dims = _check_image(size(x))
    _check_stack(c, dims, nsubbands(t, dims))
    workspace === nothing && return _synthesize_direct!(x, t, c, threaded)
    check_workspace(workspace, t, eltype(x), dims)
    _check_eltype("coefficient array", c, eltype(x))
    return _stacked_inverse!(x, c, workspace, threaded)
end

# ── SubbandVector coefficients (CT, STICT) ────────────────────────────────────

function MultiscaleTransformsBase.analyze!(
        c::SubbandVector, t::_RaggedTransform, x::AbstractArray;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    return _ragged_analyze!(c, t, x, workspace, true)
end

# Not `analyze!`: the coefficients are new on every call, so a device graph
# captured with them would never replay (`ContourletsCUDAExt`).
function MultiscaleTransformsBase.analyze(
        t::_RaggedTransform, x::AbstractArray;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    return _ragged_analyze!(similar_coefficients(t, x), t, x, workspace, false)
end

function _ragged_analyze!(c::SubbandVector, t::_RaggedTransform, x::AbstractArray, workspace, capture::Bool)
    dims = _check_image(size(x))
    Td = data_eltype(x)
    if workspace === nothing
        layout, first = _layout(t, dims)
        _check_eltype("coefficient vector", _check_layout(c, layout), Td)
        return _analyze_direct!(c, first, t, x)
    end
    check_workspace(workspace, t, Td, dims)
    _check_eltype("coefficient vector", _check_layout(c, workspace.layout), Td)
    return _ragged_forward!(c, x, workspace, capture)
end

function MultiscaleTransformsBase.synthesize!(
        x::AbstractArray, t::_RaggedTransform, c::SubbandVector;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    return _ragged_synthesize!(x, t, c, workspace, true)
end

function MultiscaleTransformsBase.synthesize(
        t::_RaggedTransform, c::SubbandVector;
        workspace = nothing, threaded::Union{Bool, Nothing} = nothing
    )
    T = workspace === nothing ? eltype(c) : eltype(workspace)
    x = similar(c.data, T, c.layout.signal_size)
    return _ragged_synthesize!(x, t, c, workspace, false)
end

function _ragged_synthesize!(x::AbstractArray, t::_RaggedTransform, c::SubbandVector, workspace, capture::Bool)
    dims = _check_image(size(x))
    if workspace === nothing
        layout, first = _layout(t, dims)
        _check_layout(c, layout)
        return _synthesize_direct!(x, first, t, c)
    end
    check_workspace(workspace, t, eltype(x), dims)
    _check_eltype("coefficient vector", _check_layout(c, workspace.layout), eltype(x))
    return _ragged_inverse!(x, c, workspace, capture)
end
