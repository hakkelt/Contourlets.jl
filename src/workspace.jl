# ContourletWorkspace — preallocated buffers for zero-per-iteration allocations.
#
# Design goals:
#   1. All temporary matrices are allocated once at construction time.
#   2. The mutating entry points ct_forward! / ct_inverse! / nsct_forward! /
#      nsct_inverse! accept a ContourletWorkspace and reuse its buffers.
#   3. Thread-safety: one workspace per thread; NOT safe for concurrent reuse.
#   4. Backward compatibility: non-! functions always allocate; workspace is opt-in.

# ── Scratch arena ─────────────────────────────────────────────────────────────
# A grow-once pool of `Matrix{T}` scratch buffers for the directional stage,
# whose transient buffers vary in number/size per node.  The allocation sequence
# is deterministic for fixed params + image size, so `reset!` at the start of a
# transform and `acquire!` hand back the same buffers on every subsequent call —
# zero allocations after the first.  Buffers are handed out by a monotonic cursor
# (never released mid-transform), so all live intermediates are distinct.

# Parametrised on the stored matrix type `M` so the same pool serves host
# (`Matrix`) and device (`CuMatrix`, …) buffers; new buffers are made with
# `similar(ref, …)` from a caller-supplied reference array.
mutable struct ScratchArena{M <: AbstractMatrix}
    bufs::Vector{M}
    cursor::Int
end
ScratchArena{M}() where {M <: AbstractMatrix} = ScratchArena{M}(M[], 0)

_arena_reset!(a::ScratchArena) = (a.cursor = 0; return a)

# Hand out a scratch `m×n` matrix shaped like `ref`, growing/resizing as needed.
function _arena_take!(a::ScratchArena{M}, ref, m::Int, n::Int) where {M}
    a.cursor += 1
    if a.cursor > length(a.bufs)
        push!(a.bufs, similar(ref, m, n)::M)
    elseif size(a.bufs[a.cursor]) != (m, n)
        a.bufs[a.cursor] = similar(ref, m, n)::M
    end
    return a.bufs[a.cursor]
end

# Task-local "active" arena.  Rather than thread an arena argument through every
# call, the pyramid stage of the workspace paths (and the device separable
# convolution's scratch) draws transient buffers from whatever arena is active on
# the current task, via `_scratch_like`.  When no matching arena is active
# `_scratch_like` falls back to `similar` (so the non-workspace paths are
# unchanged).  The arena is typed on its matrix type, so a host arena only serves
# host arrays.  Task-local ⇒ thread-safe.

_active_arena() = get(task_local_storage(), :_contourlets_scratch, nothing)

_active_filter_cache() = get(task_local_storage(), :_contourlets_filter_cache, nothing)

function _with_filter_cache(f, cache)
    store = task_local_storage()
    prev = get(store, :_contourlets_filter_cache, nothing)
    store[:_contourlets_filter_cache] = cache
    try
        return f()
    finally
        store[:_contourlets_filter_cache] = prev
    end
end

function _with_arena(f, arena)
    store = task_local_storage()
    prev = get(store, :_contourlets_scratch, nothing)
    store[:_contourlets_scratch] = arena
    try
        return f()
    finally
        store[:_contourlets_scratch] = prev
    end
end

# A scratch `m×n` matrix shaped like `ref`: drawn from the active arena when its
# stored matrix type matches `typeof(ref)`, otherwise `similar(ref, m, n)`.  The
# type match means a host arena only ever serves host arrays and a device arena
# only device arrays, so mixed/absent arenas fall back to `similar` safely.
_base_array_type(::Type{T}) where {T} = T
_base_array_type(::Type{SubArray{T, N, P, I, L}}) where {T, N, P, I, L} = _base_array_type(P)

function _scratch_like(ref::R, m::Int, n::Int) where {R <: AbstractMatrix}
    a = _active_arena()
    B = _base_array_type(R)
    if a isa ScratchArena{B}
        return _arena_take!(a, ref, m, n)
    end
    return similar(ref, m, n)
end

function _allocate_zeros(::Type{M}, Td::Type, dims::Tuple) where {M <: AbstractMatrix}
    T_concrete = M isa UnionAll ? M{Td} : M
    if T_concrete <: Array
        return zeros(Td, dims...)
    end
    dummy = similar(T_concrete, (1, 1))
    arr = similar(dummy, Td, dims)
    return fill!(arr, zero(Td))
end


"""
    ContourletWorkspace{Td, Tf}

Preallocated temporary buffers (data type `Td`, real filter precision `Tf`) for the
Contourlet and NSCT transforms.  Create with `make_workspace`; reuse across
iterations with the mutating `!` entry points.

!!! warning "Thread safety"
    A `ContourletWorkspace` is **not thread-safe**.  Create a separate workspace
    per thread (e.g., `[make_workspace(params, sz) for _ in 1:Threads.nthreads()]`).
"""
struct ContourletWorkspace{Td <: Number, Tf <: AbstractFloat, M <: AbstractMatrix{Td}, R, B, C, P, Q}
    params::ContourletParams{Tf}     # real filter precision Tf
    image_size::Tuple{Int, Int}

    # Grow-once pool for all temporary buffers (LP coarse/bandpass, transients).
    # Separate arenas for forward and inverse passes because they traverse the
    # pyramid levels in opposite orders, yielding different buffer size sequences.
    fwd_scratch::ScratchArena{M}
    inv_scratch::ScratchArena{M}

    # Size of the bandpass image at each pyramid level.
    bp_dims::Vector{Tuple{Int, Int}}

    # CT: per-level DFB runner (equivalent filters, FFT buffers and plans) for the
    # ladder DFB; `nothing` at levels without a directional split, and empty for an
    # NSCT workspace or a modulation-mode filter pair.
    dfb_runners::R

    # NSCT: per-level NSDFB bank (zero-phase equivalent filters, cropped to the
    # rfft half-plane for real data).  Empty for a CT workspace.
    nsdfb_banks::B

    # NSCT: the nonsubsampled pyramid as masks on the input's spectrum (see
    # `_nsp_masks`): `nsp_ana[j]` takes the input spectrum to the bandpass of level
    # j, `nsp_syn[j]` takes that bandpass back, and entry J + 1 serves the lowpass.
    # Empty for a CT workspace.
    nsp_ana::Vector{C}
    nsp_syn::Vector{C}

    # NSCT FFT stage: single-threaded plans between the image and its spectrum,
    # the input spectrum / synthesis accumulator, and per task (one per thread on
    # the host) an FFT buffer and, from task 2 on, a partial synthesis sum.  The
    # outputs `(j, k)` in `nsct_items` are split between the tasks.  `nothing` /
    # empty for a CT workspace.
    fft_plan_fwd::P
    fft_plan_inv::Q
    fft_buffer_c::C                 # === fft_tasks[1]
    fft_spectrum_bp::C
    fft_tasks::Vector{C}
    fft_acc::Vector{C}
    nsct_items::Vector{NTuple{2, Int}}

    # Mutable slots for device-accelerated graph execs (e.g. CUDA.CuGraphExec).
    # Layout: [] = uncaptured; [exec, img_ptr::UInt, coarse_ptr::UInt].
    # Populated by ContourletsCUDAExt on first ct_forward!/ct_inverse! call.
    graph_cache::Vector{Any}
    inv_graph_cache::Vector{Any}
end

function ContourletWorkspace{Td, Tf, M}(
        params::ContourletParams{Tf},
        image_size::Tuple{Int, Int},
        fwd_scratch::ScratchArena{M},
        inv_scratch::ScratchArena{M},
        bp_dims::Vector{Tuple{Int, Int}},
        dfb_runners::R,
        nsdfb_banks::B,
        nsp_ana::Vector{C},
        nsp_syn::Vector{C},
        fft_plan_fwd::P,
        fft_plan_inv::Q,
        fft_tasks::Vector{C},
        fft_acc::Vector{C},
        fft_spectrum_bp::C,
        nsct_items::Vector{NTuple{2, Int}},
        graph_cache::Vector{Any} = Any[],
        inv_graph_cache::Vector{Any} = Any[]
    ) where {Td, Tf, M, R, B, C, P, Q}
    first_task = isempty(fft_tasks) ? fft_spectrum_bp : fft_tasks[1]
    return ContourletWorkspace{Td, Tf, M, R, B, C, P, Q}(
        params, image_size, fwd_scratch, inv_scratch, bp_dims, dfb_runners, nsdfb_banks,
        nsp_ana, nsp_syn, fft_plan_fwd, fft_plan_inv, first_task, fft_spectrum_bp,
        fft_tasks, fft_acc, nsct_items, graph_cache, inv_graph_cache
    )
end

# Bandpass size at each of the `J` pyramid levels of the CT (halving, rounded up)
# or the NSCT (constant).
function _bandpass_dims(image_size::Tuple{Int, Int}, J::Int; nonsubsampled::Bool)
    dims = Vector{Tuple{Int, Int}}(undef, J)
    d = image_size
    for j in 1:J
        dims[j] = d
        nonsubsampled || (d = cld.(d, 2))
    end
    return dims
end

# The bank and runner types for arrays like `ref` at precision `Tf`, so that the
# workspace has one concrete type whatever the filter pair and depths (a tiny
# throwaway plan supplies the plan types).
_bank_type(ref::AbstractArray, ::Type{Tf}) where {Tf} =
    _DirectionalBank{typeof(similar(ref, Tf, 0, 0)), typeof(similar(ref, Complex{Tf}, 0))}

function _runner_type(ref::AbstractArray, ::Type{Tf}) where {Tf}
    C = typeof(similar(ref, Complex{Tf}, 0, 0))
    P, Q = typeof.(_inplace_plans(similar(ref, Complex{Tf}, 2, 2), FFTW.ESTIMATE))
    return _BankRunner{_bank_type(ref, Tf), C, P, Q}
end

# Per-level DFB runners of a CT workspace (`nothing` where a level is not split).
function _dfb_level_runners(ref::AbstractMatrix, p::ContourletParams, bp_dims, ::Type{Tf}) where {Tf}
    R = Union{Nothing, _runner_type(ref, Tf)}
    is_ladder(p.dfb_filters) || return R[]
    flags = ref isa Array ? FFTW.MEASURE : FFTW.ESTIMATE
    return R[
        l > 0 ? _bank_runner(
                _adapt_bank(ref, _directional_bank(Tf, dims, l, p.dfb_filters.f_ladder; decimated = true)),
                ref; flags = flags
            ) : nothing
            for (l, dims) in zip(p.L_array, bp_dims)
    ]
end

# The DFB runner a CT workspace holds for a bandpass of size `dims` split `l` deep.
function _workspace_runner(ws::ContourletWorkspace, dims, l::Int, half::Symbol)
    half === :both || return nothing
    for (j, r) in enumerate(ws.dfb_runners)
        r === nothing && continue
        (ws.bp_dims[j] == dims && r.bank.levels == l) && return r
    end
    return nothing
end

Base.eltype(::ContourletWorkspace{Td}) where {Td} = Td

"""
    make_workspace(params::ContourletParams, image_size::Tuple{Int,Int}; T=Float64)
    make_workspace(T::Type, image_size::Tuple{Int,Int}, params::ContourletParams)
                   -> ContourletWorkspace

Allocate a `ContourletWorkspace` for images of `image_size` with the given `params`.
The second form is a convenience overload with the element type as the first positional argument.

# Examples
```jldoctest
julia> using Contourlets

julia> p = ContourletParams(J = 2, L_array = [1, 2]);

julia> ws = make_workspace(p, (64, 64));

julia> ws isa ContourletWorkspace
true
```
"""
make_workspace(
    T::Type{<:Number}, image_size::Tuple{Int, Int},
    params::ContourletParams; kwargs...
) =
    make_workspace(params, image_size; T = T, kwargs...)

"""
    make_workspace(image::AbstractMatrix, params::ContourletParams) -> ContourletWorkspace

Allocate a `ContourletWorkspace` using the array type and element type of `image`.
This is required for allocation-free execution on GPU arrays.
"""
make_workspace(
    image::AbstractMatrix, params::ContourletParams{Tp}; kwargs...
) where {Tp} =
    make_workspace(params, size(image); T = eltype(image), M = typeof(similar(image, promote_type(Tp, eltype(image)), 0, 0)), kwargs...)

function make_workspace(
        params::ContourletParams{Tp},
        image_size::Tuple{Int, Int};
        T::Type{<:Number} = Float64,
        M::Type{<:AbstractMatrix} = Matrix{promote_type(Tp, T)},
        prewarm::Bool = true
    ) where {Tp}
    Td = promote_type(Tp, T)         # data buffer type (may be complex)
    Tf = _filter_eltype(Td)          # real filter precision
    p2 = _convert_params(Tf, params)
    bp_dims = _bandpass_dims(image_size, p2.J; nonsubsampled = false)
    ref = _allocate_zeros(M, Td, (1, 1))
    # The DFB equivalent filters, FFT buffers and plans of every level are built
    # once here (the allocating API rebuilds them per call).
    runners = _dfb_level_runners(ref, p2, bp_dims, Tf)
    ws = ContourletWorkspace{Td, Tf, M}(
        p2, image_size, ScratchArena{M}(), ScratchArena{M}(), bp_dims,
        runners, Union{Nothing, _bank_type(ref, Tf)}[], Nothing[], Nothing[],
        nothing, nothing, Nothing[], Nothing[], nothing, NTuple{2, Int}[]
    )
    # Grow the scratch arena to its steady-state size now (forward + inverse), so
    # the first user call is already allocation-free.
    prewarm && _prewarm_ct!(ws)
    return ws
end

"""
    make_nsct_workspace(params::ContourletParams, image_size::Tuple{Int,Int}; T=Float64,
                        threading=Auto()) -> ContourletWorkspace
    make_nsct_workspace(T::Type, image_size::Tuple{Int,Int}, params::ContourletParams)
                        -> ContourletWorkspace

Allocate a workspace for the NSCT transforms.  All buffers have the full `image_size`
since the NSCT performs no downsampling.
The second form is a convenience overload with the element type as the first positional argument.

!!! note "Threading"
    A host workspace holds one FFT buffer per Julia thread, and `nsct_forward!` /
    `nsct_inverse!` split their independent subband FFTs between threads when their
    own `threading` policy says so, call by call.  The `threading` kwarg here only
    applies to the warm-up pass.
"""
make_nsct_workspace(
    T::Type{<:Number}, image_size::Tuple{Int, Int},
    params::ContourletParams; kwargs...
) =
    make_nsct_workspace(params, image_size; T = T, kwargs...)

"""
    make_nsct_workspace(image::AbstractMatrix, params::ContourletParams) -> ContourletWorkspace

Allocate a workspace for the NSCT transforms using the array type and element type of `image`.
This is required for allocation-free execution on GPU arrays.
"""
make_nsct_workspace(
    image::AbstractMatrix, params::ContourletParams{Tp}; kwargs...
) where {Tp} =
    make_nsct_workspace(params, size(image); T = eltype(image), M = typeof(similar(image, promote_type(Tp, eltype(image)), 0, 0)), kwargs...)

function make_nsct_workspace(
        params::ContourletParams{Tp},
        image_size::Tuple{Int, Int};
        T::Type{<:Number} = Float64,
        M::Type{<:AbstractMatrix} = Matrix{promote_type(Tp, T)},
        prewarm::Bool = true,
        threading::ThreadingPolicy = Auto()
    ) where {Tp}
    Td = promote_type(Tp, T)         # data buffer type (may be complex)
    Tf = _filter_eltype(Td)          # real filter precision
    p2 = _convert_params(Tf, params)
    J = p2.J
    # Precompute the NSDFB equivalent filters (the bank depends on the pyramid level
    # only through its depth and its filter upsampling) and the pyramid masks; the
    # `!` paths then never rebuild them per call.
    is_real = Td <: Real
    ref = _allocate_zeros(M, Td, (1, 1))
    keys_j = [(p2.L_array[j], _nsdfb_upsampling(j)) for j in 1:J]
    banks = Dict{NTuple{2, Int}, Any}()
    for (l, m) in unique(keys_j)
        l > 0 || continue
        is_ladder(p2.dfb_filters) || throw(
            ArgumentError("the NSDFB is built from the ladder of a ladder QuincunxFilterPair such as Q2345")
        )
        banks[(l, m)] = _adapt_bank(
            ref, _directional_bank(
                Tf, image_size, l, p2.dfb_filters.f_ladder;
                decimated = false, real_data = is_real, upsample = m
            )
        )
    end
    # Levels with equal depth and upsampling share one bank.
    nsdfb_banks = Union{Nothing, _bank_type(ref, Tf)}[get(banks, key, nothing) for key in keys_j]
    n1, n2 = image_size
    init_shape = is_real ? (n1 ÷ 2 + 1, n2) : (n1, n2)
    nsp_ana, nsp_syn = _nsp_masks(Tf, p2.lp_filters, image_size, J; rows = 1:init_shape[1])
    fft_buffer_c = fill!(similar(ref, Complex{Tf}, init_shape), zero(Complex{Tf}))
    nsp_ana = [_like(fft_buffer_c, a) for a in nsp_ana]
    nsp_syn = [_like(fft_buffer_c, a) for a in nsp_syn]
    fft_spectrum_bp = similar(fft_buffer_c)
    tmp_d = similar(ref, Td, image_size)

    if M <: Array
        # Single-threaded plans: threading splits the independent subband FFTs
        # between tasks instead (see `_nsct_forward_inner!`), which at these sizes
        # scales far better than FFTW's threads splitting one FFT.
        nthreads_before = FFTW.get_num_threads()
        FFTW.set_num_threads(1)
        try
            fft_plan_fwd = is_real ? plan_rfft(tmp_d, flags = FFTW.MEASURE) : plan_fft(tmp_d, flags = FFTW.MEASURE)
            fft_plan_inv = is_real ? plan_irfft(fft_buffer_c, n1, flags = FFTW.MEASURE) : plan_ifft(fft_buffer_c, flags = FFTW.MEASURE)
        finally
            FFTW.set_num_threads(nthreads_before)
        end
    else
        fft_plan_fwd = is_real ? plan_rfft(tmp_d) : plan_fft(tmp_d)
        fft_plan_inv = is_real ? plan_irfft(fft_buffer_c, n1) : plan_ifft(fft_buffer_c)
    end
    fft_tasks = [i == 1 ? fft_buffer_c : similar(fft_buffer_c) for i in 1:_n_fft_tasks(fft_buffer_c)]
    items = [(j, k) for j in 1:J for k in 1:(nsdfb_banks[j] === nothing ? 1 : length(nsdfb_banks[j].ana))]

    ws = ContourletWorkspace{Td, Tf, M}(
        p2, image_size, ScratchArena{M}(), ScratchArena{M}(),
        _bandpass_dims(image_size, J; nonsubsampled = true),
        Union{Nothing, _runner_type(ref, Tf)}[], nsdfb_banks, nsp_ana, nsp_syn,
        fft_plan_fwd, fft_plan_inv, fft_tasks, typeof(fft_buffer_c)[similar(t) for t in fft_tasks[2:end]],
        fft_spectrum_bp, items
    )
    prewarm && _prewarm_nsct!(ws; threading = threading)
    return ws
end

"""
    estimate_workspace_size(params::ContourletParams, image_size;
                            nonsubsampled=false) -> Int

Return the total number of floating-point elements across all workspace buffers
(useful for memory planning before allocation).  Pass `nonsubsampled=true` for
workspaces created with [`make_nsct_workspace`](@ref), whose per-level buffers
all have the full image size.
"""
function estimate_workspace_size(
        params::ContourletParams,
        image_size::Tuple{Int, Int};
        nonsubsampled::Bool = false
    )::Int
    J = params.J
    n1, n2 = image_size
    total = 0
    if nonsubsampled
        total += 4 * J * n1 * n2          # coarse_j + bp_j + tmp_j + tmp2_j, full size each level
    else
        c1, c2 = n1, n2
        for j in 1:J
            d1, d2 = cld(c1, 2), cld(c2, 2)
            total += d1 * d2    # coarse buf
            total += c1 * c2    # bp buf
            c1, c2 = d1, d2
        end
    end
    total += 3 * n1 * n2    # tmp_buf + tmp_buf2 + current
    return total
end
