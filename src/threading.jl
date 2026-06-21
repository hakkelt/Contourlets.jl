# ── Threading Policy API ────────────────────────────────────────────────────────

"""
    ThreadingPolicy

Abstract type for configuring the threading behavior of transforms.
Available policies are:
- [`Auto`](@ref)
- [`Enabled`](@ref)
- [`Disabled`](@ref)
"""
abstract type ThreadingPolicy end

"""
    Auto <: ThreadingPolicy

Automatically selects the optimal threading strategy based on the data type.
By default, multithreading is disabled for `Real` data (which uses SIMD) and 
enabled for `Complex` data.
"""
struct Auto <: ThreadingPolicy end

"""
    Enabled <: ThreadingPolicy

Explicitly enables multithreading for loops.
"""
struct Enabled <: ThreadingPolicy end

"""
    Disabled <: ThreadingPolicy

Explicitly disables multithreading.
"""
struct Disabled <: ThreadingPolicy end

# Resolving whether to use threading based on type:
@inline _use_threading(::Enabled, ::Type{T}) where {T} = true
@inline _use_threading(::Disabled, ::Type{T}) where {T} = false
@inline function _use_threading(::Auto, ::Type{T}) where {T}
    # By default, disabled for Real, enabled for Complex.
    return !(T <: Real)
end
