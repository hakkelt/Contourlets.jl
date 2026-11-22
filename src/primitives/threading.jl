"""
    use_threading(threaded::Union{Bool, Nothing}, T::Type) -> Bool

Resolve a transform's `threaded` keyword for data of element type `T`: `true`
and `false` are taken as given; `nothing` (let the transform decide) threads
complex data only.  The result decides whether the mask-bank engine runs its
subband loop threaded.
"""
function use_threading end

@inline use_threading(threaded::Bool, ::Type) = threaded
@inline use_threading(::Nothing, ::Type{T}) where {T} = !(T <: Real)
