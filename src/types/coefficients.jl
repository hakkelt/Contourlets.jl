# Coefficient containers returned by the top-level transforms:
# ContourletCoefficients (CT), NSCTCoefficients (NSCT) and TICTCoefficients
# (TICT / STICT).  Each is parameterised on the data element type `Td` and the
# storage array type `A`, so coefficients stay on the device they were computed on.

"""
    ContourletCoefficients{Td, A}

Output of `ct_forward`.  `Td` is the data element type (real or complex) and
`A <: AbstractMatrix{Td}` the storage array type — so the coefficients live wherever
they were computed (a host `Matrix` for CPU input, a device array such as `CuMatrix`
for a GPU input — no implicit transfer).  Contains:
- `coarse`: the low-pass residual after `J` LP stages.
- `subbands`: `Vector` of length `J`; `subbands[j]` is a `Vector` of `2^L_array[j]`
  directional subband matrices.
"""
struct ContourletCoefficients{Td <: Number, A <: AbstractMatrix{Td}}
    coarse::A                      # storage array type A (host or device)
    subbands::Vector{Vector{A}}    # [scale][direction]
end

# Infer (Td, A) from the arguments.
ContourletCoefficients(coarse::A, subbands::Vector{Vector{A}}) where {A <: AbstractMatrix} =
    ContourletCoefficients{eltype(A), A}(coarse, subbands)

"""
    NSCTCoefficients{Td, A}

Output of `nsct_forward`.  Same structure as `ContourletCoefficients` (data type `Td`,
storage type `A`) but each subband matrix has the same spatial size as the input image
(no downsampling).

On a device, [`similar_nsct_coefficients`](@ref) and `Adapt.adapt` store the
subbands as the slices of one array (`storage`), which lets a workspace transform
them with batched FFTs in place.  Replacing a subband array is allowed; that
subband is then written through a copy.
"""
struct NSCTCoefficients{Td <: Number, A <: AbstractMatrix{Td}, S <: Union{Nothing, AbstractArray{Td, 3}}}
    coarse::A
    subbands::Vector{Vector{A}}
    # The subbands as the slices of one array, in order, or `nothing`; see
    # `_stacked_zeros`.
    storage::S
end

NSCTCoefficients(coarse::A, subbands::Vector{Vector{A}}, storage = nothing) where {A <: AbstractMatrix} =
    NSCTCoefficients{eltype(A), A, typeof(storage)}(coarse, subbands, storage)

"""
    TICTCoefficients{Td, A}

Output of [`tict_forward`](@ref) and [`stict_forward`](@ref).  `coarse` is the
lowpass image and `subbands[i][c][d]` the `d`-th directional subband of the `c`-th
of the four highpass channels at pyramid level `i` (`i = 1` finest) — Eslami &
Radha's `ρ_{i,k}^{(d)}` / `η_{i,k}^{(d)}`.  Channels 1 and 4 carry a full
directional filter bank (`2^L_array[i]` subbands), channels 2 and 3 a half tree
(`2^(L_array[i]-1) + 1` subbands; see [`tict_forward`](@ref)).  TICT subbands all
have the input's size; STICT subbands are critically sampled from full-size
channels.
"""
struct TICTCoefficients{Td <: Number, A <: AbstractMatrix{Td}, S <: Union{Nothing, AbstractArray{Td, 3}}}
    coarse::A
    subbands::Vector{Vector{Vector{A}}}    # [level][channel 1..4][direction]
    storage::S                             # as for `NSCTCoefficients`
end

TICTCoefficients(coarse::A, subbands::Vector{Vector{Vector{A}}}, storage = nothing) where {A <: AbstractMatrix} =
    TICTCoefficients{eltype(A), A, typeof(storage)}(coarse, subbands, storage)
