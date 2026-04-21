# l-level binary tree Directional Filter Bank (DFB).
#
# Algorithm (Do & Vetterli 2005, §V "Simplification Using Shearing"):
#
#   At depth d = 1 … l_levels, alternate shear direction and QFB split axis:
#     odd  depth  → horizontal shear (:h), QFB splits COLUMNS  (dir=:col)
#     even depth  → vertical   shear (:v), QFB splits ROWS     (dir=:row)
#
#   At each node:
#     1. shear(image, shear_dir)
#     2. (sb0, sb1) = qfb_decompose(sheared, qfp; dir=qfb_dir)
#     3. inv_shear!(sb0, sb0, shear_dir)
#        inv_shear!(sb1, sb1, shear_dir)
#     4. Recurse on sb0 and sb1
#
# Subband sizes after l levels (input n₁ × n₂):
#   l=0: trivial (1 subband = input)
#   l=1 (:col split): 2 subbands each  n₁ × n₂/2
#   l=2 (:row split): 4 subbands each  n₁/2 × n₂/2
#   l=3 (:col split): 8 subbands each  n₁/2 × n₂/4
#   l=4 (:row split): 16 subbands each n₁/4 × n₂/4
#
# Input dimensions must be divisible by 2^⌈l/2⌉ in each axis.

"""
    dfb_decompose(bandpass, l_levels, qfp::QuincunxFilterPair) -> Vector{Matrix}

Decompose `bandpass` into `2^l_levels` directional subbands using an
`l_levels`-deep binary tree DFB.  Subbands are ordered left-to-right in the
binary tree (LP branch first at each depth level).

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(3)
julia> x = randn(32, 32)
julia> sbs = dfb_decompose(x, 2, Q2345)
julia> length(sbs)
4
julia> size(sbs[1])
(16, 16)
```
"""
function dfb_decompose(
        bandpass::AbstractMatrix, l_levels::Int,
        qfp::QuincunxFilterPair
    )
    l_levels >= 0 || throw(ArgumentError("l_levels must be ≥ 0"))
    l_levels == 0 && return [copy(bandpass)]
    T = promote_type(eltype(bandpass), eltype(qfp))
    img = T === eltype(bandpass) ? bandpass : T.(bandpass)
    qfp2 = T === eltype(qfp) ? qfp : QuincunxFilterPair{T}(
            T.(qfp.h_q), T.(qfp.g_q),
            qfp.c_h, qfp.c_g
        )
    return _dfb_split(img, l_levels, 1, qfp2)
end

function _dfb_split(
        img::AbstractMatrix, remaining::Int, depth::Int,
        qfp::QuincunxFilterPair
    )
    shear_dir = isodd(depth) ? :h : :v
    qfb_dir = isodd(depth) ? :col : :row
    sh = shear(img, shear_dir)
    sb0, sb1 = qfb_decompose(sh, qfp; dir = qfb_dir)
    # inv_shear! with aliased src/dst would corrupt — allocate fresh
    sb0 = inv_shear(sb0, shear_dir)
    sb1 = inv_shear(sb1, shear_dir)
    remaining == 1 && return [sb0, sb1]
    return vcat(
        _dfb_split(sb0, remaining - 1, depth + 1, qfp),
        _dfb_split(sb1, remaining - 1, depth + 1, qfp)
    )
end

"""
    dfb_reconstruct(subbands, qfp::QuincunxFilterPair) -> bandpass

Reconstruct a bandpass image from its `2^l` directional subbands.

# Examples
```jldoctest
julia> using Contourlets, Random; Random.seed!(3)
julia> x = randn(32, 32)
julia> sbs = dfb_decompose(x, 2, Q2345)
julia> rec = dfb_reconstruct(sbs, Q2345)
julia> maximum(abs, rec .- x) < 1e-10
true
```
"""
function dfb_reconstruct(
        subbands::Vector{<:AbstractMatrix},
        qfp::QuincunxFilterPair
    )
    n = length(subbands)
    n >= 1   || throw(ArgumentError("subbands must be non-empty"))
    ispow2(n) || throw(ArgumentError("number of subbands must be a power of 2"))
    n == 1 && return copy(subbands[1])
    l_levels = round(Int, log2(n))
    T = promote_type(eltype(subbands[1]), eltype(qfp))
    qfp2 = T === eltype(qfp) ? qfp : QuincunxFilterPair{T}(
            T.(qfp.h_q), T.(qfp.g_q),
            qfp.c_h, qfp.c_g
        )
    return _dfb_merge(subbands, l_levels, 1, qfp2)
end

function _dfb_merge(
        sbs::Vector{<:AbstractMatrix}, l::Int, depth::Int,
        qfp::QuincunxFilterPair
    )
    shear_dir = isodd(depth) ? :h : :v
    qfb_dir = isodd(depth) ? :col : :row
    half = length(sbs) ÷ 2
    if l == 1
        sb0, sb1 = sbs[1], sbs[2]
    else
        sb0 = _dfb_merge(sbs[1:half], l - 1, depth + 1, qfp)
        sb1 = _dfb_merge(sbs[(half + 1):end], l - 1, depth + 1, qfp)
    end
    sh0 = shear(sb0, shear_dir)
    sh1 = shear(sb1, shear_dir)
    rec = qfb_reconstruct(sh0, sh1, qfp; dir = qfb_dir)
    return inv_shear(rec, shear_dir)
end
