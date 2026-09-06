# 3-D NSCT Example

The 3-D Nonsubsampled Contourlet Transform decomposes a volume into a coarse
residual plus `3 * 2^L` directional subbands per scale, all at the input size.
Like its 2-D counterpart it is exactly equivariant under circular shifts.

```@example nsct3
using Contourlets, Random

vol    = randn(Xoshiro(1), 32, 32, 32)
params = ContourletParams3(J = 2, L_array = [1, 2])
```

---

## Variant 1 — Allocating

```@example nsct3
c   = nsct3_forward(vol, params)
rec = nsct3_inverse(c, params)

println("PR error: ", maximum(abs, rec .- vol))
println("coarse: ", size(c.coarse))
for j in 1:params.J
    println("scale $j: $(length(c.subbands[j])) subbands of $(size(c.subbands[j][1]))")
end
@assert maximum(abs, rec .- vol) < 1e-12
```

Note how fast the coefficient count grows: `3 * 2^L` subbands per scale. That is
the price of shift invariance, and it is why the streaming form below exists.

---

## Variant 2 — Workspace (for iterative algorithms)

Build the plans once and reuse them. This is the entry point for reconstruction
loops, where the transform is applied hundreds of times to the same volume size.

```@example nsct3
ws  = make_nsct3_workspace(params, size(vol), Float64)
out = similar(vol)

for _ in 1:3
    coeffs = nsct3_forward(vol, params; workspace = ws)
    nsct3_inverse!(out, coeffs, params; workspace = ws)
end
println("PR error with workspace: ", maximum(abs, out .- vol))
@assert maximum(abs, out .- vol) < 1e-12
```

---

## Variant 3 — Streaming (the one to use at working sizes)

At 256³ `Float32` a single subband is 64 MB, so `J = 3` with `L = [2, 2, 3]`
would materialise 49 volumes — 3.1 GB — before you do anything with them. The
streaming form calls back once per subband and reuses one buffer, so it holds
two volumes plus the plans.

```@example nsct3
energy = Dict{Tuple{Int, Int}, Float64}()

coarse = nsct3_forward(vol, params) do j, k, band
    # `band` is valid only until this callback returns — copy or reduce here.
    energy[(j, k)] = sum(abs2, band)
end

println("coarse energy:      ", round(sum(abs2, coarse), digits = 2))
println("directional energy: ", round(sum(values(energy)), digits = 2))
```

The inverse streams too: `f(j, k)` returns each subband on demand, so a caller
that keeps coefficients on disk — or regenerates them — never holds the full
set.

```@example nsct3
r = nsct3_inverse(c.coarse, params) do j, k
    c.subbands[j][k]
end
println("streaming inverse PR error: ", maximum(abs, r .- vol))
@assert maximum(abs, r .- vol) < 1e-12
```

---

## Shift invariance

```@example nsct3
s = (3, -2, 5)
a = nsct3_forward(vol, params)
b = nsct3_forward(circshift(vol, s), params)

err = maximum(
    maximum(abs, b.subbands[j][k] .- circshift(a.subbands[j][k], s))
        for j in 1:params.J for k in eachindex(a.subbands[j])
)
println("shift equivariance error: ", err)
@assert err < 1e-10
```

---

## Directional selectivity

A plane wave lands in the subband whose wedge contains its frequency vector.
Here the frequency is steepest along axis 3, so the winning subband belongs to
the third hourglass.

```@example nsct3
n = 32
k = (2, 5, 13)
wave = [
    cospi(2 * (k[1] * (i - 1) + k[2] * (j - 1) + k[3] * (l - 1)) / n)
        for i in 1:n, j in 1:n, l in 1:n
]

fb = make_nsdfb3((n, n, n), Float64; levels = 4, transition = 0.5)
e  = zeros(nsdfb3_nsubbands(fb))
nsdfb3_decompose(wave, fb) do idx, band
    e[idx] = sum(abs2, band)
end

best = argmax(e)
hourglass, p, q = nsdfb3_subband_triple(fb, best)
println("winning subband $best: hourglass $hourglass, wedge ($p, $q)")
println("its share of the energy: ", round(e[best] / sum(e), digits = 3))
@assert hourglass == 3
```

---

## Complex volumes

Complex data is filtered as two real passes through the identical kernel, so the
transform is exactly real-linear: `T(x + i·y) == T(x) + i·T(y)`, bit for bit.
That makes it a usable correctness check on your own code built on top.

```@example nsct3
xr = randn(Xoshiro(2), 16, 16, 16)
xi = randn(Xoshiro(3), 16, 16, 16)
p  = ContourletParams3(J = 1, L_array = [1])

cc = nsct3_forward(complex.(xr, xi), p)
cr = nsct3_forward(xr, p)
ci = nsct3_forward(xi, p)

println(
    "linearity error: ",
    maximum(abs, cc.subbands[1][1] .- complex.(cr.subbands[1][1], ci.subbands[1][1]))
)
```


---

## Decimated: the surfacelet transform

The NSCT3 keeps every subband at full size. The surfacelet transform stores each
one at its own reduced resolution instead, giving up shift invariance for a
smaller coefficient set. Reconstruction is still exact.

```@example nsct3
sp = SurfaceletParams3(J = 2, L_array = [1, 1])
pl = make_surfacelet3(sp, size(vol), Float64)

sc = surfacelet3_forward(vol, sp; plan = pl)
sr = surfacelet3_inverse(sc, sp; plan = pl)

println("PR error: ", maximum(abs, sr .- vol))
println("redundancy: ", round(surfacelet3_redundancy(pl), digits = 2))
println("coarse subband: ", size(sc.coarse), " from a ", size(vol), " volume")
println("scale 1 subband: ", size(sc.subbands[1][1]))
println("scale 2 subband: ", size(sc.subbands[2][1]))
@assert maximum(abs, sr .- vol) < 1e-11
```

Note the sizes: coarser scales cover smaller frequency shells and are stored on
correspondingly smaller grids. See [Theory — 3-D](@ref theory3d) for how far
this decimation goes and, specifically, where it stops short of the reference
construction.
