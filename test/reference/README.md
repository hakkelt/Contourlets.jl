# Reference data

`toolbox_dfb_l12.bin` holds the output of `dfbdec_l(x, ldfilter('pkva'), l)` from
the MATLAB Contourlet Toolbox for `l = 1, 2`, run as a black box (its source was
not consulted). The input is the 32 × 32 image

```matlab
[i, j] = ndgrid(1:32, 1:32);
x = mod(73 * i.^2 + 179 * j.^2 + 31 * i .* j, 1031) / 1031 - 0.5;
```

and the file is the column-major `Float64` (little-endian) subbands concatenated
in order: the two `l = 1` subbands, then the four `l = 2` subbands.

Only `l ≤ 2` is kept. From `l = 3` on, the toolbox's tree is not the multichannel
filter bank of Do's thesis (eq. 3.23) on a periodic image: its subbands are not
equivariant to input shifts on their own sampling lattice (not even to an `(8, 8)`
shift, which lies on every channel's lattice). The package implements the thesis
definition, so the two agree exactly at `l ≤ 2` and differ from `l = 3`.
