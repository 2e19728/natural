# Design notes

This is the short version of how `nat` works and why it looks the way it does. Numbers
quoted here are measured; see [`results.md`](results.md) for the protocol and the full tables.

## 1. The modulus choice

The library multiplies 64-bit limbs. The engine is a three-modulus NTT with CRT
recombination, so the transform modulus `p` and the count of moduli have to satisfy

1. `p` fits a 64-bit limb and `p = α·2^k + 1` so that both `2^k`-th roots of unity and the
   final `2^-(scale-1)` shift are cheap (the shift becomes one multiply by `p >> k` plus a
   mask, no full modular inverse);
2. the twiddle multiply is a 64×64 → 128-bit product that stays inside one limb's worth of
   lazy range, which caps `p` at 56 bits with the lazy `[0, 2p)` representation;
3. `n·(2^64 − 1)^2 < ∏ p` for the largest coefficient that must be recovered exactly, which
   at 56 bits forces **three** moduli (two do not fit even for a 2^16-limb product).

At 56 bits there are exactly three primes of the form `α·2^56 + 1` below `2^64` that are
usable for the required transform lengths: `27·2^56 + 1`, `58·2^56 + 1`, `87·2^56 + 1`. So
the parameters are forced, not tuned.

## 2. Data layout and the transform

Each operand is held as three arrays of `2^scale` limbs, one per modulus. The forward
transform is DIF (decimation in frequency) and the inverse is DIT (decimation in time), which
means **no bit reversal is ever required**: the pair of transforms cancels the permutation.
The root table is stored in bit-reversed order and extended by prefix reuse, so a table of
size `2^k` costs one extra level of work over a table of size `2^(k-1)`.

Butterfly outputs are deliberately **not reduced**: they stay in `[0, 2p)`. That removes the
conditional subtraction from the inner loop (fixed instruction count, no data-dependent
branch) and it is the reason the twiddle multiply uses Shoup's trick — the precomputed
reciprocal handles `v < 2^64`, whereas a general Barrett step would need a reduction of the
operand first. Measured, Shoup is 1.55–1.80x faster than Barrett here and 1.86–2.32x faster
than a hardware division, so the choice of representation pays for itself twice.

## 3. Schedule

The transform is a fixed sequence of DIF layers at distances `2^(k-1), …, 2^1`. Two facts
drive the schedule:

* every layer with `2^(j+1) <= 2^T` is block diagonal with respect to a split of the array
  into `2^T`-element chunks, so those layers may be run chunk by chunk in any interleaving
  as long as all larger-distance layers have already run;
* the twiddle cursor is purely positional: a pass over `[base, base + 2^T)` must start at
  table entry `base >> j`. There is no per-chunk renormalisation to compute.

So the scheduled layers are run one level per memory level, coarse to fine: DRAM (the whole
array), then the L3 / L2 / L1 working sets — `2^20`-element chunks (8 MiB), `2^16` (512 KiB)
and `2^12` (32 KiB), one modulus at a time under the 24 MiB L3 / 1.25 MiB L2 / 48 KiB L1d.
Each level is a plain range of layers,
one pass per layer, and every pass derives its twiddle cursor from the block base, and inside
a level the layers are run chunk-major — every layer of the level over one chunk before moving on — so the chunk stays
resident across the whole level and the level streams the array once. The cut points are the
`ntt_scale_l1/l2/l3_threshold` constants, clamped to the array size and de-duplicated, so a
scale too small to fill a level simply has fewer levels (below `l1`, all four collapse into a
single DRAM pass).

Within a level the layers run descending (largest distance first) forward and ascending
inverse, which makes the forward and the inverse level runners mirror images of each other
without any special case for the finest level.

The three "edge" passes of the transform are not paid for separately; they are fused into
passes that already touch the data:

| edge pass | fused into |
|---|---|
| distance `N/2` fold (the first forward layer) | `load()` |
| last forward layer + pointwise product + first inverse layer | `nat_asmNttMul` |
| last inverse layer + the `2^-(scale-1)` shift | `intt_shr` |

The classic four-step / six-step FFT is deliberately *not* used. Its algebra needs the
chunk-local cascade to be a standalone DFT plus a rank-1 twiddle matrix, but a DIF
sub-problem keeps the global generator, so the correction factor depends on the layer index
and is not one elementwise multiply. Folding the rotation into the table (which this schedule
does) costs only the sequential table arcs — about 2 % of the data traffic at scale 23 —
whereas a four-step would spend a whole extra pass over the array on the twiddle multiply.

## 4. One size shorter (the wrap path)

`get_NTT_scale()` must round up to a power of two, so an operand a few limbs over a boundary
gets a transform that is half empty, and the cost jumps by ~1.7x for a 0.2 % larger input. A
cyclic convolution of length `N` yields `c_i = p_i + p_{i+N}`, and for `len1 + len2 − 1 <= 2N`
only the top `w` coefficients alias. With `C = Σ c_i B^i` and `H` the aliased part,

```
a·b = C + (B^N − 1)·H                                            (exact)
```

so one correction product of `(w+1)`-limb operands replaces a transform twice as large. The
correction product is itself an NTT when `w` is large, so a cost model decides per size
between the direct transform, the classic `a0/a1` split and the wrap path. Measured, this
turns the worst sawtooth case (1,068,871 limbs) from 388.6 ms into 238.9 ms, a 1.63x
improvement. Notably the wrap identity needs **no per-coefficient correction at all** — the
correction is a single carry-propagating add/sub of a small integer — which is what makes it
cheap enough to be worth planning for.

## 5. Base cases and squaring

Below 24 limbs the product is a long × short schoolbook (`src/mul_basecase.s`), split into
two kernels by the parity of the *short* operand so that every inner loop has an even trip
count and can be unrolled by two `mulx`. Above that, a balanced Toom-22 (Karatsuba)
recursion; an unbalanced operand is chopped into short-operand-sized blocks. Above 896 limbs
the NTT wins.

Squaring gets its own path because both operands are the same array:

* the leaf (`nat_sqr_basecase`) uses `s_j = Σ_{i+k=j} a_i a_k = 2·Σ_{i<k} a_i a_k +
  [j even]·a²_{j/2}`, i.e. about half the multiplies of an equal-length product. One round
  handles two adjacent columns with `j` odd, so the middle square always falls in the second
  column; both columns' pair sums go into **one** four-limb window at weights `2^0` and
  `2^64`, and a single doubling of that window at the end of the round doubles both columns
  (doubling is linear and preserves the relative weights). The only thing that must stay out
  of the window is the carry coming in from the previous round, which must not be doubled.
* the recursion (`sqr_toom22`) uses `a² = P0 + ((a0+a1)² − P0 − P2)·B^{s0} + P2·B^{2s0}`:
  three squarings, one sum instead of two, no unbalanced block loop, `3m` instead of `4m`
  workspace limbs per level.
* the leaf cutoff is 37 limbs: measured, the assembly leaf wins up to 36 and one recursion
  level wins from 37 (`mul_toom22_threshold` for products is 24, so the square leaf being
  1.6–1.9x cheaper per limb moves the crossover up as expected).

## 6. Workspaces

A transform at scale 23 needs three 64 MiB arrays per operand. Allocating and freeing those
once per multiply hands hundreds of megabytes back to the kernel each time and faults them
in again (~150 000 minor faults per multiply, >95 % of which disappear once the blocks are
recycled). They therefore come from a small per-thread LIFO cache of power-of-two blocks
(`nat_block_pool`, 8 slots / 768 MiB in total), which is why the library's peak RSS is ~1.9x
GNU MP's but its time does not include page-fault churn.

## 7. Where the remaining gap is

Against GNU MP's SSA the interesting comparison is not asymptotic — both are `O(n log n)`
with a similar layer count — but *execution resources per elementary operation*. GNU MP is
limited by limb movement (measured: 93 % of a cache-resident `mpn_add_n` pass, i.e. its SSA
is close to its own memory ceiling), while this engine is limited by execution ports in
cache and by memory at 64 MiB per modulus. The consequence is visible in the results: the
sizes where GNU MP wins are exactly the ones where this engine is forced into a half-empty
transform and cannot use the wrap path, and the win comes from moving 2.9x fewer load/store
operations, not from cheaper arithmetic.
