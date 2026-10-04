# natural

A C++20 arbitrary-precision unsigned integer on a hand-written, scalar x86-64
number-theoretic-transform engine. One thread, no SIMD, no external dependencies.

For balanced products of **2^16 … 2^26 limbs** (64 KiB … 4 MiB operands, 4 … 268 Mbit) it
matches or beats **GNU MP 6.3.0** on most sizes, by up to **1.47x**:

| 41 sizes, 2^16 … 2^26 limbs, interleaved A/B, mains power | |
|---|---|
| clear wins (>1.05x) / ties (0.95–1.05x) / losses (<0.95x) | **29 / 7 / 5** |
| largest lead | **1.47x** at 1,048,576 limbs |
| powers of two 2^17 … 2^26 | all ten are clear wins, **1.25x … 1.47x** |
| squaring, against an equal-length product | **1.42x** geometric mean (24 … 8192 limbs) |
| peak memory for one product | 1.9x GNU MP's |

The full tables, the machine, the measurement protocol and where the five losses come from
are in [`docs/results.md`](docs/results.md) and [`docs/design.md`](docs/design.md).

## Requirements

* **x86-64** (System V ABI), Linux. The assembly needs **BMI2** (`mulx`) and is Intel-syntax
  GNU as; `-march=x86-64-v3` is the lowest level that provides it.
* A C++20 standard library with `<format>` (GCC 13+, Clang 17+ with libstdc++ 13+).
* Consumers must compile with `-masm=intel`: the public headers contain Intel-syntax inline
  assembly (`include/natural/basic_arithmetics.h`). **The CMake target adds this for you**;
  if you build by hand, do not forget it (without it you get ~160 assembler errors).

## Using it

CMake, as a subdirectory or via `FetchContent`:

```cmake
add_subdirectory(natural)              # or FetchContent_MakeAvailable(natural)
target_link_libraries(app PRIVATE natural::natural)
```

Installed and consumed from another project:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j && cmake --install build
```

```cmake
find_package(natural 4 REQUIRED)
target_link_libraries(app PRIVATE natural::natural)
```

Then:

```cpp
#include <natural/natural.h>

int main() {
	nat::natural a("123456789012345678901234567890");
	nat::natural b = nat::sqr(a);          // or sqr(a): ADL finds it
	return b == a * a ? 0 : 1;
}
```

Without CMake, the `Makefile` does the same thing with plain `g++`:

```sh
make test        # build + run the self-checking suite
make bench       # throughput of the multiply/square paths
make example     # build + run examples/square.cpp
```

Or manually, if you prefer:

```sh
g++ -O3 -march=native -std=c++20 -masm=intel -Iinclude \
    src/mul_basecase.s src/mul_ntt.s your_program.cpp -o your_program
```

## How it works

* **Engine.** Three 56-bit NTT-friendly primes (`m = 27 / 58 / 87 · 2^56 + 1`), a `2^scale`
  transform, DIF forward / DIT inverse so **no bit reversal** is needed anywhere, and
  butterfly outputs kept in the lazy range `[0, 2p)` so add/sub need no correction step.
  Twiddle multiplication is Shoup's trick with a precomputed reciprocal; the three residues
  are recombined by `nat_asmCRT`.
* **DRAM / L3 / L2 / L1 schedule.** The scheduled layers are run level by level, chunk-major,
  one level per memory level: the whole array (DRAM), then `2^20`-element chunks (8 MiB, one
  modulus at a time, under the 24 MiB L3), `2^16` (512 KiB, L2) and `2^12` (32 KiB, L1). Each
  level is a plain range of layers,
  **one pass per layer**, with the twiddle cursor derived positionally from the block base
  (`base >> j`); the cut points are clamped to the array size and de-duplicated per transform,
  so a small scale simply has fewer levels. Layers run descending (largest distance first)
  forward and ascending inverse, so the forward and inverse level runners stay mirror images
  of each other. The three edge passes are *fused* into passes that were already touching the
  data (fold in `load()`, last forward layer + pointwise + first inverse layer in
  `nat_asmNttMul`, last inverse layer + the final shift in `intt_shr`).
* **One size shorter.** Just above a power of two the transform would be half empty; instead
  a cyclic convolution of length `N` plus the exact correction `a·b = C + (B^N − 1)·H`
  replaces it, and a cost model picks the cheaper of the direct, split and wrap paths.
* **Base cases.** Below 24 limbs a long × short schoolbook (`nat_mul_basecase_even/odd`),
  then Toom-22, then the NTT above 896 limbs.
* **Squaring** has its own path: a symmetric leaf that does ~half the multiplies
  (`nat_sqr_basecase`) and a Karatsuba square that needs one operand and `3m` instead of
  `4m` workspace limbs per level (`sqr_toom22`, cutoff 37 limbs).

## Testing and benchmarking

```sh
ctest --test-dir build --output-on-failure
./build/natural_bench 67108864          # limbs, powers of two from 2^14
```

The test suite is self-contained: it checks every arithmetic path against a plain O(n·m)
128-bit schoolbook written inside the test, and checks every pair of internal paths against
each other (base case vs Toom-22 vs NTT, planner on/off, wrap correction on/off, `sqr` vs
`a*a`, the NTT schedule plan and its explicit-scale products vs the base case, division
inverting multiplication). It runs in a few seconds.

## Repository layout

```
include/natural/    public headers; natural.h is the entry point
src/                the assembly kernels (mul_basecase.s, mul_ntt.s)
tests/              self-checking test suite (CTest)
benchmarks/         throughput benchmark
examples/           minimal usage example
docs/               design and measurement notes
cmake/              package config for find_package(natural)
```

## Limitations

* Single-threaded, no SIMD, x86-64 only. Peak memory is about 1.9x GNU MP's per product.
* The five sizes where GNU MP wins (81,920 … 196,608 limbs) are transform-fill valleys: the
  wrap path cannot reach them and a half-empty transform costs more than it saves. The loss
  is bounded at 0.83x (see `docs/results.md`).
* Division uses a Newton iteration with a conservative accuracy heuristic. The shapes where
  that heuristic degenerates -- a short divisor, or a short quotient -- are now dispatched to
  an exact multi-limb schoolbook division instead (`div_schoolbook_divisor_max` /
  `div_schoolbook_quotient_max`), and the iterative path keeps a bounded fallback as a safety
  net, so the one-call hang this used to have (`(2^192-1) / (2^64+1)`) is gone. Replacing the
  heuristic itself remains the standing recommendation; the history and the measurements are
  at the top of `include/natural/natural.h`. Multiplication, squaring and the NTT are
  unaffected.
* The headers use inline assembly, which is why `-masm=intel` is a public requirement;
  moving those few routines into the assembly files would remove it.

## License

MIT — see [LICENSE](LICENSE).
