# Changelog

All notable changes to this project are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The NTT layer schedule is now one rule-based plan instead of three hard-coded nesting levels
  plus two fallback schedules.  `ntt_sched_for(k)` returns up to four *levels* (`ntt_sched`,
  `ntt_level`) — DRAM (the whole array) and the L3 / L2 / L1 working sets — cut by the
  `inline constexpr` `ntt_scale_l1_threshold` / `_l2_` / `_l3_` constants, clamped to the array
  size and de-duplicated per transform, so a small scale simply has fewer levels.  The
  `ntt_sched_v3` A/B switch and the `ntt_sched_min_scale` boundary are gone, as are the plain
  single-layer loop and the two-level schedule they selected: every scale now runs the same
  code.  The runtime tunables `ntt_sched_la` / `ntt_sched_lb` and
  `ntt_workspace::sched_levels()` are gone too; the cut points can still be scanned from the
  command line (`-Dntt_scale_l2_threshold=14`).  Each level is a plain range of layers and runs
  **one pass per layer** (descending forward, ascending inverse), so the two directions stay
  mirror images of each other without any pairing rule.
- The engine is radix-2 only.  The radix-4 merged-pass kernels (`nat_asmNtt_radix4`,
  `nat_asmNtt2_radix4`, `nat_asmINtt_radix4`, `nat_asmINtt2_radix4`, 415 lines) are removed
  from `src/mul_ntt.s`, and the parity-aligned pairing rule they required is gone with them.
  The radix-2 schedule is the engine every published throughput number was measured on.

### Fixed
- Division could spin forever on a two-limb divisor whose top limb is 1 (`(2^192-1)/(2^64+1)`
  is the one-call reproducer).  Such a divisor is left unnormalized, so `reciprocal()` had a
  single Newton iteration to run from a zero estimate; the quotient digit stayed 0 and
  `div_iterative` never reduced the remainder.  Shapes of that kind -- a short divisor, or a
  short quotient -- now go to an exact multi-limb schoolbook (Knuth D) division
  (`div_schoolbook_divisor_max` / `div_schoolbook_quotient_max`, 64 limbs each, either one
  qualifying), which is also faster than the iterative path below those bounds because it
  never builds a reciprocal.  The iterative path keeps a bounded fallback for any other shape
  whose estimate degenerates, and the `division regression` test group pins the reproducer,
  the measured family, both dispatch boundaries and the schoolbook add-back branch.

## [4.0.0] - 2026-09-16

First packaged release of the `nat` library.

### Added
- Dedicated squaring path: `nat_sqr_basecase` (symmetric leaf, ~half the multiplies of an
  equal-length product) and `sqr_toom22` (one operand, no unbalanced loop, `3m` instead of
  `4m` workspace limbs per level), selected by `sqr_toom22_threshold = 37`.  Measured 1.42x
  faster (geometric mean, 24 .. 8192 limbs) than squaring via the multiply path.
- CMake package: `find_package(natural)` provides the `natural::natural` target; the headers
  are installed under `include/natural/`.
- Self-checking test suite (`tests/`, 3600+ assertions against an independent 128-bit
  schoolbook reference) wired to CTest, plus a throughput benchmark.

### Changed
- Everything now lives in `namespace nat`; the assembly entry points are exported with a
  `nat_` prefix (`nat_asmNtt*`, `nat_asmINtt*`, `nat_asmNttMul`, `nat_asmCRT`,
  `nat_mul_basecase_even/odd`, `nat_sqr_basecase`).
- The multiply base-case kernels (`mul_basecase_even/odd`) and the square leaf
  (`sqr_basecase`) were merged into one translation unit, `src/mul_basecase.s`.
- Default include is `<natural/natural.h>`.

### Known limitations
- x86-64 only (System V ABI), requires BMI2 (`mulx`) and a C++20 standard library with
  `<format>`.  Consumers must compile with `-masm=intel` (the CMake target adds it).
- Single-threaded; no SIMD.  Peak memory is about 1.9x GNU MP's for one product.
- `reciprocal()` uses a conservative Newton accuracy heuristic - see the notes at the top of
  `include/natural/natural.h`.  Multiplication, squaring and the NTT are unaffected.
