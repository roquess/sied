# Single Instruction Erlang Data (sied)

High-performance SIMD operations for Erlang through Rust NIFs.

[![Hex.pm](https://img.shields.io/hexpm/v/sied.svg)](https://hex.pm/packages/sied)
[![Hex Docs](https://img.shields.io/badge/hex-docs-blue.svg)](https://hexdocs.pm/sied)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Motivation

Erlang excels at building concurrent, fault-tolerant systems, but numerical computations on large datasets can be a bottleneck. Modern CPUs offer SIMD (Single Instruction, Multiple Data) instructions that can process multiple data elements simultaneously, providing significant performance improvements for vectorized operations.

**sied** bridges this gap by exposing SIMD-accelerated mathematical operations to Erlang through a safe Rust NIF. The name combines **SI**MD with **E**rlang and **D**ata, representing the library's core purpose: bringing efficient data-parallel processing to the Erlang ecosystem.

This library enables Erlang applications to perform high-throughput numerical computations without sacrificing the language's strengths in concurrency and reliability.

## Features

- **Zero unsafe code**: All operations implemented in 100% safe Rust
- **Automatic SIMD optimization**: Compiler leverages AVX2, AVX-512, NEON, and other instruction sets
- **Simple API**: Consistent interface with proper error handling
- **Production-ready**: Comprehensive test suite with benchmarks
- **Cross-platform**: Works across different CPU architectures with graceful fallback

## Implementation

This library uses [Rustler](https://crates.io/crates/rustler) to create Native Implemented Functions (NIFs) that bridge Erlang and Rust. Rustler provides a safe, idiomatic way to write Erlang NIFs in Rust with automatic memory management and type conversion.

The mathematical operations are implemented using safe iterator-based patterns that the LLVM compiler automatically vectorizes into SIMD instructions when beneficial. This approach provides excellent performance while maintaining memory safety guarantees.

## Installation

### From Hex.pm

Add to your `rebar.config`:

### From GitHub

```erlang
{deps, [
    {sied, {git, "https://github.com/roquess/sied.git", {branch, "main"}}}
]}.
```

## Building

```bash
rebar3 compile
```

**Requirements:**
- Erlang/OTP 24 or later
- Rust 1.70 or later
- Cargo (included with Rust)

The Rust toolchain must be available in your PATH. Visit [rustup.rs](https://rustup.rs) to install Rust.

## API Reference

All functions return `{ok, Result}` on success or `{error, Reason}` on failure.

### Vector Addition

Element-wise addition of two vectors.

```erlang
%% Single-precision (f32)
{ok, Result} = sied:add_f32([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).
%% Result = [5.0, 7.0, 9.0]

%% Double-precision (f64)
{ok, Result} = sied:add_f64([1.0, 2.0], [3.0, 4.0]).
%% Result = [4.0, 6.0]
```

### Vector Multiplication

Element-wise multiplication of two vectors.

```erlang
{ok, Result} = sied:multiply_f32([2.0, 3.0, 4.0], [5.0, 6.0, 7.0]).
%% Result = [10.0, 18.0, 28.0]

{ok, Result} = sied:multiply_f64([2.0, 3.0], [4.0, 5.0]).
%% Result = [8.0, 15.0]
```

### Dot Product

Scalar product of two vectors: sum(a[i] * b[i])

```erlang
{ok, Dot} = sied:dot_product_f32([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).
%% Dot = 32.0  (1*4 + 2*5 + 3*6)

{ok, Dot} = sied:dot_product_f64([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).
%% Dot = 32.0
```

### Sum

Sum of all elements in a vector.

```erlang
{ok, Sum} = sied:sum_f32([1.0, 2.0, 3.0, 4.0, 5.0]).
%% Sum = 15.0

{ok, Sum} = sied:sum_f64([1.0, 2.0, 3.0, 4.0, 5.0]).
%% Sum = 15.0
```

## Error Handling

Binary operations require vectors of equal length:

```erlang
case sied:add_f32([1.0, 2.0], [3.0]) of
    {ok, Result} ->
        io:format("Success: ~p~n", [Result]);
    {error, length_mismatch} ->
        io:format("Error: vectors must have equal length~n")
end.
```

## Testing

Run the complete test suite:

```bash
rebar3 eunit
```

The test suite includes:
- Functional correctness tests
- Edge cases (empty vectors, single elements, negative values)
- Large vector operations (10,000+ elements)
- Performance benchmarks comparing NIF vs pure Erlang implementations

### Running Benchmarks

```bash
rebar3 eunit --module=sied
```

Benchmarks test operations on 100,000-element vectors over multiple iterations to measure:
- Per-operation latency
- Throughput for large datasets
- Speedup compared to native Erlang implementations

## Performance Characteristics

Performance varies based on vector size due to NIF call overhead and SIMD efficiency:

- **Small vectors** (< 100 elements): NIF overhead may dominate, modest improvements
- **Medium vectors** (100-10,000 elements): Good speedup over pure Erlang
- **Large vectors** (> 10,000 elements): Maximum SIMD benefit, significant speedup

The implementation is optimized for real-world workloads across all vector sizes, with the compiler making intelligent decisions about SIMD usage based on data alignment, vector length, and available CPU features.

## Project Structure

```
sied/
├── src/
│   ├── sied.app.src          # OTP application metadata
│   └── sied.erl              # Erlang API module
├── native/
│   └── sied/
│       ├── Cargo.toml        # Rust dependencies and configuration
│       └── src/
│           └── lib.rs        # Rust NIF implementation
├── test/
│   └── sied_tests.erl        # EUnit test suite
├── rebar.config              # rebar3 build configuration
└── README.md
```

## Safety and Correctness

This library prioritizes safety and correctness:

1. **No unsafe code**: Every Rust function uses safe abstractions verified by the compiler
2. **Automatic bounds checking**: All vector accesses are bounds-checked, preventing buffer overflows
3. **Memory safety**: Rust's ownership system eliminates use-after-free and data races
4. **Graceful degradation**: Falls back to scalar operations when SIMD is unavailable
5. **Comprehensive testing**: Both Rust unit tests and Erlang integration tests

While hand-written SIMD with unsafe code can sometimes achieve higher performance, this library chooses safety and maintainability. The performance from compiler auto-vectorization is excellent for the vast majority of applications, and the safety guarantees are invaluable in production environments.

## Dependencies

This library uses the following Rust crates:

- [rustler](https://crates.io/crates/rustler) - Safe Rust bindings for writing Erlang NIFs

Special thanks to the Rustler team for making safe, high-performance Erlang NIFs practical.

## Contributing

Contributions are welcome! When contributing, please ensure:

- All code remains 100% safe Rust (no unsafe blocks)
- Tests pass: `rebar3 eunit`
- Code is well-documented with clear comments
- New features include appropriate tests
- Benchmarks show no performance regressions

## Links

- GitHub: [https://github.com/roquess/sied](https://github.com/roquess/sied)
- Hex.pm: [https://hex.pm/packages/sied](https://hex.pm/packages/sied)
- Rustler on crates.io: [https://crates.io/crates/rustler](https://crates.io/crates/rustler)
- Simdeez on crates.io: [https://crates.io/crates/simdeez](https://crates.io/crates/simdeez)

