%%%-------------------------------------------------------------------
%%% @doc sied_bin — binary-first, typed SIMD primitives (step 1, WIP).
%%%
%%% This module is the forward-looking, <b>binary-first</b> surface of sied. The
%%% existing {@link sied} module takes and returns Erlang float lists; at raster
%%% scale (a Sentinel-2 band is ~120 M pixels) lists are a non-starter — one band
%%% as a float list costs ~5 GB of BEAM heap versus 240 MB as a `uint16' binary.
%%%
%%% Everything here operates on <b>little-endian binaries</b> and adds two things
%%% the list API lacks:
%%%
%%%   1. <b>Typed decode</b> — `u16' source samples widened to `f32' inside the
%%%      kernel, halving memory traffic on the dominant (memory-bound) cost.
%%%   2. <b>Fusion</b> — a normalized difference `(a-b)/(a+b)' in one pass rather
%%%      than three separate list ops that each re-read the data.
%%%
%%% This is <b>additive and non-breaking</b>: nothing in the existing {@link sied}
%%% list API changes, so kvex keeps working unchanged. Each function delegates to
%%% a DirtyCpu Rustler NIF (registered under the `sied' module) — see `ROADMAP.md'.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(sied_bin).

%% Typed decode
-export([decode_u16_f32/2]).

%% Binary-first elementwise (f32)
-export([add_f32_bin/2, subtract_f32_bin/2, multiply_f32_bin/2, divide_f32_bin/2]).

%% Binary-first reductions (f32)
-export([sum_f32_bin/1, dot_product_f32_bin/2]).

%% Fused primitive (domain-agnostic; NDVI/NDWI are named instances of it)
-export([normalized_difference_f32_bin/2]).

-type f32_bin() :: binary().   %% little-endian IEEE-754 f32 samples
-type u16_bin() :: binary().   %% little-endian uint16 samples
-type reason()  :: length_mismatch | odd_length | empty | not_implemented | term().

-export_type([f32_bin/0, u16_bin/0]).

%%%===================================================================
%%% Typed decode
%%%===================================================================

%% @doc Widen a little-endian `u16' binary to a little-endian `f32' binary,
%% scaling each sample by `Scale' (`1.0' for a plain widening cast).
-spec decode_u16_f32(u16_bin(), float()) -> {ok, f32_bin()} | {error, reason()}.
decode_u16_f32(Data, Scale) when is_binary(Data), is_number(Scale) ->
    sied:decode_u16_f32(Data, float(Scale)).

%%%===================================================================
%%% Binary-first elementwise (f32)
%%%===================================================================

%% @doc Element-wise `A + B' over two equal-length `f32' binaries.
-spec add_f32_bin(f32_bin(), f32_bin()) -> {ok, f32_bin()} | {error, reason()}.
add_f32_bin(A, B) when is_binary(A), is_binary(B) ->
    sied:add_f32_bin(A, B).

%% @doc Element-wise `A - B' over two equal-length `f32' binaries.
-spec subtract_f32_bin(f32_bin(), f32_bin()) -> {ok, f32_bin()} | {error, reason()}.
subtract_f32_bin(A, B) when is_binary(A), is_binary(B) ->
    sied:subtract_f32_bin(A, B).

%% @doc Element-wise `A * B' over two equal-length `f32' binaries.
-spec multiply_f32_bin(f32_bin(), f32_bin()) -> {ok, f32_bin()} | {error, reason()}.
multiply_f32_bin(A, B) when is_binary(A), is_binary(B) ->
    sied:multiply_f32_bin(A, B).

%% @doc Element-wise `A / B' over two equal-length `f32' binaries.
-spec divide_f32_bin(f32_bin(), f32_bin()) -> {ok, f32_bin()} | {error, reason()}.
divide_f32_bin(A, B) when is_binary(A), is_binary(B) ->
    sied:divide_f32_bin(A, B).

%%%===================================================================
%%% Binary-first reductions (f32)
%%%===================================================================

%% @doc Sum of all `f32' samples in a binary.
-spec sum_f32_bin(f32_bin()) -> {ok, float()} | {error, reason()}.
sum_f32_bin(A) when is_binary(A) ->
    sied:sum_f32_bin(A).

%% @doc Dot product of two equal-length `f32' binaries.
%% (Single-vector counterpart to {@link sied:dot_product_batch_f32_bin/2}.)
-spec dot_product_f32_bin(f32_bin(), f32_bin()) -> {ok, float()} | {error, reason()}.
dot_product_f32_bin(A, B) when is_binary(A), is_binary(B) ->
    sied:dot_product_f32_bin(A, B).

%%%===================================================================
%%% Fused primitive
%%%===================================================================

%% @doc Normalized difference `(A - B) / (A + B)', computed in a single fused
%% pass over two equal-length `f32' binaries; `0.0' where `A + B == 0'. This is
%% the domain-agnostic shape; rast names its instances (NDVI = ND(NIR, Red),
%% NDWI = ND(Green, NIR), ...).
-spec normalized_difference_f32_bin(f32_bin(), f32_bin()) ->
        {ok, f32_bin()} | {error, reason()}.
normalized_difference_f32_bin(A, B) when is_binary(A), is_binary(B) ->
    sied:normalized_difference_f32_bin(A, B).
