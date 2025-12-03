%%%-------------------------------------------------------------------
%%% @doc sied - SIMD operations for Erlang
%%%
%%% High-performance vectorized operations using SIMD instructions
%%% via Rust NIF with simdeez. Provides runtime SIMD detection and
%%% automatic dispatch to SSE2, SSE4.1, AVX2, or NEON instructions.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(sied).

%% Basic Arithmetic Operations
-export([
    add_f32/2,
    add_f64/2,
    subtract_f32/2,
    subtract_f64/2,
    multiply_f32/2,
    multiply_f64/2,
    divide_f32/2,
    divide_f64/2
]).

%% Reduction Operations
-export([
    dot_product_f32/2,
    dot_product_f64/2,
    sum_f32/1,
    sum_f64/1
]).

%% Statistical Operations
-export([
    mean_f32/1,
    mean_f64/1,
    variance_f32/1,
    variance_f64/1,
    std_dev_f32/1,
    std_dev_f64/1
]).

%% Min/Max Operations
-export([
    min_f32/1,
    min_f64/1,
    max_f32/1,
    max_f64/1,
    min_elementwise_f32/2,
    min_elementwise_f64/2,
    max_elementwise_f32/2,
    max_elementwise_f64/2
]).

%% Unary Operations
-export([
    abs_f32/1,
    abs_f64/1,
    sqrt_f32/1,
    sqrt_f64/1,
    negate_f32/1,
    negate_f64/1
]).

-on_load(init/0).

-define(APPNAME, sied).
-define(LIBNAME, sied).

%%%===================================================================
%%% NIF Loading
%%%===================================================================

%% @private
%% @doc Initialize and load the NIF library
init() ->
    SoName = case code:priv_dir(?APPNAME) of
        {error, bad_name} ->
            case filelib:is_dir(filename:join(["..", priv])) of
                true ->
                    filename:join(["..", priv, ?LIBNAME]);
                false ->
                    filename:join([priv, ?LIBNAME])
            end;
        Dir ->
            filename:join(Dir, ?LIBNAME)
    end,
    erlang:load_nif(SoName, 0).

%%%===================================================================
%%% Basic Arithmetic Operations
%%%===================================================================

%% @doc Element-wise addition of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec add_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
add_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise addition of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec add_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
add_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise subtraction of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec subtract_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
subtract_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise subtraction of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec subtract_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
subtract_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise multiplication of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec multiply_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
multiply_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise multiplication of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec multiply_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
multiply_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise division of two f32 vectors
%% @param A First list of floats (numerators)
%% @param B Second list of floats (denominators, must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec divide_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
divide_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise division of two f64 vectors
%% @param A First list of doubles (numerators)
%% @param B Second list of doubles (denominators, must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec divide_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
divide_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Reduction Operations
%%%===================================================================

%% @doc Compute dot product of two f32 vectors
%% Computes the scalar product: sum(A[i] * B[i])
%% @param A First vector
%% @param B Second vector (must be same length)
%% @returns {ok, Scalar} | {error, Reason}
-spec dot_product_f32([float()], [float()]) -> {ok, float()} | {error, term()}.
dot_product_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute dot product of two f64 vectors
%% Computes the scalar product: sum(A[i] * B[i])
%% @param A First vector
%% @param B Second vector (must be same length)
%% @returns {ok, Scalar} | {error, Reason}
-spec dot_product_f64([float()], [float()]) -> {ok, float()} | {error, term()}.
dot_product_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute sum of all elements in an f32 vector
%% @param A List of floats to sum
%% @returns {ok, Sum} | {error, Reason}
-spec sum_f32([float()]) -> {ok, float()} | {error, term()}.
sum_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute sum of all elements in an f64 vector
%% @param A List of doubles to sum
%% @returns {ok, Sum} | {error, Reason}
-spec sum_f64([float()]) -> {ok, float()} | {error, term()}.
sum_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Statistical Operations
%%%===================================================================

%% @doc Compute arithmetic mean of an f32 vector
%% @param A List of floats
%% @returns {ok, Mean} | {error, Reason}
-spec mean_f32([float()]) -> {ok, float()} | {error, term()}.
mean_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute arithmetic mean of an f64 vector
%% @param A List of doubles
%% @returns {ok, Mean} | {error, Reason}
-spec mean_f64([float()]) -> {ok, float()} | {error, term()}.
mean_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute variance of an f32 vector
%% @param A List of floats
%% @returns {ok, Variance} | {error, Reason}
-spec variance_f32([float()]) -> {ok, float()} | {error, term()}.
variance_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute variance of an f64 vector
%% @param A List of doubles
%% @returns {ok, Variance} | {error, Reason}
-spec variance_f64([float()]) -> {ok, float()} | {error, term()}.
variance_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute standard deviation of an f32 vector
%% @param A List of floats
%% @returns {ok, StdDev} | {error, Reason}
-spec std_dev_f32([float()]) -> {ok, float()} | {error, term()}.
std_dev_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute standard deviation of an f64 vector
%% @param A List of doubles
%% @returns {ok, StdDev} | {error, Reason}
-spec std_dev_f64([float()]) -> {ok, float()} | {error, term()}.
std_dev_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Min/Max Operations
%%%===================================================================

%% @doc Find minimum value in an f32 vector
%% @param A List of floats
%% @returns {ok, Min} | {error, Reason}
-spec min_f32([float()]) -> {ok, float()} | {error, term()}.
min_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Find minimum value in an f64 vector
%% @param A List of doubles
%% @returns {ok, Min} | {error, Reason}
-spec min_f64([float()]) -> {ok, float()} | {error, term()}.
min_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Find maximum value in an f32 vector
%% @param A List of floats
%% @returns {ok, Max} | {error, Reason}
-spec max_f32([float()]) -> {ok, float()} | {error, term()}.
max_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Find maximum value in an f64 vector
%% @param A List of doubles
%% @returns {ok, Max} | {error, Reason}
-spec max_f64([float()]) -> {ok, float()} | {error, term()}.
max_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise minimum of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec min_elementwise_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
min_elementwise_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise minimum of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec min_elementwise_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
min_elementwise_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise maximum of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec max_elementwise_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
max_elementwise_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise maximum of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec max_elementwise_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
max_elementwise_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Unary Operations
%%%===================================================================

%% @doc Compute absolute value of an f32 vector
%% @param A List of floats
%% @returns {ok, Result} | {error, Reason}
-spec abs_f32([float()]) -> {ok, [float()]} | {error, term()}.
abs_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute absolute value of an f64 vector
%% @param A List of doubles
%% @returns {ok, Result} | {error, Reason}
-spec abs_f64([float()]) -> {ok, [float()]} | {error, term()}.
abs_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute square root of an f32 vector
%% @param A List of floats (must be non-negative)
%% @returns {ok, Result} | {error, Reason}
-spec sqrt_f32([float()]) -> {ok, [float()]} | {error, term()}.
sqrt_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute square root of an f64 vector
%% @param A List of doubles (must be non-negative)
%% @returns {ok, Result} | {error, Reason}
-spec sqrt_f64([float()]) -> {ok, [float()]} | {error, term()}.
sqrt_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Negate an f32 vector (multiply by -1)
%% @param A List of floats
%% @returns {ok, Result} | {error, Reason}
-spec negate_f32([float()]) -> {ok, [float()]} | {error, term()}.
negate_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Negate an f64 vector (multiply by -1)
%% @param A List of doubles
%% @returns {ok, Result} | {error, Reason}
-spec negate_f64([float()]) -> {ok, [float()]} | {error, term()}.
negate_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).
