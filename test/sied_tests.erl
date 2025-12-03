-module(sied_tests).

%%%===================================================================
%%% EUnit Tests
%%%===================================================================
-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

%%--------------------------------------------------------------------
%% Basic functionality tests
%%--------------------------------------------------------------------

add_f32_basic_test() ->
    {ok, Result} = sied:add_f32([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]),
    ?assertEqual([6.0, 8.0, 10.0, 12.0], Result).

add_f64_basic_test() ->
    {ok, Result} = sied:add_f64([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ?assertEqual([5.0, 7.0, 9.0], Result).

multiply_f32_basic_test() ->
    {ok, Result} = sied:multiply_f32([2.0, 3.0, 4.0], [5.0, 6.0, 7.0]),
    ?assertEqual([10.0, 18.0, 28.0], Result).

multiply_f64_basic_test() ->
    {ok, Result} = sied:multiply_f64([2.0, 3.0], [4.0, 5.0]),
    ?assertEqual([8.0, 15.0], Result).

dot_product_f32_basic_test() ->
    {ok, Result} = sied:dot_product_f32([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    Expected = 1.0*4.0 + 2.0*5.0 + 3.0*6.0, % 32.0
    ?assertEqual(Expected, Result).

dot_product_f64_basic_test() ->
    {ok, Result} = sied:dot_product_f64([2.0, 3.0, 4.0], [5.0, 6.0, 7.0]),
    Expected = 2.0*5.0 + 3.0*6.0 + 4.0*7.0, % 56.0
    ?assertEqual(Expected, Result).

sum_f32_basic_test() ->
    {ok, Result} = sied:sum_f32([1.0, 2.0, 3.0, 4.0, 5.0]),
    ?assertEqual(15.0, Result).

sum_f64_basic_test() ->
    {ok, Result} = sied:sum_f64([10.0, 20.0, 30.0]),
    ?assertEqual(60.0, Result).

%%--------------------------------------------------------------------
%% Edge cases
%%--------------------------------------------------------------------

add_f32_empty_test() ->
    {ok, Result} = sied:add_f32([], []),
    ?assertEqual([], Result).

add_f32_single_test() ->
    {ok, Result} = sied:add_f32([42.0], [8.0]),
    ?assertEqual([50.0], Result).

add_f32_length_mismatch_test() ->
    {error, length_mismatch} = sied:add_f32([1.0, 2.0], [3.0]).

sum_f32_empty_test() ->
    {ok, Result} = sied:sum_f32([]),
    ?assertEqual(0.0, Result).

sum_f32_single_test() ->
    {ok, Result} = sied:sum_f32([42.0]),
    ?assertEqual(42.0, Result).

%%--------------------------------------------------------------------
%% Large vector tests (SIMD performance matters here)
%%--------------------------------------------------------------------

add_f32_large_test() ->
    Size = 10000,
    A = [float(X) || X <- lists:seq(1, Size)],
    B = [float(X) || X <- lists:seq(1, Size)],
    {ok, Result} = sied:add_f32(A, B),
    ?assertEqual(Size, length(Result)),
    % Verify first and last elements
    ?assertEqual(2.0, lists:nth(1, Result)),
    ?assertEqual(float(Size * 2), lists:nth(Size, Result)).

dot_product_f32_large_test() ->
    Size = 1000,
    A = [1.0 || _ <- lists:seq(1, Size)],
    B = [2.0 || _ <- lists:seq(1, Size)],
    {ok, Result} = sied:dot_product_f32(A, B),
    Expected = float(Size * 2),
    ?assertEqual(Expected, Result).

sum_f32_large_test() ->
    Size = 5000,
    A = [1.0 || _ <- lists:seq(1, Size)],
    {ok, Result} = sied:sum_f32(A),
    ?assertEqual(float(Size), Result).

%%--------------------------------------------------------------------
%% Negative numbers and zeros
%%--------------------------------------------------------------------

add_f32_negative_test() ->
    {ok, Result} = sied:add_f32([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]),
    ?assertEqual([0.0, 0.0, 0.0], Result).

multiply_f32_zero_test() ->
    {ok, Result} = sied:multiply_f32([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
    ?assertEqual([0.0, 0.0, 0.0], Result).

multiply_f32_negative_test() ->
    {ok, Result} = sied:multiply_f32([-2.0, 3.0], [4.0, -5.0]),
    ?assertEqual([-8.0, -15.0], Result).

%%--------------------------------------------------------------------
%% Benchmark tests
%%--------------------------------------------------------------------

-define(BENCH_SIZE, 100000).
-define(BENCH_ITERATIONS, 100).

benchmark_add_f32_test_() ->
    {timeout, 60, fun() ->
        A = [float(X) || X <- lists:seq(1, ?BENCH_SIZE)],
        B = [float(X) || X <- lists:seq(1, ?BENCH_SIZE)],
        
        % Warm up
        {ok, _} = sied:add_f32(A, B),
        
        % Benchmark
        {Time, _} = timer:tc(fun() ->
            lists:foreach(fun(_) ->
                {ok, _} = sied:add_f32(A, B)
            end, lists:seq(1, ?BENCH_ITERATIONS))
        end),
        
        AvgTime = Time / ?BENCH_ITERATIONS,
        io:format("~nAdd F32 (~p elements): ~.2f µs per operation~n", 
                  [?BENCH_SIZE, AvgTime]),
        
        % Performance assertion: should be faster than 10ms per operation
        ?assert(AvgTime < 10000)
    end}.

benchmark_dot_product_f32_test_() ->
    {timeout, 60, fun() ->
        A = [float(X) || X <- lists:seq(1, ?BENCH_SIZE)],
        B = [float(X) || X <- lists:seq(1, ?BENCH_SIZE)],
        
        % Warm up
        {ok, _} = sied:dot_product_f32(A, B),
        
        % Benchmark
        {Time, _} = timer:tc(fun() ->
            lists:foreach(fun(_) ->
                {ok, _} = sied:dot_product_f32(A, B)
            end, lists:seq(1, ?BENCH_ITERATIONS))
        end),
        
        AvgTime = Time / ?BENCH_ITERATIONS,
        io:format("~nDot Product F32 (~p elements): ~.2f µs per operation~n", 
                  [?BENCH_SIZE, AvgTime]),
        
        ?assert(AvgTime < 10000)
    end}.

benchmark_sum_f32_test_() ->
    {timeout, 60, fun() ->
        A = [float(X) || X <- lists:seq(1, ?BENCH_SIZE)],
        
        % Warm up
        {ok, _} = sied:sum_f32(A),
        
        % Benchmark
        {Time, _} = timer:tc(fun() ->
            lists:foreach(fun(_) ->
                {ok, _} = sied:sum_f32(A)
            end, lists:seq(1, ?BENCH_ITERATIONS))
        end),
        
        AvgTime = Time / ?BENCH_ITERATIONS,
        io:format("~nSum F32 (~p elements): ~.2f µs per operation~n", 
                  [?BENCH_SIZE, AvgTime]),
        
        ?assert(AvgTime < 10000)
    end}.

benchmark_comparison_erlang_vs_nif_test_() ->
    {timeout, 60, fun() ->
        Size = 10000,
        A = [float(X) || X <- lists:seq(1, Size)],
        B = [float(X) || X <- lists:seq(1, Size)],
        Iterations = 100,
        
        % Erlang native implementation
        ErlangAdd = fun(L1, L2) ->
            lists:zipwith(fun(X, Y) -> X + Y end, L1, L2)
        end,
        
        % Benchmark Erlang
        {ErlangTime, _} = timer:tc(fun() ->
            lists:foreach(fun(_) ->
                _ = ErlangAdd(A, B)
            end, lists:seq(1, Iterations))
        end),
        
        % Benchmark NIF
        {NifTime, _} = timer:tc(fun() ->
            lists:foreach(fun(_) ->
                {ok, _} = sied:add_f32(A, B)
            end, lists:seq(1, Iterations))
        end),
        
        Speedup = ErlangTime / NifTime,
        io:format("~nComparison (~p elements, ~p iterations):~n", [Size, Iterations]),
        io:format("  Erlang: ~.2f µs per op~n", [ErlangTime / Iterations]),
        io:format("  NIF:    ~.2f µs per op~n", [NifTime / Iterations]),
        io:format("  Speedup: ~.2fx~n", [Speedup]),
        
        % NIF should be at least as fast as Erlang
        ?assert(Speedup >= 1.0)
    end}.

-endif.
