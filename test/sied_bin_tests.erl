-module(sied_bin_tests).

%%%===================================================================
%%% EUnit tests for the binary-first API (sied_bin).
%%%===================================================================
-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

-define(F32(Vals), (<< <<V:32/float-little>> || V <- Vals >>)).
-define(U16(Vals), (<< <<V:16/unsigned-little>> || V <- Vals >>)).
-define(EPS, 1.0e-5).

%%--------------------------------------------------------------------
%% Typed decode
%%--------------------------------------------------------------------

decode_u16_f32_test() ->
    {ok, Out} = sied_bin:decode_u16_f32(?U16([0, 1, 2, 65535]), 1.0),
    ?assertEqual([0.0, 1.0, 2.0, 65535.0], f32_list(Out)).

decode_u16_f32_scaled_test() ->
    {ok, Out} = sied_bin:decode_u16_f32(?U16([2, 4, 10]), 0.5),
    ?assertEqual([1.0, 2.0, 5.0], f32_list(Out)).

decode_u16_f32_odd_test() ->
    ?assertEqual({error, invalid_input}, sied_bin:decode_u16_f32(<<1, 0, 2>>, 1.0)).

%%--------------------------------------------------------------------
%% Elementwise
%%--------------------------------------------------------------------

add_f32_bin_test() ->
    {ok, Out} = sied_bin:add_f32_bin(?F32([1.0, 2.0, 3.0]), ?F32([4.0, 5.0, 6.0])),
    ?assertEqual([5.0, 7.0, 9.0], f32_list(Out)).

subtract_f32_bin_test() ->
    {ok, Out} = sied_bin:subtract_f32_bin(?F32([5.0, 3.0]), ?F32([2.0, 1.0])),
    ?assertEqual([3.0, 2.0], f32_list(Out)).

multiply_f32_bin_test() ->
    {ok, Out} = sied_bin:multiply_f32_bin(?F32([2.0, 3.0]), ?F32([4.0, 5.0])),
    ?assertEqual([8.0, 15.0], f32_list(Out)).

divide_f32_bin_test() ->
    {ok, Out} = sied_bin:divide_f32_bin(?F32([6.0, 9.0]), ?F32([2.0, 3.0])),
    ?assertEqual([3.0, 3.0], f32_list(Out)).

elementwise_length_mismatch_test() ->
    ?assertEqual({error, length_mismatch},
                 sied_bin:add_f32_bin(?F32([1.0, 2.0]), ?F32([3.0]))).

%%--------------------------------------------------------------------
%% Reductions
%%--------------------------------------------------------------------

sum_f32_bin_test() ->
    {ok, S} = sied_bin:sum_f32_bin(?F32([1.0, 2.0, 3.0, 4.0])),
    ?assert(abs(S - 10.0) < ?EPS).

dot_product_f32_bin_test() ->
    {ok, D} = sied_bin:dot_product_f32_bin(?F32([1.0, 2.0, 3.0]), ?F32([4.0, 5.0, 6.0])),
    ?assert(abs(D - 32.0) < ?EPS).

%%--------------------------------------------------------------------
%% Fused normalized difference
%%--------------------------------------------------------------------

normalized_difference_test() ->
    {ok, Out} = sied_bin:normalized_difference_f32_bin(?F32([0.8, 1.0, 5.0]),
                                                       ?F32([0.2, 0.0, 5.0])),
    %% (0.8-0.2)/1.0=0.6 ; (1.0-0.0)/1.0=1.0 ; (5-5)/10=0.0
    [V1, V2, V3] = f32_list(Out),
    ?assert(abs(V1 - 0.6) < ?EPS),
    ?assert(abs(V2 - 1.0) < ?EPS),
    ?assert(abs(V3 - 0.0) < ?EPS).

normalized_difference_zero_denominator_test() ->
    {ok, Out} = sied_bin:normalized_difference_f32_bin(?F32([0.0]), ?F32([0.0])),
    ?assertEqual([0.0], f32_list(Out)).

%%--------------------------------------------------------------------

f32_list(<<>>) -> [];
f32_list(<<V:32/float-little, R/binary>>) -> [V | f32_list(R)].

-endif.
