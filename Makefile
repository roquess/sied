.PHONY: build test publish clean docs

# Auto-detect platform
ifeq ($(OS),Windows_NT)
    DLL_EXT := .dll
    NIF_NAME := sied.dll
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        DLL_EXT := .so
        NIF_NAME := libsied.so
    endif
    ifeq ($(UNAME_S),Darwin)
        DLL_EXT := .dylib
        NIF_NAME := libsied.dylib
    endif
endif

build:
	cd native/sied && cargo build --release
	mkdir -p priv
	cp native/sied/target/release/$(NIF_NAME) priv/

test: build
	rebar3 do eunit, ct

publish: test docs
	rebar3 hex publish

docs:
	rebar3 edoc

clean:
	rm -rf _build native/sied/target priv/*

