.PHONY: build build_if_needed test publish clean docs

# Per-platform NIF name: sied-<os>-<arch>. Erlang loads NIFs as .so on every
# Unix (macOS included) and .dll on Windows, so the built .dylib is copied to .so.
ifeq ($(OS),Windows_NT)
    OS_TAG   := windows
    NIF_EXT  := dll
    BUILT    := sied.dll
    UNAME_M  := $(PROCESSOR_ARCHITECTURE)
else
    UNAME_S := $(shell uname -s)
    UNAME_M := $(shell uname -m)
    NIF_EXT := so
    ifeq ($(UNAME_S),Darwin)
        OS_TAG := darwin
        BUILT  := libsied.dylib
    else
        OS_TAG := linux
        BUILT  := libsied.so
    endif
endif

ifeq ($(filter arm64 aarch64 ARM64,$(UNAME_M)),)
    ARCH_TAG := x86_64
else
    ARCH_TAG := aarch64
endif

NIF_NAME := sied-$(OS_TAG)-$(ARCH_TAG).$(NIF_EXT)

# Locate cargo: prefer PATH, fall back to the default rustup location.
CARGO := $(shell command -v cargo 2>/dev/null || echo $(HOME)/.cargo/bin/cargo)

build:
	cd native/sied && $(CARGO) build --release
	mkdir -p priv
	cp native/sied/target/release/$(BUILT) priv/$(NIF_NAME)

build_if_needed:
	@if [ ! -f "priv/$(NIF_NAME)" ]; then $(MAKE) build; fi

test: build
	rebar3 do eunit, ct

publish: test docs
	rebar3 hex publish

docs:
	rebar3 edoc

clean:
	rm -rf _build native/sied/target priv/*

