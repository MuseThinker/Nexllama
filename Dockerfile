# syntax=docker/dockerfile:1
#
# CPU image for the Nexllama inference daemon.
#
# Build:
#   docker build -t nexllama/infer-daemon:latest .
#   docker build --build-arg CRATES_MIRROR=https://rsproxy.cn/index/ -t nexllama/infer-daemon:latest .
#
# Run:
#   docker run --rm -p 8090:8090 -v "$PWD/models:/models:ro" \
#     nexllama/infer-daemon:latest --model /models/model.gguf

FROM rust:1-bookworm AS builder

ARG CRATES_MIRROR=
ARG FEATURES=

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      clang libclang-dev llvm-dev cmake ninja-build pkg-config \
 && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$CRATES_MIRROR" ]; then \
      printf '[source.crates-io]\nreplace-with = "mirror"\n\n[source.mirror]\nregistry = "sparse+%s"\n' "$CRATES_MIRROR" > "$CARGO_HOME/config.toml"; \
    fi

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
# Every workspace member, not only the daemon's own path dependencies: the root
# `Cargo.toml` names them all, so a manifest that was not copied fails the build
# before anything is compiled — and the list only grows. `crates/app` is the
# desktop app's own workspace and is excluded in `.dockerignore`.
COPY crates ./crates

RUN if [ -n "$FEATURES" ]; then \
      cargo build --release -p infer-daemon --features "$FEATURES"; \
    else \
      cargo build --release -p infer-daemon; \
    fi

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/infer-daemon /usr/local/bin/infer-daemon
# The llama.cpp shared libraries go in a library directory, not in `bin`: the
# dynamic loader does not search `bin`, so the binary only started by luck of
# whatever fallback this base image happened to have.
COPY --from=builder /src/target/release/*.so* /usr/local/lib/
RUN ldconfig

# Unprivileged: reading model files and serving a port needs no root, and the
# daemon parses whatever GGUF it is pointed at.
RUN useradd --system --no-create-home --uid 10001 nexllama
USER nexllama

ENV RUST_LOG=info
EXPOSE 8090
VOLUME ["/models"]

ENTRYPOINT ["infer-daemon"]
# `--allow-unauthenticated` is deliberate: the container binds 0.0.0.0 so the
# host can reach it, and nothing in front of it authenticates callers. Put a
# reverse proxy (or `--api-key`) in front before exposing this to a network.
CMD ["--host", "0.0.0.0", "--port", "8090", "--allow-unauthenticated"]
