FROM ocaml/opam:debian-12-ocaml-5.3 AS builder

# Install system build tools needed to build Rocq packages. Python tooling runs
# on the host via uv; this image only needs the Rocq toolchain.
RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        make \
        build-essential \
        pkg-config \
        libgmp-dev \
        linux-libc-dev \
    && sudo rm -rf /var/lib/apt/lists/*

# Add the rocq-prover opam repository, which carries rocq-stdlib.9.1.0 (not
# yet in the default opam repository).  Then install Rocq 9.2.0, rocq-stdlib
# 9.1.0, and a pinned dune, and clean up the opam cache.  rocq-stdlib.9.1.0
# provides PrimInt63, PrimFloat, and PrimString (needed for MLuint/MLfloat/
# MLstring test coverage); it is compatible with rocq-core >= 9.0 so it
# works with rocq-core.9.2.0.
#
# rocq-core is pinned to 9.2.0 because earlier 9.1.x versions have a bug
# where pattern matching on a nullary constructor of a parameterised inductive
# (e.g. MNil in MyList A) fails with:
#
#   Error: The constructor MNil (in type MyList) is expected to be applied
#   to 1 argument while it is actually applied to no arguments.
#
# even with [Unset Universe Polymorphism] and [Arguments MNil {A}] in place.
# In 9.1.x the inductive parameter is not automatically made implicit in
# constructor arguments during pattern elaboration; 9.2.0 restores the
# expected behaviour.
#
# g_python_extraction.mlg uses [DirPath.Set.empty] (Rocq 9.2+ API) instead
# of [[]] for the [used_modules] argument to [preamble]; that change is also
# present in this branch.
#
# dune is pinned to 3.21.1 because dune-project declares (lang dune 3.21);
# this keeps the build tool in sync with the declared language version.
#
# The install is wrapped in a retry loop because the rocq-prover release
# server occasionally returns 504 Gateway Timeout on the stdlib tarball.
# Three attempts with 20-second back-off is enough to ride out transient
# upstream hiccups without failing the whole image build.
RUN opam repo add rocq-released https://rocq-prover.org/opam/released \
    && for attempt in 1 2 3; do \
         opam install -y rocq-core.9.2.0 rocq-stdlib.9.1.0 dune.3.21.1 && break; \
         echo "opam install attempt $attempt failed — retrying in 20 s..."; \
         sleep 20; \
       done \
    && opam clean --all

FROM debian:12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        gcc \
        libc6-dev \
        libgmp-dev \
        linux-libc-dev \
        make \
        pkg-config \
        tar \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf \
        /usr/share/doc/* \
        /usr/share/info/* \
        /usr/share/locale/* \
        /usr/share/man/* \
    && useradd -m -s /bin/bash opam

COPY --from=builder --chown=opam:opam /home/opam/.opam/5.3 /home/opam/.opam/5.3

ENV OPAM_SWITCH_PREFIX="/home/opam/.opam/5.3"
ENV CAML_LD_LIBRARY_PATH="/home/opam/.opam/5.3/lib/stublibs"
ENV OCAML_TOPLEVEL_PATH="/home/opam/.opam/5.3/lib/toplevel"
ENV OCAMLTOP_INCLUDE_PATH="/home/opam/.opam/5.3/lib/toplevel"
ENV PATH="/home/opam/.opam/5.3/bin:${PATH}"

RUN printf '%s\n' \
      '#!/bin/sh' \
      'set -eu' \
      'if [ "$#" -ge 1 ] && [ "$1" = "exec" ]; then' \
      '  shift' \
      '  if [ "$#" -ge 1 ] && [ "$1" = "--" ]; then shift; fi' \
      '  exec "$@"' \
      'fi' \
      'echo "minimal opam shim only supports: opam exec -- <command>" >&2' \
      'exit 2' \
      > /usr/local/bin/opam \
    && chmod +x /usr/local/bin/opam

WORKDIR /workspace
USER opam
