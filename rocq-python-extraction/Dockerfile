FROM ocaml/opam:debian-12-ocaml-5.3

# Install system build tools and Python 3 (needed for .py syntax validation)
RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        make \
        build-essential \
        pkg-config \
        libgmp-dev \
        linux-libc-dev \
        python3 \
    && sudo rm -rf /var/lib/apt/lists/*

# Add the rocq-prover opam repository, which carries rocq-stdlib.9.1.0 (not
# yet in the default opam repository).  Then install Rocq 9.1, rocq-stdlib
# 9.1.0, and dune, and clean up the opam cache.  rocq-stdlib.9.1.0 provides
# PrimInt63, PrimFloat, and PrimString (needed for MLuint/MLfloat/MLstring
# test coverage).
#
# The install is wrapped in a retry loop because the rocq-prover release
# server occasionally returns 504 Gateway Timeout on the stdlib tarball.
# Three attempts with 20-second back-off is enough to ride out transient
# upstream hiccups without failing the whole image build.
RUN opam repo add rocq-released https://rocq-prover.org/opam/released \
    && for attempt in 1 2 3; do \
         opam install -y rocq-core.9.1.0 rocq-stdlib.9.1.0 dune && break; \
         echo "opam install attempt $attempt failed — retrying in 20 s..."; \
         sleep 20; \
       done \
    && opam clean --all

WORKDIR /workspace
USER opam
