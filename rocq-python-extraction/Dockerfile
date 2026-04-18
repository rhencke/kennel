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

# Install Rocq 9.2 and dune, clean up opam cache.
# rocq-stdlib has no release compatible with rocq-core.9.2.0 in the default
# opam repo; the Stdlib theories are bundled inside rocq-core itself and found
# via Rocq's own search path without a separate opam package.
RUN opam install -y rocq-core.9.2.0 dune \
    && opam clean --all

WORKDIR /workspace
USER opam
