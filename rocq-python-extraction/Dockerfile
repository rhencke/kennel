FROM ocaml/opam:debian-12-ocaml-5.3

# Install system build tools
RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        make \
        build-essential \
        pkg-config \
        libgmp-dev \
        linux-libc-dev \
    && sudo rm -rf /var/lib/apt/lists/*

# Install Rocq 9.2 (core + stdlib) and dune, clean up opam cache
RUN opam install -y rocq-core.9.2.0 rocq-stdlib.9.2.0 dune \
    && opam clean --all

WORKDIR /workspace
USER opam
