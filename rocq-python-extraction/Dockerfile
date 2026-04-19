FROM ocaml/opam:debian-12-ocaml-5.3

# Install system build tools.  Python is installed via uv below; we do NOT
# use Debian's python3 package because we target Python 3.14t exclusively
# (free-threaded, no-GIL build) and that version is not in the Debian 12
# package archive.
RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
        make \
        build-essential \
        pkg-config \
        libgmp-dev \
        linux-libc-dev \
        curl \
    && sudo rm -rf /var/lib/apt/lists/*

# Install uv, then use it to fetch CPython 3.14t (free-threaded, no-GIL).
# Symlink the interpreter to /usr/local/bin/python3 so that Makefile and
# dune test rules can invoke it as plain [python3].  This is the sole Python
# runtime in the image — no system Python is installed.
ENV PATH="/home/opam/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.14t \
    && sudo ln -sf "$(uv python find 3.14t)" /usr/local/bin/python3 \
    && python3 --version

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

WORKDIR /workspace
USER opam
