# S3CAMR base image (Python 3.12 on Ubuntu 24.04 LTS).
#
# This image builds and verifies the BASE execution path only: the random
# simulation engine (`scamr.py -s ... --prop-check`), which needs no external
# solvers. The optional falsification engines (SAL, Yices, GLPK, Z3, pysmt,
# gurobi, ipopt) are intentionally NOT installed here yet -- see the section
# at the bottom for how to add them later.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System Python 3.12 (Ubuntu 24.04 default) + build tools for any sdist deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        build-essential \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use an isolated virtualenv (Ubuntu 24.04 marks the system env as externally
# managed per PEP 668, so we must not pip-install into it).
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /opt/S3CAMR

# Install base Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# pyutils provides the `utils`, `err`, `fileops`, `constraints`, `streampickle`
# helper modules imported as top-level packages. Cloned from the py3-compat
# branch (Python 3 fixes).
RUN git clone --depth 1 --branch py3-compat \
        https://github.com/zutshi/pyutils.git /opt/pyutils

# Copy the (locally modernized) S3CAMR source tree.
COPY . /opt/S3CAMR

# `utils`, `err`, ... (pyutils) and the source packages live on PYTHONPATH.
ENV PYTHONPATH="/opt/pyutils:/opt/S3CAMR/src"
# Matplotlib has no display in the container; use a headless backend.
ENV MPLBACKEND=Agg

WORKDIR /opt/S3CAMR/src

# Smoke-test the base path at build time: 10 random simulations of the
# Van der Pol example. Fails the build if the base engine is broken.
RUN python -O ./scamr.py -f ../examples/vdp/vanDerPol.tst -s 10 --seed 0 --prop-check

# Default: print the CLI help.
CMD ["python", "-O", "./scamr.py", "--help"]

# ---------------------------------------------------------------------------
# OPTIONAL ENGINES (deferred -- add when needed):
#
#   Z3          pip install z3-solver           (--opt-engine z3 / bmc pysmtbmc)
#   pysmt       pip install pysmt               (--bmc-engine pysmtbmc)
#   GLPK        apt-get install libglpk-dev glpk-utils   (--opt-engine glpk;
#               the pyglpk C binding must also build against GLPK 5.0)
#   SAL 3.3     https://sal.csl.sri.com/opendownload/sal-3.3-bin-x86_64-unknown-linux-gnu-no-yices.tar.gz
#   Yices 2.7.0 https://github.com/SRI-CSL/yices2/releases/download/yices-2.7.0/yices-2.7.0-x86_64-pc-linux-gnu-static-gmp.tar.gz
#   gurobi      pip install gurobipy            (requires a valid license)
# ---------------------------------------------------------------------------
