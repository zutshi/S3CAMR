# S3CAMR
Simulate and Scatter + Relationalization

S3CAMR is an improvement to S3CAM where refinement is carried symbolically.
Instead of using iterative refinement, we first build a piece-wise affine (PWA)
model. This discrete model is then queried for violation, which if found is
checked in the original continuous model.

## Current Status: [Partial Implementation]
- SAL + YICES are required as they are the only working BMC engine.

## Installation
Clone the repository and install the dependencies.

Python modules available on PyPI (Python Package Index) can be installed using the Makefile

    sudo make install

External Dependencies can be installed as required. Refer to the below section.

## Dependencies

- py_utils
    Clone the below repo and add it to the Python Path environment variable: PYTHONPATH

    ```
    https://github.com/zutshi/pyutils.git
    ```

- Graph Library (two options)
    1. Networkx
        - Slower
        - Gets installed using make
    2. graph-tool-2.13
        - Faster
        - Warning: Takes a few hours to compile (and painful to install)
        - Partial integration. Instead of K-shortest paths, All-shortest paths are being computed!
        - Needs Boost >= 1.60 [set environment variable BOOST_ROOT - not working]
            - Install using `./configure .... `
            - set `LD_LIBRARY_PATH LD_LIBRARY_PATH+=:../boost-1.60.0/lib/`

- BMC engine (two options)
    1. SAL
        - Download and install SAL: http://sal.csl.sri.com/
        - set environment variable SAL_PATH to point to the installation
        ```
        export SAL_PATH='<path>/sal-3.3/'
        ```
        - Yices2 [Performs better than Yices]
            - Download and install Yices2: http://yices.csl.sri.com/

    2. S3CAMSMT [**under development**]
    
        ```
        https://github.com/cuplv/S3CAMSMT.git
        ```

## Usage

**Print List of Options**
    
    python ./scamr.py --help

#### Example runs

Navigate to the source directory `./src` before running the below examples.

**Random Testing using Simulations:**
Run 10 simulations

    python -O ./scamr.py -f <filename>.tst -s 10

**Plotting: only supported for random simulations**

- Plot all state variables against time using either Matplotlib(mp) or PyQTGraph(pg)
    ```
    --p [mp, pg]
    ```
- Plot only x0 vs x1, t vs x2 and x4 vs x7
    ```
    --p [mp, pg] --plots x0-x1 t-x2 x4-x7
    ```

**Falsification Search using S3CAM:**

    python -O ./scamr.py -f <filename>.tst -cn

**Falsification Search using S3CAMR (using different time-discretization models):**
- Use DFT (Discrete-time Fixed Time Step) model

    ```
    python -O ./scamr.py -f <filename>.tst -cn --refine model_dft
    ```
- Use DMT (Discrete-time Multi Time Step) model (**broken**)

    ```
    python -O ./scamr.py -f <filename>.tst -cn --refine model_dmt
    ```
- Use DCT (Discrete-time Continuous Time Step) model (**not-implemented**)

    ```
    python -O ./scamr.py -f <filename>.tst -cn --refine model_dct
    ```

**Reproducible output using seeds:**
S3CAMR is random in nature. It's output can be made reproducible by using the same seed passed using the switch.

    --seed <integer>

**Example with all switches**

    python -O ./scamr.py -f ../examples/vdp/vanDerPol.tst -cn --prop-check --refine model-dft --max-model-error 10 --incl-error -p --seed 0

Explanation: Run optimized scamr (no debug prints, no assertion checks) `-O`. Falsify Van der Pol example using scatter-and-simulate algorithm `-cn`, but use a discrete fixed time model for refinement `--refine model-dft`. The model must not have L2-norm error of more than 10 `--max-model-error 10`, if it does, split cells till the condition is satisfied. Include any error `--incl-error` as x' = Ax +-error in the BMC encoding. Plot the results using the default library. Determinize the pseudorandom generator by fixing the seed `--seed 0`.

## Generate Benchmark Results

Navigate to the source directory `./src` before running the below examples.

    time ./scamr.py -f <test-file> -cn  --refine model-dft --prop-check --incl-error --max-model-error 10 --max-paths 0 --bmc-engine pwa --clustering cell

Where the `<test-file>` is one of the below
* `../examples/vdp/vanDerPol_s3camr.tst`
* `../examples/brusselator/brusselator.tst`
* `../examples/lorenz/lorenz.tst`
* `../examples/nav/nav30P.tst`
* `../examples/nav/nav30Q.tst`
* `../examples/nav/nav30R.tst`
* `../examples/nav/nav30S.tst`
* `../examples/bball/bball.tst`

10 runs were generated using
`for i in $(seq 1 10); do { <> ; } done > output_log  2>&1`
    
    
## Common Issues

- PyGObject related issues

    **Reason**: Missing GTK3 libraries
    
    **Details**: S3CAMR uses Matplotlib as one of the optional plotting lib. It also uses graph-tool as an optional graph lib. Matplotlib by default (at least on Ubuntu 12.04-14.04) uses Qt4Agg as its backend which uses GTK by default. graph-tool on the other hand uses GTK3 as its backend. As both GTK2 and GTK3 are not compatible we switch Matplotlib's backend to GTK3Agg (plotMP.py).
    
    **Solutions**: 
    - If not using graph-tool, simply uncomment the line `matplotlib.use('GTK3Agg')` in plotMP.py
    - Otherwise, either switch Matplotlib's backend to something else than GTK2/3 or install GTK3 [suggestions for Ubuntu only!].

            sudo apt-get install python-gobject python-gi
            sudo apt-get install libgtk-3-dev
    
    **Refer**: 
    -   https://git.skewed.de/count0/graph-tool/issues/98
    -   http://matplotlib.org/faq/usage_faq.html
