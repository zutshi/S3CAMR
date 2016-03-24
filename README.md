#Simulate and Scatter + Relationalization

S3CAMR is an improvement to S3CAM where refinement is carried symbolically.
Instead of using iterative refinement, we first build a piece-wise affine (PWA)
model. This discrete model is then queried for violation, which if found is
checked in the original continous model.

##Current Status: [Partial Implementation]
- SAL + YICES are required as they are the only working BMC engine.

##Installation
Clone the repository and install the dependencies.

##### Python Modules: Install using the Makefile

    sudo make install
#####External Dependencies [Required]
- py_utils
    Clone the below repo and add it to the Python Path environment variable: PYTHONPATH

    ```
    https://github.com/zutshi/pyutils.git
    ```

- SAL [optional: model checker]
    - Download and install SAL: http://sal.csl.sri.com/
    - set environment variable SAL_PATH
    - Yices2 [Performs better than Yices]
        - Download and install Yices2: http://yices.csl.sri.com/

#####External Dependencies [Optional]
- S3CAMSMT [**optional bmc engine**]
    Still being developed

    ```
    https://github.com/cuplv/S3CAMSMT.git
    ```

- ghraph-tool-2.13 [ (**optional**): faster graph library]
  Partial integration. Instead of K-shortest paths, All-shortest paths are being computed!

    - Warning: Takes a few hours to compile (not painless)
    - Needs Boost >= 1.60 [set environment variable BOOST_ROOT - not working]
        - Install using `./configure .... `
        - set `LD_LIBRARY_PATH LD_LIBRARY_PATH+=:../boost-1.60.0/lib/`

##Usage

**Print List of Options**
    
    python ./scamr.py --help

####Example runs

Navigate to the source directory ./src before running the below examples.

**Random Testing using Simulations:**
Run 10 simulations

    python -O ./scamr.py -f <filename>.tst -s 10

**Plotting: only supported for random simulations**

- Plot all state variabless against time using either Matplotlib(mp) or PyQTGraph(pg)
    
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
S3CAM/R are random in nature. Their output can be made reproducible by using the same seed passed using the switch.

    --seed <integer>
