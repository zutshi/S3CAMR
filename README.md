#Simulate and Scatter + Relationalization

S3CAMR is an improvement to S3CAM where refinement is carried symbolically.
Instead of using iterative refinement, we first build a piece-wise affine (PWA)
model. This discrete model is then queried for violation, which if found is
checked in the original continous model.

##Installation
Clone the repository and install the dependencies.

##### Python Modules: Install using the Makefile
    sudo make install

#####External Dependencies
- S3CAMSMT

    ```
    https://github.com/cuplv/S3CAMSMT.git
    ```
- py_utils

    ```
    https://github.com/zutshi/pyutils.git
    ```

- SAL [optional: model checker]
    - Download and install SAL: http://sal.csl.sri.com/
    - set environment variable SAL_PATH
    - ~~Yices~~
        - Download and install Yices: http://yices.csl.sri.com/

- ghraph-tool-2.13 [optional: faster graph library]
    - Needs Boost >= 1.60 [set environment variable BOOST_ROOT - not working]
        - Install using `./configure .... `
        - set `LD_LIBRARY_PATH LD_LIBRARY_PATH+=:../boost-1.60.0/lib/`

##Usage

**Print List of Options**
    
    python ./scamr.py --help

####Examples
**Random Testing using Simulations:**
Run 10 simulations

    python -O ./scamr.py -f <filename>.tst -s 10
and plot them...

    python -O ./scamr.py -f <filename>.tst -s 10 -p

**Falsification Search using S3CAM:**
    python -O ./scamr.py -f <filename>.tst -cn

**Falsification Search using S3CAMR (using different time-discretization models):**
- Use DFT (Discrete-time Fixed Time Step) model

    ```
    python -O ./scamr.py -f <filename>.tst -cn --refine model_dft
    ```
- Use DMT (Discrete-time Multi Time Step) model

    ```
    python -O ./scamr.py -f <filename>.tst -cn --refine model_dmt
    ```
- Use DCT (Discrete-time Continuous Time Step) model [NOT-IMPLEMENTED]

    ```
    python -O ./scamr.py -f <filename>.tst -cn --refine model_dct
    ```

**Reproducible output using seeds:**
S3CAM/R are random in nature. Their output can be made reproducible by using the same seed passed using the switch.

    --seed <integer>
