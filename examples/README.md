#Examples

Various demos/benchmarks.

### Example Name [ODE solver library used]
  - Van Der Pol [scipy]
  - Bouncing ball [pydstool]
  - Bouncing ball + sinusoidally oscillating platform [Assimulo]
  - Helicopter [scipy]

### Dependencies
  - scipy
  - pydstool: `pip install pydstool`
  - Assimulo: [Install](http://www.jmodelica.org/assimulo_home/installation.html)
    Assimulo can be installed from PyPI.
    Before installing Assimulo, make sure that [sundials](https://computation.llnl.gov/casc/sundials/download/download.php) built with -fPIC is installed in the system.
    Download sundials and read the docs to install it. Version 2.6.2 can be installed as below.

        ```
        tar -xvf sundials-2.6.2.tar.gz
        cd sundials-2.6.2
        mkdir build
        cd build
        cmake ../ -DCMAKE_C_FLAGS=-fPIC
        make
        make install
        ```
    Now install Assimulo using `pip install assimulo`.
