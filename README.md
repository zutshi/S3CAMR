Simulate and Scatter + Relationalization
===

S3CAMR is an improvement to S3CAM where refinement is carried symbolically.
Instead of using iterative refinement, we first build a piece-wise affine (PWA)
model. This discrete model is then queried for violation, which if found is
checked in the original continous model.

Installation
===
Clone the repository and install the dependencies using pip.
pip install -r requirements.txt

Dependencies (TODO)
===
###External
- SAL
    set SAL_PATH
- Yices
- S3CAMSMT
- Graph-Tool [graph-tool-2.13]
    - Needs Boost >= 1.60 [set environment variable BOOST_ROOT - not working]
        - ./configure .... 
        - set LD_LIBRARY_PATH LD_LIBRARY_PATH+=:../boost-1.60.0/lib/

###Python Modules
- Will include requirements.txt
