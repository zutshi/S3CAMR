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

###Python Modules
- Will include requirements.txt
