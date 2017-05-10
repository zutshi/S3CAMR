# REFERENCE:
# EMSOFT paper

inf = float('inf')
plant_pvt_init_data = None


# x0 = np.array([4, 5], [21, 22])
# v0 = np.array([[0.2, 0.5], [-0.5, 0.5]])

# Property
initial_set = [[4.0, 21.0, -1.0, -1.0],
                [5.0, 22.0, 1.0, 1.0]]

ROI = [[-1, -1, -5,-5],
       [26, 26, 5, 5]]


P = [[6., 7., -inf, -inf],
     [7., 8., inf, inf]]

Q = [[7., 9., -inf, -inf],
     [8., 10., inf, inf]]

R = [[1., 6., -inf, -inf],
     [2., 7., inf, inf]]

S = [[22., 11., -inf, -inf],
     [23., 12., inf, inf]]

error_set = Q

# P, Q, R
T = 20.0

# S
#T = 35.0

# Abstraction params: P, Q, R, S
#grid_eps = [0.2, 0.2, 0.4, 0.4]
#grid_eps = [0.21, 0.21, 0.41, 0.41]
#grid_eps = [0.51]*4


# Working set for P, Q
# ./scamr.py -f ../examples/nav/nav30.tst -cn  --refine model-dft --prop-check --incl-error --seed 0 --max-model-error 10 --max-paths 1000
#
# grid_eps = [1.1]*4
# delta_t = 5.0
# num_samples = 100
#
# grid_eps = [0.11]*4
# delta_t = 5.0
# num_samples = 2

grid_eps = [0.2]*4
delta_t = 5.0
num_samples = 10


# Gets the right x0 from linprog: but the interval is big enough to
# concretize succesfully
# grid_eps = [5.1]*4
# delta_t = 5.0
# num_samples = 100


# grid_eps = [1.1]*4
# delta_t = 5.0
# num_samples = 10 #[but all paths]


# #SCAM
# grid_eps = [0.21, 0.21, 0.41, 0.41]
# delta_t = 5.0
# num_samples = 2


# Worked for R
# grid_eps = [0.6]*4
# delta_t = 5.0
# num_samples = 20

# [1.1]*4 did not work for S
# Trying below
# grid_eps = [0.6]*4
# delta_t = 5.0
# num_samples = 200

MAX_ITER = 4

plant_description = 'python'
plant_path = 'nav30.py'

#############################################
############## Don't care params ############
#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0

ci = [[], []]
pi = [[],[]]

controller_path = None
controller_path_dir_path = None
initial_discrete_state = []
initial_private_state = []


# Viloations
#
# S
# x0=[  4.14327461  21.51742883   0.8471542    0.9884877 ] -> x=[ 22.44414369  11.999        0.22792075  -0.86384343], t=30.1382177243
# x0=[  4.50024758  21.82478644   0.97146715   0.6993149 ] -> x=[ 22.4524101   11.999        0.22748131  -0.86401213], t=30.1798284278

