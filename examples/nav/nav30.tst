# REFERENCE:
# EMSOFT paper

raise NotIimplementedError('Do not run this test. Instead, run the tests created for each property.')

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

error_set = S

# P, Q, R
#T = 20.0

# S
T = 35.0

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

grid_eps = [0.1]*4
delta_t = 5.0
num_samples = 2


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

# R
# random_testing.py:62::x0=[  4.00016738  21.69066716  -0.73087554  -0.5708336 ] -> x=[ 1.14161854  6.999       0.21690255 -0.88975963], t=17.3354007675
# random_testing.py:62::x0=[  4.0486398   21.18421083  -0.31169147  -0.92112214] -> x=[ 1.16761391  6.999       0.22853109 -0.88741714], t=16.1150375174
# random_testing.py:62::x0=[  4.65965605  21.19273923  -0.90129258  -0.58789132] -> x=[ 1.17786732  6.999       0.2356895  -0.88866338], t=16.4221797659
# random_testing.py:62::x0=[  4.51301209  21.67417795  -0.89205661  -0.60742149] -> x=[ 1.18638306  6.999       0.2419798  -0.88942524], t=17.0332007719

