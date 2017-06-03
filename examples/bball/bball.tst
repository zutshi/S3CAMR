# REFERENCE:
# EMSOFT paper

inf = float('inf')
plant_pvt_init_data = None

# Property

ROI = [[-10, 0, 5.0, -15.0],
       [500, 20, 10.0, 15.0]]

initial_set = [[-0.1, 3.0, 5.0, -5.0],
               [0.1, 4.0, 10.0, 5.0]]

# initial_set = [[0, 3.0, 9.4, 3.91689],
#                [0, 3.322, 9.4, 3.91689]]


# Solves the original problem (in matlab atlaest)
# x(0) = -0.09996, y(0) = 3.999, vx(0) = 9.8672, vy(0) = 4.968.
# example error
# random_testing.py:208::
# x0=[ 0.04264138  3.48107684  9.99909701  4.98550744] ->
# x=[ 330.01284287    1.06944299    9.99909701   -1.0229609 ], t=33.0

# X0 leading to error set
#initial_set = [[-0.04308, 3.4704, 9.9862, 4.9871],
#               [0.04300, 3.4705, 9.9863, 4.9872]]


error_set = [[330.0, 1.0, -inf, -inf],
            [330.1, 1.1, inf, inf]]

# Easier
# error_set = [[330.0, 0.5, -inf, -inf],
#              [330.1, 1.1, inf, inf]]

# error_set = [[330.0, -inf, -inf, -inf],
#              [inf, inf, inf, inf]]

T = 40.0

# Abstraction params used by matlabprot
#grid_eps = [1]*4

# Does work at times...with delta_t = 40
#grid_eps = [1.1, 1.1, 1.1, 1.1]

# Might work, but have not observed success yet.
#grid_eps = [10, 1, 1, 1]
#delta_t = 40.0
#num_samples = 2

# Specially confusing, since the BMC comes back with no results...
# delta_t = 5.0
# num_samples = 10

# Abstraction params
#grid_eps = [20, 2, 2, 2]
#delta_t = 10.0
#num_samples = 50

# Abstraction params: coarse
#grid_eps = [50, 20, 20, 20]
#
# delta_t = 5.0
# num_samples = 100

# Abstraction params: coarse
#grid_eps = [50.1, 10.1, 10.1, 10.1]
# grid_eps = [25.1, 2.1, 25.1, 2.1]
#grid_eps = [0.001]*4
#delta_t = 5.0
#num_samples = 20

# works..as in, it comes up with a path which can be concretized
# However, the graph is very very large, with many many paths
#delta_t = 5.0
#num_samples = 100
#grid_eps = [25.1, 2.1, 25.1, 2.1]

# 0.5, 1, 2, 5, 10
#grid_eps = [2]*4
# grid_eps = [0.5]*4
# delta_t = 5.0
# num_samples = 20

#grid_eps = [1.0]*4
#delta_t = 20.0
#num_samples = 10


# takes a long time: for seed 0 returned safe after checking all paths
# using bmc: pwa
# grid_eps = [0.5]*4
# delta_t = 10.0
# num_samples = 50

# # Used for emsoft submission: gives negative result
# grid_eps = [0.4]*4
# delta_t = 10.0
# num_samples = 10


# Used for HyCU (matlabprot)
grid_eps = [1]*4
delta_t = 10.0
num_samples = 2


MAX_ITER = 4

plant_description = 'python'
plant_path = 'bball.py'

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
