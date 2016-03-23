# This example is from HyCU [Matlab's S3CAM implementation]
# Which was in turn taken from
# Reference:
# The Dynamics of a Bouncing Ball with a SinusoidallyVibrating Table Revisited
# ALBERT C. Jo LUO and RAY P. S. HAN



inf = float('inf')
from math import sin, cos, pi
plant_pvt_init_data = None

# %% 9-a, stable node(+)
# y = 0.0219918;
# %% 9-d, 2nd saddle, chaotic
phi = 1.0241592
# %% 9-e, 1st saddle, chaotic
# %y = 5.2590265;

# 9-a/b/c/d, stable node(+), stable focus. 1st/2nd saddle, chaotic
Y0 = -13.0666666

# %% 9-a, stable node(+)
# %y = 1.0400639;
# %% 9-d/e, 1st/2nd saddle, chaotic
A = 2.0003417

w = pi

initial_set = [[A*sin(phi), 1+A*sin(phi), A*w*cos(phi), Y0 - A*w*cos(phi)-1, 0],
               [A*sin(phi), 1+A*sin(phi+pi/4), A*w*cos(phi), Y0 - A*w*cos(phi)+1, 0]]
# P1
error_set = [[-inf, 16, -inf, -inf, 9.8],
             [inf, 19, inf, inf, 10.4]]
# # P2
# error_set = [[-inf, 16, -inf, -inf, 9.8],
#              [inf, 18.4, inf, inf, 10.4]]
# # P3
# error_set = [[-inf, 16, -inf, -inf, 9.8],
#              [inf, 18.2, inf, inf, 10.4]]

grid_eps = [0.001, 0.2, 0.001, 0.2, 0.001]

# Founds the most falsifications
delta_t = 0.1

delta_t = 2

num_samples = 2

T = 11.0

plant_description = 'python'
plant_path = 'bball_sin1_ass.py'

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
MAX_ITER = 4
