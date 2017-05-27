# REFERENCE:
# EMSOFT paper


# Viloations
#
# S
# x0=[  4.14327461  21.51742883   0.8471542    0.9884877 ] -> x=[ 22.44414369  11.999        0.22792075  -0.86384343], t=30.1382177243
# x0=[  4.50024758  21.82478644   0.97146715   0.6993149 ] -> x=[ 22.4524101   11.999        0.22748131  -0.86401213], t=30.1798284278


inf = float('inf')
plant_pvt_init_data = None

# Property
initial_set = [[4.0, 21.0, -1.0, -1.0],
                [5.0, 22.0, 1.0, 1.0]]

ROI = [[-1, -1, -5,-5],
       [26, 26, 5, 5]]

S = [[22., 11., -inf, -inf],
     [23., 12., inf, inf]]

error_set = S

T = 35.0

# Abstraction params: P, Q, R, S
#grid_eps = [0.2, 0.2, 0.4, 0.4]
#grid_eps = [0.21, 0.21, 0.41, 0.41]
#grid_eps = [0.51]*4


# # works but not everytime
grid_eps = [0.1]*4
delta_t = 10.0
num_samples = 2


#grid_eps = [0.21]*4
#delta_t = 10.0
#num_samples = 10



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
