# REFERENCE:
# EMSOFT paper

inf = float('inf')
plant_pvt_init_data = None

# Property
initial_set = [[-0.1, 3.0, 5.0, -5.0],
               [0.1, 4.0, 10.0, 5.0]]

error_set = [[330.0, 1.0, -inf, -inf],
             [330.1, 1.1, inf, inf]]
# error_set = [[330.0, -inf, -inf, -inf],
#              [inf, inf, inf, inf]]

T = 40.0

# Abstraction params
grid_eps = [5.0]*4
delta_t = 10.0
num_samples = 50


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
