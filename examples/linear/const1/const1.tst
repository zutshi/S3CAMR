inf = float('inf')
plant_pvt_init_data = None

initial_set = [[0, 0], [1, 1]]
error_set = [[-inf, 5], [inf, inf]] # y >= 5


# Param 1
grid_eps = [1, 1]
num_samples = 10

delta_t = 0.2
T = 3.0

# All below work for dyn1. Untested for others
# grid_eps = [1, 1]
# num_samples = 10
#
# grid_eps = [2, 2]
# num_samples = 10
#
# grid_eps = [5, 5]
# num_samples = 10
#
# grid_eps = [0.1, 0.1]
# num_samples = 1


plant_description = 'python'
plant_path = 'const1.py'


ci = [[], []]
pi = [[],[]]

#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0
controller_path = None
controller_path_dir_path = None
initial_discrete_state = []
initial_private_state = []
MAX_ITER = 4
