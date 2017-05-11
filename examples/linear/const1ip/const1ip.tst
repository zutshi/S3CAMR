inf = float('inf')
plant_pvt_init_data = None

initial_set = [[0, 0], [1, 1]]
error_set = [[-inf, 5], [inf, inf]] # y >= 5


# Param 1
grid_eps = [1, 1]
num_samples = 10

delta_t = 0.2
T = 3.0


plant_description = 'python'
plant_path = 'const1ip.py'


pi = [[-1, -1],[0.999, 0.999]]
pi_grid_eps = [1, 1]

#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0
controller_path = None
controller_path_dir_path = None
initial_discrete_state = []
initial_private_state = []
ci = [[], []]

MAX_ITER = 4
