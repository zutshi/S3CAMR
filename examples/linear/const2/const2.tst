inf = float('inf')
plant_pvt_init_data = None

initial_set = [[-1, -1], [1, 1]]
error_set = [[5, 5], [6, 6]] # x\in[5,6], y\in[5,6]

grid_eps = [0.1, 0.1]
num_samples = 1
delta_t = 0.2
T = 3.0

plant_description = 'python'
plant_path = 'const2.py'


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
