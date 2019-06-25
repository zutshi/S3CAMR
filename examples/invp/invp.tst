inf = float('inf')
plant_pvt_init_data = None

# Property
initial_set = [[0, -0.1, 0, 0], [0, 0.1, 0, 0]]
error_set = [[-inf, -3, -inf, -inf], [inf, 3, inf, inf]]
T = 20.0

grid_eps = [0.5]*4
delta_t = 1
num_samples = 2


MAX_ITER = 5


plant_description = 'python'
plant_path = 'invp.py'

#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0

ci = [[],[]]
pi = [[],[]]
controller_path = None
controller_path_dir_path = None
initial_discrete_state = []
initial_private_state = []
