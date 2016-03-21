inf = float('inf')
plant_pvt_init_data = None

nd = 28

# x_i \in [0, 0.1] if i = {1..8}
#     \in [0, 0] else
initial_set = [[0.]*nd,
               [0.1]*8 + [0.]*20]

error_set = [[0]*nd,
             [0]*nd]

grid_eps = [0.04, 0.04]
delta_t = 0.1
num_samples = 5


T = 30.0


MAX_ITER = 4


plant_description = 'python'
plant_path = 'heli.py'

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
