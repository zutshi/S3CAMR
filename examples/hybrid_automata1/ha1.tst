inf = float('inf')
plant_pvt_init_data = None

initial_set = [[60], [62]]
error_set = [[72], [80]]

################
# what worked
################
grid_eps = [1]
num_samples = 2
delta_t = 1
pi = [[-50],[50]]
pi_grid_eps = [50]


initial_discrete_state = [0]
initial_private_state = []

T = 100.0


MAX_ITER = inf


plant_description = 'python'
plant_path = 'ha1.py'

#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0

ci = [[], []]
controller_path = None
controller_path_dir_path = None
