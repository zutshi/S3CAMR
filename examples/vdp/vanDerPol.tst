inf = float('inf')
plant_pvt_init_data = None

initial_set = [[-0.4, -0.4], [0.4, 0.4]]
error_set = [[-1, -6.5], [-0.7, -5.6]]
#error_set = [[-inf, -inf], [-1, inf]]

grid_eps = [0.15, 0.15]
num_samples = 3
delta_t = 0.4

initial_discrete_state = []
initial_private_state = []

T = 1.0


MAX_ITER = 4


plant_description = 'python'
plant_path = 'vanDerPol.py'

#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0

ci = [[], []]
pi = [[],[]]
controller_path = None
controller_path_dir_path = None
