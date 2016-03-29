#inf = float('inf')
plant_pvt_init_data = None

initial_set = [[5,0],[7,2]]
error_set = [[0,0],[0,0]]
grid_eps = [0.04, 0.04]
delta_t = 0.5
num_samples = 1

T = 20.0

plant_description = 'python'
plant_path = 'bball_2dim.py'

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
