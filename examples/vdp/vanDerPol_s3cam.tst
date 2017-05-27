inf = float('inf')
plant_pvt_init_data = None

initial_set = [[-0.4, -0.4], [0.4, 0.4]]
error_set = [[-1, -6.5], [-0.7, -5.6]]


# comes from HyCU
# grid_eps = [0.1, 0.1]
# num_samples = 3
# delta_t = 0.4


# # param 5: works for S3CAM
grid_eps = [0.11, 0.11]
delta_t = 0.1
num_samples = 3



initial_discrete_state = []
initial_private_state = []

T = 1.0


MAX_ITER = inf


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
