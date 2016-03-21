inf = float('inf')
plant_pvt_init_data = None

initial_set = [[-0.4, -0.4], [0.4, 0.4]]
error_set = [[-1, -6.5], [-0.7, -5.6]]

################
# what worked
################
# NOTES:
#   grid_eps <= [0.04, 0.04] for model_dft to succeed...why?

# param 1
grid_eps = [0.04, 0.04]
delta_t = 0.1
num_samples = 5

# param 2
#grid_eps <= [0.01, 0.01]
#num_samples = 1

# param 3
#grid_eps <= [0.05, 0.05]
#num_samples = 10


# param 4
# This does not work even after splitting because the underlying paths
# are very coarse?
#grid_eps = [0.1, 0.1]
#delta_t = 0.1
#num_samples = 10

######################
# what did not worked
######################

# grid_eps = [0.50, 0.04]
# delta_t = 0.4
# num_samples = 2

# grid_eps = [0.1, 0.1]
# num_samples = 3
# delta_t = 0.4

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
