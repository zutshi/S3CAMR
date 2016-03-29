inf = float('inf')
plant_pvt_init_data = None

# Property
# 0<=x0<=1 and 1<=x1<=2
initial_set = [[0, 1], [1, 2]]
# 2.3<=x0<=2.4 and 1.2<=x1<=1.3
error_set = [[2.3, 1.2], [2.4, 1.3]]
T = 10.0

grid_eps = [1, 1]
delta_t = 1
num_samples = 10


MAX_ITER = 2


plant_description = 'python'
plant_path = 'brusselator.py'

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
