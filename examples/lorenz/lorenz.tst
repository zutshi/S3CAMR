inf = float('inf')
plant_pvt_init_data = None

# Property
# -2<=xi<=2 and t=0
initial_set = [[-2, -2, -2, 0], [2, 2, 2, 0.0001]]
# 2.5<=x0<=3.0 and 0<=x1<=4 and t<=10
error_set = [[2.5, 0, -inf, 0], [3, 4, inf, 10]]
T = 10.0

grid_eps = [0.5]*4
delta_t = 5
num_samples = 2


MAX_ITER = 5


plant_description = 'python'
plant_path = 'lorenz.py'

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
