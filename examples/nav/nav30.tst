# REFERENCE:
# EMSOFT paper

inf = float('inf')
plant_pvt_init_data = None


# x0 = np.array([4, 5], [21, 22])
# v0 = np.array([[0.2, 0.5], [-0.5, 0.5]])

# Property
initial_set = [[4.0, 21.0, -1.0, -1.0],
               [5.0, 22.0, 1.0, 1.0]]

#initial_set = [[5.,          21.22936519,  -0.959,   0.],
#               [5.,          21.22936519,  -0.959,   0.078]]

P = [[6., 7., -inf, -inf],
     [7., 8., inf, inf]]

Q = [[7., 9., -inf, -inf],
     [8., 10., inf, inf]]

R = [[1., 6., -inf, -inf],
     [2., 7., inf, inf]]

S = [[22., 11., -inf, -inf],
     [23., 12., inf, inf]]

error_set = P

T = 20.0

# Abstraction params: P, Q, R, S
#grid_eps = [0.2, 0.2, 0.4, 0.4]
grid_eps = [0.51]*4
delta_t = 5.0
num_samples = 5

MAX_ITER = 4

plant_description = 'python'
plant_path = 'nav30.py'

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
