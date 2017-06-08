inf = float('inf')
plant_pvt_init_data = None

T = 4
delta_t = 0.01

dl = 0.05

initial_set = [[0-dl, 0-dl, 0-dl, 22-dl, 0-dl , 0-dl, -2.1854-dl, 0-dl],
               [0+dl, 0+dl, 0+dl, 22+dl, 0+dl , 0+dl, -2.1854+dl, 0+dl]]

#3.3 might be reachable, 3.6 probably not
error_set = [[29, 3.6, -inf, -inf, -inf, -inf, -inf, -inf],
             [34, 4, inf, inf, 70, 8, inf, inf]]
# error_set = [[-inf, -inf, -inf, -inf, 60, 7, -inf, -inf],
#              [inf, inf, inf, inf, 70, 8, inf, inf]]

grid_eps = [1]*8
num_samples = 100
pi = [[-dl]*26,[dl]*26]
pi_grid_eps = [1]*26
# pi = [[
#    10.0000 - dl,
#     9.1641 - dl,
#    -0.0000 - dl,
#    10.1048 - dl,
#    -0.0000 - dl,
#          0 - dl,
#     4.6054 - dl,
#          0 - dl,
#    -0.1350 - dl,
#     0.1474 - dl,
#     3.1278 - dl,
#     0.0000 - dl,
#     0.6186 - dl,
#     2.2056 - dl,
#     0.0000 - dl,
#    -1.5035 - dl,
#          0 - dl,
#          0 - dl,
#          0 - dl,
#    22.0000 - dl,
#          0 - dl,
#          0 - dl,
#    -2.1854 - dl,
#          0 - dl,
#          0 - dl,
#    -0.0000 - dl,
# ],
# [

#    10.0000 + dl,
#     9.1641 + dl,
#    -0.0000 + dl,
#    10.1048 + dl,
#    -0.0000 + dl,
#          0 + dl,
#     4.6054 + dl,
#          0 + dl,
#    -0.1350 + dl,
#     0.1474 + dl,
#     3.1278 + dl,
#     0.0000 + dl,
#     0.6186 + dl,
#     2.2056 + dl,
#     0.0000 + dl,
#    -1.5035 + dl,
#          0 + dl,
#          0 + dl,
#          0 + dl,
#    22.0000 + dl,
#          0 + dl,
#          0 + dl,
#    -2.1854 + dl,
#          0 + dl,
#          0 + dl,
#    -0.0000 + dl,
# ]]

MAX_ITER = inf


plant_description = 'python'
plant_path = 'bicycle.py'

#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0

ci = [[], []]
controller_path = None
controller_path_dir_path = None
initial_discrete_state = [0]
initial_private_state = []
