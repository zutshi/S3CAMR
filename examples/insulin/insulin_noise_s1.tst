inf = float('inf')
plant_pvt_init_data = None

T = 720.0
delta_t = 0.5

#     HybridConfig initialConfig(initialSet, 2, 0, 0);
# Initial Mode
initial_discrete_state = [2]

#     Interval init_Gp = init_Gs;
#     init_Gp.mul_assign(1.9152);

# States
# idx           0  1      2       3       4           5    6    7       8       9    10 11
# name          X, Isc1,  Isc2,   Gt,     Gp,         Il,  Ip,  I1,     Id,     Gs,  t, uc
initial_set = [[0, 72.43, 141.15, 162.45, 120*1.9152, 3.2, 5.5, 100.25, 100.25, 120, 0, 50],
               [0, 72.43, 141.15, 162.45, 160*1.9152, 3.2, 5.5, 100.25, 100.25, 160, 0, 90]]
error_set = [[-inf]*12, [inf]*12]

# The blood glucose levels should never fall below 70 mg/dl. Levels below 70 mg/dl are
# called hypoglycemia, and may lead to loss of consciousness or coma.
# • The blood glucose levels should never rise above 300 mg/dl. Levels above 300 mg/dl
# expose the patient to a dangerous condition called ketoacidosis.
# • The blood glucose should be in the euglycemic range [70, 180] mg/dl during “wakeup”
# t ∈ [600, 720].

grid_eps = [0, 72.43, 141.15, 162.45, 120, 3.2, 5.5, 100.25, 100.25, 120, 0, 50]
pi[[-10], [10]]
grid_pi = [5]
num_samples = 2


MAX_ITER = inf

plant_description = 'python'
plant_path = 'insulin_noise_s1.py'

#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0
initial_private_state = []

ci = [[], []]
controller_path = None
controller_path_dir_path = None
