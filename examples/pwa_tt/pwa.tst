########################
# PWA TT dynamical system
# The change in the dynamics is triggered by time.
########################


inf = float('inf')
plant_pvt_init_data = None

initial_set = [[0, 0, 0], [1, 1, 0]]
error_set = [[1., 48.5, 0], [4., 49.5, 8]] #


# Worked
# ./scamr.py -f ../examples/pwa/pwa.tst -cn  --refine model-dft --seed 3 --max-model-error 10 --prop-check --bmc-engine sal --incl-error --pvt-init-data 1 -pmp --plots x0-x1

grid_eps = [1., 1., 0.05]
num_samples = 40
# confirmed success for delta_t = .5, but 1 also seems to work
delta_t = 1

# Did not work.....why? Either the wrong path comes up: which might be
# a specific case. And exploring more discrete paths might the be the
# way to go. But more bothersome is the scenario, when the bmc says :
# no path found.
# num_samples = 10, 20, ....

# Works for num_samples = 40


T = 3.5

plant_description = 'python'
plant_path = 'pwa.py'


ci = [[], []]
pi = [[],[]]

#############################################
initial_controller_integer_state = []
initial_controller_float_state = []
num_control_inputs = 0
min_smt_sample_dist = 0
controller_path = None
controller_path_dir_path = None
initial_discrete_state = []
initial_private_state = []
MAX_ITER = 4

