# REFERENCE:
# EMSOFT paper

inf = float('inf')
plant_pvt_init_data = None


# Property
initial_set = [[4.0, 21.0, -1.0, -1.0],
                [5.0, 22.0, 1.0, 1.0]]

ROI = [[-1, -1, -5,-5],
       [26, 26, 5, 5]]


P = [[6., 7., -inf, -inf],
     [7., 8., inf, inf]]

error_set = P

T = 20.0

# Working set for P, Q
# ./scamr.py -f ../examples/nav/nav30.tst -cn  --refine model-dft --prop-check --incl-error --seed 0 --max-model-error 10 --max-paths 1000
#
# grid_eps = [1.1]*4
# delta_t = 5.0
# num_samples = 100
#
# grid_eps = [0.11]*4
# delta_t = 5.0
# num_samples = 2

# grid_eps = [0.1]*4
# delta_t = 5.0
# num_samples = 2


# EMSOFT'17
# grid_eps = [0.2]*4
# delta_t = 5.0
# num_samples = 10

# Fast (4min) but no vio found for 20 runs 
# grid_eps = [1.1]*4
# delta_t = 5.0
# num_samples = 10
#
#
# grid_eps = [1.1]*4
# delta_t = 5.0
# num_samples = 50

# tried for poly=2
#grid_eps = [1.0]*4
#delta_t = 5.0
#num_samples = 20


# Trials
grid_eps = [1]*4#[2, 2, 2, 2]#[1]*4
delta_t = 5.0
num_samples = 50

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
