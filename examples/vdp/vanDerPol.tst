inf = float('inf')
plant_pvt_init_data = None

initial_set = [[-0.4, -0.4], [0.4, 0.4]]
error_set = [[-1, -6.5], [-0.7, -5.6]]

################
# what worked
################
# NOTES:
#   grid_eps <= [0.04, 0.04] for model_dft to succeed...why?

# param 1: works for S3CAMR
# grid_eps = [0.04, 0.04]
# delta_t = 0.1
# num_samples = 5

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


# param 5: works for S3CAM
# grid_eps = [0.1, 0.1]
# delta_t = 0.1
# num_samples = 3


# # Must have prop check ON
# grid_eps = [.21, .21]
# num_samples = 10
# delta_t = 0.5


# Gets a CE but its very in-accurate
# python -O ./scamr.py -f ../examples/vdp/vanDerPol.tst -cn --refine model_dft --seed 0 -pmp --prop-check --incl-error --max-model-error 10000
grid_eps = [.51, .51]
num_samples = 50
delta_t = 0.2

# Gets a reproducible CE
# python -O ./scamr.py -f ../examples/vdp/vanDerPol.tst -cn --refine model_dft --seed 0 -pmp --prop-check --incl-error --max-model-error 10000
# grid_eps = [.011, .011]
# num_samples = 2
# delta_t = 0.5



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
