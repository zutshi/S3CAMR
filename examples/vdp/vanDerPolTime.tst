inf = float('inf')
plant_pvt_init_data = None

# xi \in [-0.4, 0.4], t = 0.0
initial_set = [[-0.4, -0.4, 0.0], [0.4, 0.4, 0.0]]

prop1 = [[-1.0, -6.5, 2.0], [-0.7, -5.6, 4.5]]
porp2 = [[-1.0, -6.5, 2.0], [-0.7, -5.6, 2.5]]
prop3 = [[-1.0, -6.5, 2.5], [-0.7, -5.6, 4.5]]

T = 5.0

error_set = prop1
# EMSOFT params
grid_eps = [.51, .51, .51]
# num_samples = 10
# delta_t = 0.5

# Fruther experimentation required
num_samples = 50
delta_t = 2

# works with refine init at times, and gives somewhat
# acceptable results with dft, but none with rel. Need to experiment
# more
# delta_t = 5
# num_samples = 100

# Not tried yet
# error_set = prop2
# Not tried yet
# error_set = prop3

plant_description = 'python'
plant_path = 'vanDerPolTime.py'

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


MAX_ITER = inf
