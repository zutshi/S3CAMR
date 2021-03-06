# REFERENCE:
# Monitoring Bounded LTL Properties Using Interval Analysis
# Daisuke Ishii1 Naoki Yonezaki2
# Tokyo Institute of Technology, Tokyo, Japan
# Alexandre Goldsztejn3
# CNRS, IRCCyN, Nantes, France


inf = float('inf')
plant_pvt_init_data = None

initial_set = [[2, 0, 0],
               [7, 0, 0]]

# initial_set = [[3, 0, 0],
#                [3, 0, 0]]

error_set = [[0,0,0],
             [0,0,0]]


true_until_5_x0_gt_2 = 'U[0,5] (TRUE, x0 > 2)'
true_until_10 = 'U[0,10] (TRUE, NOT({}))'.format(true_until_5_x0_gt_2)

ltl_property_1 = 'x0 > 2'
ltl_property_2 = 'U[0,5] (TRUE, x0 > 2)'
ltl_property_2 = 'NOT (U[0,5] (TRUE, x0 > 2))'
ltl_property_3 = 'U[0,10] (TRUE, NOT(U[0,5](TRUE), x0 > 2))'
ltl_property_4 = 'NOT(U[0,10] (TRUE, NOT(U[0,5](TRUE), x0 > 2)))'

grid_eps = [0.04, 0.04]

delta_t = 1
num_samples = 1

T = 1000

plant_description = 'python'
plant_path = 'bball_sin2.py'

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
MAX_ITER = 4
