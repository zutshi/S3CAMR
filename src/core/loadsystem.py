###############################################################################
# File name: loadsystem.py
# Author: Aditya
# Python Version: 2.7
#
#                       #### Description ####
# Provides a parse function and structures to load a parsed system.
# Reads the system description from the provided .tst file,
# and populates respective structures and returns them.
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import sys as SYS
import imp

from .numdims import NumDims
from . import psim

import utils as U
from utils import print
import fileops as fp
import constraints as cns


class MissingSystemDefError(Exception):
    def __init__(self, args):
        missing_attr = args[0].replace(''''module' object has no attribute ''', '')
        error_msg = 'Missing definition in the .tst file: {}'.format(missing_attr)
        self.args = (error_msg,)
        return


# For efficiency, Consider named tuples instead? or something else?
class System(object):
    def __init__(self, sys_name, controller_path, num_dims,
                 plant_config_dict, delta_t, controller_path_dir_path,
                 controller_object_str, path, plant_pvt_init_data,
                 min_smt_sample_dist, ci_grid_eps, pi_grid_eps):

        self.sys_name = sys_name
        self.controller_path = controller_path
        self.controller_object_str = controller_object_str
        self.num_dims = num_dims
        self.plant_pvt_init_data = plant_pvt_init_data
        # split plant_config into sys,opt, and props. Also, cleanup names and
        # make them uniform with controller names
        self.plant_config_dict = plant_config_dict

        self.delta_t = delta_t
        self.controller_path_dir_path = controller_path_dir_path
        self.path = path

        # TODO: sampling_dist along with abstraction params need to be
        # separated and the plat abs dict needs to go!
        self.min_smt_sample_dist = min_smt_sample_dist

        self.ci_grid_eps = np.array(ci_grid_eps, dtype=float)
        self.pi_grid_eps = np.array(pi_grid_eps, dtype=float)

        self.sanity_check = U.assert_no_Nones

        return

    def init_sims(self, plt_lib, property_checker, psim_args):
        self.plant_sim = psim.simulator_factory(
            self.plant_config_dict,
            self.path,
            property_checker,
            plt=plt_lib,
            plant_pvt_init_data=self.plant_pvt_init_data,
            parallel=False,
            sim_args=psim_args)# TAG:MSH
        self.controller_sim = None


class Property(object):
    def __init__(self, T, init_cons_list, init_cons, final_cons, ci, pi,
                 initial_discrete_state, initial_controller_state, MAX_ITER,
                 num_segments, ROI):
        self.T = T
        self.init_cons_list = init_cons_list
        self.init_cons = init_cons
        self.final_cons = final_cons
        self.ci = ci
        self.pi = pi
        self.initial_discrete_state = initial_discrete_state
        self.initial_controller_state = initial_controller_state
        self.MAX_ITER = MAX_ITER
        self.num_segments = num_segments
        # Region of interest
        self.ROI = ROI

        self.sanity_check = U.assert_no_Nones
        #print init_cons.to_numpy_array()
        #print final_cons.to_numpy_array()
        #print ci.to_numpy_array()

        return


# No longer being used
# Instead, all options are now passed using the commandline
class Options(object):
    def __init__(self, plot, MODE, num_sim_samples, METHOD, symbolic_analyzer):
        self.plot = plot
        self.MODE = MODE
        self.num_sim_samples = num_sim_samples
        self.METHOD = METHOD
        self.symbolic_analyzer = symbolic_analyzer

        self.sanity_check = U.assert_no_Nones

        return


def parse(file_path, plant_pvt_init_data):
    test_des_file = file_path
    path = fp.get_abs_base_path(file_path)
    sys_name = fp.split_filename_ext(
                fp.get_file_name_from_path(
                    file_path)
                )[0]

    test_dir = fp.get_abs_base_path(test_des_file)
    SYS.path.append(test_dir)

    try:
        sut = imp.load_source('test_des', test_des_file)
    except IOError as e:
        print('File not found: {}'.format(test_des_file))
        raise e

    sut.plant_pvt_init_data = plant_pvt_init_data

    try:
        assert(len(sut.initial_set) == 2)
        assert(len(sut.error_set) == 2)
        assert(len(sut.ci) == 2)
        assert(len(sut.pi) == 2)

        assert(len(sut.ci[0]) == len(sut.ci[1]))
        assert(len(sut.pi[0]) == len(sut.pi[1]))

        assert(len(sut.initial_set[0]) == len(sut.initial_set[1]))
        assert(len(sut.error_set[0]) == len(sut.error_set[1]))

        assert(len(sut.initial_set[0]) == len(sut.error_set[0]))

        assert(len(sut.initial_set[0]) == len(sut.grid_eps))

        num_dims = NumDims(
            si=len(sut.initial_controller_integer_state),
            sf=len(sut.initial_controller_float_state),
            s=len(sut.initial_controller_integer_state)+len(sut.initial_controller_float_state),
            x=len(sut.initial_set[0]),
            u=sut.num_control_inputs,
            ci=len(sut.ci[0]),
            pi=len(sut.pi[0]),
            d=len(sut.initial_discrete_state),
            pvt=0,
            )

        if not hasattr(sut, 'ci_grid_eps') or len(sut.ci_grid_eps) == 0:
            sut.ci_grid_eps = []

#         if hasattr(sut, 'ci_grid_eps'):
#             if len(sut.ci_grid_eps) == 0:
#                 sut.ci_grid_eps = None
#         else:
#             sut.ci_grid_eps = None

        if not hasattr(sut, 'pi_grid_eps') or len(sut.pi_grid_eps) == 0:
            sut.pi_grid_eps = []

#         if hasattr(sut, 'pi_grid_eps'):
#             if len(sut.pi_grid_eps) == 0:
#                 sut.pi_grid_eps = None
#         else:
#             sut.pi_grid_eps = None

        if num_dims.ci == 0:
            ci = None
        else:
            ci = cns.IntervalCons(np.array(sut.ci[0]), np.array(sut.ci[1]))

        if num_dims.pi == 0:
            pi = None
        else:
            pi = cns.IntervalCons(np.array(sut.pi[0]), np.array(sut.pi[1]))

        assert((pi is None and sut.pi_grid_eps == []) or
               (pi is not None and sut.pi_grid_eps != []))
        assert((ci is None and sut.ci_grid_eps == []) or
               (ci is not None and sut.ci_grid_eps != []))

        if pi is not None:
            assert(len(sut.pi[0]) == len(sut.pi_grid_eps))

        if ci is not None:
            assert(len(sut.ci[0]) == len(sut.ci_grid_eps))

        plant_config_dict = {'plant_description': sut.plant_description,
                             'plant_path': sut.plant_path,
                             #'refinement_factor': sut.refinement_factor,
                             'eps': np.array(sut.grid_eps, dtype=float),
                             #'pi_eps': np.array(sut.pi_grid_eps, dtype=float),
                             'num_samples': int(sut.num_samples),
                             'delta_t': float(sut.delta_t),
                             #'type': 'string',
                             'type': 'value',
                             }
        #print(plant_config_dict)
        #for k, d in plant_config_dict.iteritems():
        #    plant_config_dict[k] = str(d)

        # TODO: fix the list hack, once the controller ifc matures

        init_cons = cns.IntervalCons(np.array(sut.initial_set[0]),
                                     np.array(sut.initial_set[1]))

        init_cons_list = [init_cons]

        final_cons = cns.IntervalCons(np.array(sut.error_set[0]),
                                      np.array(sut.error_set[1]))
        T = sut.T
        delta_t = sut.delta_t

        if sut.controller_path is not None:
            controller_object_str = fp.split_filename_ext(sut.controller_path)[0]
            controller_path_dir_path = fp.construct_path(sut.controller_path_dir_path, path)
        else:
            controller_object_str = None
            controller_path_dir_path = None

        initial_discrete_state = sut.initial_discrete_state
        initial_controller_state = (sut.initial_controller_integer_state +
                                    sut.initial_controller_float_state)
        MAX_ITER = sut.MAX_ITER

        num_segments = int(np.ceil(T / delta_t))

        controller_path = sut.controller_path
        plant_pvt_init_data = sut.plant_pvt_init_data

        min_smt_sample_dist = sut.min_smt_sample_dist
        ci_grid_eps = sut.ci_grid_eps
        pi_grid_eps = sut.pi_grid_eps

        #num_sim_samples = sut.num_sim_samples
        #METHOD = sut.METHOD
        #symbolic_analyzer = sut.symbolic_analyzer
        #plot = sut.plot
        #MODE = sut.MODE
        #opts = Options(plot, MODE, num_sim_samples, METHOD, symbolic_analyzer)
    except AttributeError as e:
        raise MissingSystemDefError(e)

    # ROI is not mandatory and defaults to xi \in (-inf, inf)
    try:
        ROI = cns.IntervalCons(*sut.ROI)
        assert(ROI.dim == num_dims.x)
    except AttributeError:
        ROI = cns.top2ic(num_dims.x)

    sys = System(sys_name, controller_path, num_dims, plant_config_dict,
                 delta_t, controller_path_dir_path,
                 controller_object_str, path, plant_pvt_init_data,
                 min_smt_sample_dist, ci_grid_eps,
                 pi_grid_eps)
    prop = Property(T, init_cons_list, init_cons, final_cons, ci,
                    pi, initial_discrete_state, initial_controller_state,
                    MAX_ITER, num_segments, ROI)

    print('system loaded...')
    return sys, prop


# # older function: did not have any sanity checking
# def parse_without_sanity_check(file_path):
#     test_des_file = file_path
#     path = fp.get_abs_base_path(file_path)

#     test_dir = fp.get_abs_base_path(test_des_file)
#     SYS.path.append(test_dir)

#     sut = imp.load_source('test_des', test_des_file)

#     plant_config_dict = {'plant_description': sut.plant_description,
#                          'plant_path': sut.plant_path,
#                          'refinement_factor': sut.refinement_factor,
#                          'grid_eps': sut.grid_eps,
#                          'num_samples': sut.num_samples,
#                          'delta_t': sut.delta_t,
#                          'type': 'string',
#                          }

#     for k, d in plant_config_dict.iteritems():
#         plant_config_dict[k] = str(d)

#     num_dims = NumDims(
#         si=len(sut.initial_controller_integer_state),
#         sf=len(sut.initial_controller_float_state),
#         s=len(sut.initial_controller_integer_state)+len(sut.initial_controller_float_state),
#         x=len(sut.initial_set[0]),
#         u=sut.num_control_inputs,
#         ci=len(sut.ci[0]),
#         pi=0,
#         d=len(sut.initial_discrete_state),
#         pvt=0,
#         )

#     if num_dims.ci == 0:
#         ci = None
#     else:

#         ci = cns.IntervalCons(np.array(sut.ci[0]), np.array(sut.ci[1]))
#     if num_dims.pi == 0:
#         pi = None
#     else:
#         pi = cns.IntervalCons(np.array(sut.pi[0]), np.array(sut.pi[1]))

#     # TODO: fix the list hack, once the controller ifc matures

#     init_cons = cns.IntervalCons(np.array(sut.initial_set[0]),
#                                  np.array(sut.initial_set[1]))

#     init_cons_list = [init_cons]

#     final_cons = cns.IntervalCons(np.array(sut.error_set[0]),
#                                   np.array(sut.error_set[1]))
#     T = sut.T

#     if sut.controller_path is not None:
#         controller_object_str = fp.split_filename_ext(sut.controller_path)[0]
#         controller_path_dir_path = fp.construct_path(sut.controller_path_dir_path, path)
#     else:
#         controller_object_str = None
#         controller_path_dir_path = None

#     delta_t = sut.delta_t
#     #try:  # enforce it for now!
#     #except:
#     #    controller_path_dir_path = None
#     initial_discrete_state = sut.initial_discrete_state
#     initial_controller_state = sut.initial_controller_integer_state \
#                              + sut.initial_controller_float_state
#     MAX_ITER = sut.MAX_ITER
#     #try:  # enforce it for now!
#     #except:
#     #    None
#     num_segments = int(np.ceil(T / delta_t))

#     controller_path = sut.controller_path
#     plant_pvt_init_data = sut.plant_pvt_init_data

#     sys = System(controller_path, num_dims, plant_config_dict, delta_t, controller_path_dir_path, controller_object_str, path, plant_pvt_init_data, sut.min_smt_sample_dist)
#     prop = Property(T, init_cons_list, init_cons, final_cons, ci, pi, initial_discrete_state, initial_controller_state, MAX_ITER, num_segments)

#     #num_sim_samples = sut.num_sim_samples
#     #METHOD = sut.METHOD
#     #symbolic_analyzer = sut.symbolic_analyzer
#     #plot = sut.plot
#     #MODE = sut.MODE
#     #opts = Options(plot, MODE, num_sim_samples, METHOD, symbolic_analyzer)
#     return sys, prop
