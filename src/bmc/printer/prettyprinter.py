from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import blessings

import fileops as fops

TERM = blessings.Terminal()


def pretty_print(sys, prop, vs, pwa_model, init_cons, final_cons,
                 init_ps, final_ps, fname_constructor, sys_name,
                 model_type, *args):
    sep = '-'*20
    s = []

    s += ['system: {}'.format(sys)]
    s += ['type: {}'.format(type(sys))]
    s += [sep]

    s += ['prop: {}'.format(prop)]
    s += ['type: {}'.format(type(prop))]
    s += [sep]

    s += ['variables: {}'.format(vs)]
    s += ['type: {} of {}'.format(type(vs), type(list(vs)[0]))]
    s += [sep]

    s += ['pwa-model: {}'.format(pwa_model)]
    s += ['type: {}'.format(type(pwa_model))]
    s += [sep]

    s += ['constraints on the INITIAL states: {}'.format(init_cons)]
    s += ['type: {}'.format(type(init_cons))]
    s += [sep]

    s += ['constraints on the ERROR/FINAL states: {}'.format(final_cons)]
    s += ['type: {}'.format(type(final_cons))]
    s += [sep]

    s += ['Initial locations: {}'.format(init_ps)]
    s += ['type: {} of {}'.format(type(init_ps), type(list(init_ps)[0]))]
    s += [sep]

    s += ['Final locations:{}'.format(final_ps)]
    s += ['type:{} of {}'.format(type(final_ps),  type(list(final_ps)[0]))]
    s += [sep]

    s += ['convinience function for constructing a filename by prepending the path of the directory and the name of the fname_constructortem and time stamp to a string: {}'.format(fname_constructor)]
    s += ['usage: fname_constructor("example_dump.log") := {}'.format(fname_constructor('example_dump.log'))]
    s += [sep]

    s += ['name of the sys_nametem: {}'.format(sys_name)]
    s += ['type: {}'.format(type(sys_name))]
    s += [sep]

    s += ['type of pwa_model: {}'.format(model_type)]
    s += ['type: {}'.format(type(model_type))]
    s += [sep]

    s += pretty_print_pwa_model(pwa_model)

    fname = fname_constructor('pretty_print_bmc_args')
    fops.write_data(fname, '\n'.join(s))

    print(TERM.green('pretty printing output: {}'.format(fname)))
    print('exiting...')
    exit(0)
    return


def pretty_print_pwa_model(pwa_model):
    s = []

    nodes = pwa_model.nodes()
    edges = pwa_model.all_edges()

    s += ['========== pwa_model: A graph of type: {} =========='.format(type(pwa_model))]
    s += ['Nodes']
    s += ['pwa_model has the below nodes of type: {}'.format(type(list(nodes)[0]))]
    s += ['\n'.join(str(n) for n in nodes)]

    q0 = nodes[0]
    e0 = edges[0]
    s += ['Each node has a location: {} of type: {}'
          'and each edge has an affine map: {} of type: {}'
          'associated with it'.format(pwa_model.node_p(q0),
                                      type(pwa_model.node_p(q0)),
                                      pwa_model.edge_m(e0),
                                      type(pwa_model.edge_m(e0)))]

    s += ['pwa_model has the below edges of type: {}'.format(type(list(edges)[0]))]
    s += ['\n'.join(str(e) for e in edges)]

    return s
