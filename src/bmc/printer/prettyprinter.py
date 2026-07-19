import blessed

import fileops as fops

TERM = blessed.Terminal()


def pretty_print(sys, prop, vs, pwa_model, init_cons, final_cons,
                 init_ps, final_ps, fname_constructor, sys_name,
                 model_type, *args):
    sep = '-'*20
    s = []

    s += [f'system: {sys}']
    s += [f'type: {type(sys)}']
    s += [sep]

    s += [f'prop: {prop}']
    s += [f'type: {type(prop)}']
    s += [sep]

    s += [f'variables: {vs}']
    s += [f'type: {type(vs)} of {type(list(vs)[0])}']
    s += [sep]

    s += [f'pwa-model: {pwa_model}']
    s += [f'type: {type(pwa_model)}']
    s += [sep]

    s += [f'constraints on the INITIAL states: {init_cons}']
    s += [f'type: {type(init_cons)}']
    s += [sep]

    s += [f'constraints on the ERROR/FINAL states: {final_cons}']
    s += [f'type: {type(final_cons)}']
    s += [sep]

    s += [f'Initial locations: {init_ps}']
    s += [f'type: {type(init_ps)} of {type(list(init_ps)[0])}']
    s += [sep]

    s += [f'Final locations:{final_ps}']
    s += [f'type:{type(final_ps)} of {type(list(final_ps)[0])}']
    s += [sep]

    s += [f'convinience function for constructing a filename by prepending the path of the directory and the name of the fname_constructortem and time stamp to a string: {fname_constructor}']
    s += ['usage: fname_constructor("example_dump.log") := {}'.format(fname_constructor('example_dump.log'))]
    s += [sep]

    s += [f'name of the sys_nametem: {sys_name}']
    s += [f'type: {type(sys_name)}']
    s += [sep]

    s += [f'type of pwa_model: {model_type}']
    s += [f'type: {type(model_type)}']
    s += [sep]

    s += pretty_print_pwa_model(pwa_model)

    fname = fname_constructor('pretty_print_bmc_args')
    fops.write_data(fname, '\n'.join(s))

    print(TERM.green(f'pretty printing output: {fname}'))
    print('exiting...')
    exit(0)
    return


def pretty_print_pwa_model(pwa_model):
    s = []

    nodes = pwa_model.nodes()
    edges = pwa_model.all_edges()

    s += [f'========== pwa_model: A graph of type: {type(pwa_model)} ==========']
    s += ['Nodes']
    s += [f'pwa_model has the below nodes of type: {type(list(nodes)[0])}']
    s += ['\n'.join(str(n) for n in nodes)]

    q0 = list(nodes)[0]
    e0 = list(edges)[0]
    s += ['Each node has a location: {} of type: {}'
          'and each edge has an affine map: {} of type: {}'
          'associated with it'.format(pwa_model.node_p(q0),
                                      type(pwa_model.node_p(q0)),
                                      pwa_model.edge_m(e0),
                                      type(pwa_model.edge_m(e0)))]

    s += [f'pwa_model has the below edges of type: {type(list(edges)[0])}']
    s += ['\n'.join(str(e) for e in edges)]

    return s
