from modeling.pwa.pwagraph import convert_pwarel2pwagraph
from modeling.pwa import relational as rel

import utils as U


def pbyp(pwa_sys_prop):
    pwa_model = pwa_sys_prop.pwa_model
    init_partitions = pwa_sys_prop.init_partitions
    prop_partitions = pwa_sys_prop.final_partitions

    pwa_graph = convert_pwarel2pwagraph(pwa_model)

    sources = {p.ID for p in init_partitions}
    targets = {p.ID for p in prop_partitions}
    path_gen = pwa_graph.get_all_path_generator(sources, targets)

    # for each path, create a pwa model
    for path in path_gen:
        path_pwa_model = rel.PWARelational()

        for qi, qj in U.pairwise(path):
            pi, pj = pwa_graph.node_p(qi), pwa_graph.node_p(qj)
            mij = pwa_graph.edge_m((qi, qj))
            sub_model = rel.KPath(mij, pi, [pj], [])
            path_pwa_model.add(sub_model)

        # add the mandatory self loop
        #TODO: this needs to be fixed...why should it always be added?
        q0, qf = path[0], path[-1]
        p0 = pwa_graph.node_p(q0)
        pf = pwa_graph.node_p(qf)
        mff = pwa_graph.edge_m((qf, qf))
        terminal_sub_model = rel.KPath(mff, pf, [pf], [])
        path_pwa_model.add(terminal_sub_model)
        assert(qf == qj)

        init = {p0}
        final = {pf}
        yield path_pwa_model, init, final

    return
