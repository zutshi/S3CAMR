# TODO: Freeze the arg list if they are deemed the same for every bmc
# engine
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import settings
from globalopts import opts as gopts


# def factory(bmc_engine_id,
#             vars,
#             pwa_model,
#             init_state,
#             safety_prop,
#             sys_name,
#             model_type):
def factory(bmc_engine,
            sys,
            prop,
            vs,
            pwa_model,
            init_state,
            final_states,
            init_partitions,
            prop_partitions,
            fname_constructor,
            sys_name,
            model_type,
            *args):
    """factory

    Parameters
    ----------
    bmc_engine :
    vs :
    pwa_model :
    init_state :
    final_states :
    sys_name :
    model_type :
    prec : number of digits after the decimal
    *args :

    Returns
    -------

    Notes
    ------
    """

    # remove prop_partitions
    assert(settings.CE)

    if bmc_engine == 'sal':
        from .sal_bmc import salbmc
        return salbmc.BMC(
             vs,
             pwa_model,
             init_state,
             final_states,
             prop_partitions,
             fname_constructor,
             sys_name,
             model_type,
             *args)
    elif bmc_engine == 's3camsmt':
        from S3CAMSMT.s3camsmt.bmc.bmc_pwa import BMC_PWA
        if model_type == 'rel':
            raise NotImplementedError
        # TODO: fix and remove the hack
        else:# model_type == 'dft':
            model_type = 'rel'
            pwa_model = convert_KPath2Relational(pwa_model)
            #exit()
        return BMC_PWA(
             vs,
             pwa_model,
             init_state,
             final_states,
             sys_name,
             model_type,
             gopts.bmc_prec,
             *args)
    elif bmc_engine == 'pwa':
        from .pwa_bmc import linprogbmc
        from pwa.pwagraph import convert_pwarel2pwagraph

        return linprogbmc.BMC(
             sys,
             prop,
             vs,
             convert_pwarel2pwagraph(pwa_model),
             init_state,
             final_states,
             init_partitions,
             prop_partitions,
             fname_constructor,
             sys_name,
             model_type,
             *args)
    else:
        raise NotImplementedError('Req. bmc engine: {}'.format(bmc_engine))


# TODO: fix and remove the hack
# ignores the constraints on pnexts
def convert_KPath2Relational(pwa_model):
    import pwa.relational as R
    assert(isinstance(pwa_model, R.PWARelational))
    new_pwa_model = R.PWARelational()
    for sm in pwa_model:
        assert(isinstance(sm, R.KPath))
        assert(len(sm.pnexts) >= 1)
        for p_ in sm.pnexts:
            new_sm = R.Relation(sm.p, p_, sm.m)
            new_pwa_model.add(new_sm)

    return new_pwa_model
