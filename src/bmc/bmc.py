# TODO: Freeze the arg list if they are deemed the same for every bmc
# engine


# def factory(bmc_engine_id,
#             vars,
#             pwa_model,
#             init_state,
#             safety_prop,
#             sys_name,
#             model_type):
def factory(bmc_engine,
            vs,
            pwa_model,
            init_state,
            final_states,
            sys_name,
            model_type,
            *args):
    if bmc_engine == 'sal':
        import sal_bmc.salbmc
        return sal_bmc.salbmc.BMC(
             vs,
             pwa_model,
             init_state,
             final_states,
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
             *args)
    else:
        raise NotImplementedError('Req. bmc engine: {}'.format(bmc_engine))


# TODO: fix and remove the hack
def convert_KPath2Relational(pwa_model):
    import modeling.pwa.relational as R
    assert(isinstance(pwa_model, R.PWARelational))
    new_pwa_model = R.PWARelational()
    for sm in pwa_model:
        assert(isinstance(sm, R.KPath))
        assert(len(sm.pnexts) >= 1)
        for p_ in sm.pnexts:
            new_sm = R.Relation(sm.p, p_, sm.m)
            new_pwa_model.add(new_sm)

    return new_pwa_model
