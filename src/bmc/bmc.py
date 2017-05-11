# TODO: Freeze the arg list if they are deemed the same for every bmc
# engine


# def factory(bmc_engine_id,
#             num_state_dims,
#             pwa_model,
#             init_state,
#             safety_prop,
#             sys_name,
#             model_type):
def factory(bmc_engine, *args):
    if bmc_engine == 'sal':
        import sal_bmc.salbmc
        return sal_bmc.salbmc.BMC(*args)
    elif bmc_engine == 's3camsmt':
        from S3CAMSMT.s3camsmt.bmc.bmc_pwa import import BMC_PWA
        return BMC_PWA(*args)
    else:
        raise NotImplementedError
