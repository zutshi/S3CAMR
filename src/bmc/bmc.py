from sal_bmc import salbmc


def factory(bmc_engine='sal'):
    if bmc_engine == 'sal':
        #import salbmc as sbmc
        return SALBMC(salbmc)
    else:
        raise NotImplementedError


class SALBMC(object):
    def __init__(self, sbmc):
        self.sbmc = sbmc

    def init(self, nd, pwa_model, init_state, safety_prop, sys_name, model_type):
        self.sbmc = self.sbmc.BMC(nd, pwa_model, init_state, safety_prop, sys_name, model_type)
        return

    def check(self, depth):
        self.sbmc.dump()
        self.sbmc.check(depth=depth)
        return
