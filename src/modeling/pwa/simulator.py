
class SimError(Exception):
    pass

# class Simulator(object):
#     """Simulates pwa system"""

#     def __init__(self, sys):
#         self.sys = sys


def simulate(sys, x0, n):
    x = x0
    x_trace = [x0]
    t_trace = range(n)
    for i in range(n - 1):
        x_ = sys.predict(x)
        x_trace.append(x_)
        x = x_
    return t_trace, x_trace


# class PWA_SIM(PWA):

#     def __init__(self):
#         super(PWA_SIM, self).__init__()
#         return

#     # returns the first sub_model whose parition the point x belongs
#     # to
#     # TODO: Brute force search, very inefficient
#     def find_sub_model(self, x):
#         for sub_model in self.sub_models.itervalues():
#             if poly_sat(sub_model.p, x):
#                 return sub_model
#         raise SimError('no appropriate submodel found')

#     def predict(self, x):
#         try:
#             sub_model = self.find_sub_model(x)
#         except SimError:
#             return x

#         m = sub_model.m
#         return m.A * x + m.b


