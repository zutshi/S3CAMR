from sklearn import linear_model as skl_lm
import numpy as np

import constraints as cons


# TODO: UNUSED
def factory(model_type):
    if model_type == 'approx':
        return RegressionModel()
    elif model_type == 'sound':
        return FlowStarModel()
    else:
        raise NotImplementedError


class RegressionModel(object):
    def __init__(self, x, y):
        clf_ = skl_lm.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
        self.clf_ = clf_
        clf_.fit(x, y)
        #if __debug__:
        #    print clf_.coef_, clf_.intercept_

    @property
    def clf(self):
        return self.clf_

    @property
    def Ab(self):
        return self.clf.coef_, self.clf.intercept_

    def predict(self, X):
        Y = self.clf.predict(X)
        return Y

    def error_pc(self, X, Y):
        """error_pc

        Parameters
        ----------
        X : Test input
        Y : Test Output

        Returns
        -------
        error % vector

        Notes
        ------
        Indicates model quality for debug purposes
        """
        '''computes relative error% along each dimension'''
        Y_ = self.predict(X)
        #if __debug__:
        #    print 'score: ', self.clf.score(X, Y)
        #    print abs((Y - Y_)/Y)*100
        return np.max((abs((Y - Y_)/Y))*100, axis=0)

    def error(self, X, Y):
        """error

        Parameters
        ----------
        X : Test input
        Y : Test Output

        Returns
        -------
        Computes signed error against passed in test samples.
        Sound error interval vector (non-symmeteric).
        The interal must be added and NOT subtracted to make a sound
        model. e.g. Y_sound = A*X + e
        or,
            predict(X) + e_l <= Y_sound <= predict(X) + e_h
            Y_sound = predict(X) + [e_l, e_h]

        Notes
        ------
        Computes interval vector e = Y_true - Y_predict, such that
        the interval is sound w.r.t. to passed in test samples
        """
        Yp = self.predict(X)
        delta = Y - Yp
        max_e = np.max((delta), axis=0)
        min_e = np.min((delta), axis=0)
        return cons.IntervalCons(min_e, max_e)


class FlowStarModel(object):
    """FlowStarModel
    Implements only purely continuous models for now.
    Can add support for H.A. later."""

    def __init__(self, ODE_str):
        self.ODE = ODE_str
        return

    def get_model(self, X, t):
        return self.get_reach_set(X, t)

    def get_reach_set(self, X, t):
        """interface with flow*"""
        raise NotImplementedError
