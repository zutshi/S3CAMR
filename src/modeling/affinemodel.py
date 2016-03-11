from sklearn import linear_model as skl_lm
import numpy as np


class AffineModel(object):
    def __init__(self, x, y):
        clf_ = skl_lm.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
        self.clf_ = clf_
        clf_.fit(x, y)
        if __debug__:
            print clf_.coef_, clf_.intercept_

    @property
    def clf(self):
        return self.clf_

    @property
    def Ab(self):
        return self.clf.coef_, self.clf.intercept_

    def predict(self, X):
        Y = self.clf.predict(X)
        return Y

    def model_error(self, X, Y):
        '''computes relative error along each dimension'''
        Y_ = self.predict(X)
        if __debug__:
            print 'score: ', self.clf.score(X, Y)
            print abs((Y - Y_)/Y)*100
        return np.max((abs((Y - Y_)/Y))*100, axis=0)
