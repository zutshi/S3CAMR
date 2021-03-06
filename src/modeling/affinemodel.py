from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging
import abc
import six

from sklearn import linear_model as skl_lm
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

from IPython import embed

import fileops as fops
import constraints as cons
import err
from utils import print
import utils as U

from polynomial.poly import power_array2Poly
from globalopts import opts as gopts
import settings

logger = logging.getLogger(__name__)


def Model(*args, **kwargs):
    if gopts.model_type == 'affine':
        return OLS(*args, **kwargs)
    elif gopts.model_type == 'poly':
        return Poly(*args, **kwargs)
    else:
        raise NotImplementedError
    return


def is_undetermined(x):
    num_samples, ndim = x.shape
    if num_samples <= ndim:
        err.warn('regression is underdetermined: {}'.format(x.shape))#: {}'.format(x))
        return True
    else:
        return False

    #elif len(x) <= 10:
        #err.warn('Less than 10 training samples!')#: {}'.format(x))


class UdetError(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class RegressionModel(object):
    #__metaclass__ = abc.ABCMeta

    def __init__(self, x, y):
        self.udet = is_undetermined(x)
        if self.udet:
            raise UdetError
        self.X, self.Y = x, y

    @abc.abstractproperty
    def A(self):
        raise NotImplementedError

    @abc.abstractproperty
    def b(self):
        raise NotImplementedError

    @abc.abstractproperty
    def fit_error(self):
        """ return fit_error """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        """ return Y """
        raise NotImplementedError

#     def error_pc_old_wrong_useless(self, X, Y):
#         err.warn_severe('DEPRCATED. Use error_rel_scaled_pc')
#         Y_ = self.predict(X)
#         #if settings.debug:
#         #    print 'score: ', self.clf.score(X, Y)
#         #    print abs((Y - Y_)/Y)*100
#         return (abs((Y - Y_)/Y))*100

    def error_pc(self, X, Y):
        """Relative error % scaled by estimated range

        Parameters
        ----------
        X : Test input
        Y : Test Output

        Returns
        -------
        error % vector

        Notes
        ------
        Can not really compute the exact range. Hence we just use the
        range of Y to scale e%. Justification: It is a dynamical system,
        and the next states Y are assumed to be close by.
        """
        '''computes relative error% along each dimension'''
#         Y_ = self.predict(X)
#         yrange = np.max(Y, axis=0) - np.min(Y, axis=0)
#         return (abs((Y - Y_)/yrange))*100

        # Usual relative error computation
        Y_ = self.predict(X)
        #epc = (abs((Y - Y_)/Y))*100
        error = Y - Y_
        #error_range = np.max(error, axis=0) - np.min(error, axis=0)
        Y_range = np.max(Y, axis=0) - np.min(Y, axis=0)
        epc = abs(error/Y_range)*100
        return epc

    def max_error_pc(self, X, Y):
        """Maximum error %

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
        #return np.max(self.error_pc(X, Y), axis=0)

        Y_ = self.predict(X)
        error = Y - Y_
        error_range = np.max(error, axis=0) - np.min(error, axis=0)
        Y_range = np.max(Y, axis=0) - np.min(Y, axis=0)
        return abs(error_range/Y_range)*100

#     def max_error_pc_old_wrong(self, X, Y):
#         Y_ = self.predict(X)
#         return np.max((abs((Y - Y_)/Y))*100, axis=0)

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
        assert(Y.shape == Yp.shape)
        delta = Y - Yp
        max_e = np.max(delta, axis=0)
        min_e = np.min(delta, axis=0)
        return cons.IntervalCons(min_e, max_e)

    def plot(self, X, Y, tol, title='unknown'):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        dimX = X.shape[1]
        assert(dimX == 2)

        # if 1 or less rows
        if X.shape[0] <= 1:
            err.warn('cant plot, only 1 value!!')
            return

        div = 50
        X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
        step0 = (X_max[0] - X_min[0])/div
        step1 = (X_max[1] - X_min[1])/div
        step = min(step0, step1)
        #print('x0 range:', X_min[0], X_max[0])
        #print('x1 range:', X_min[1], X_max[1])

        # Predict data of estimated models
        xx1, xx2 = np.mgrid[X_min[0]:X_max[0]:step, X_min[1]:X_max[1]:step]
        xx = np.vstack([xx1.ravel(), xx2.ravel()]).T
        yy = self.predict(xx)

        yy0 = yy[:, 0]
        yy1 = yy[:, 1]
        Y0 = Y[:, 0]
        Y1 = Y[:, 1]

        error_pc = self.error_pc(X, Y)
        outlier0_idx = error_pc[:, 0] > tol
        outlier1_idx = error_pc[:, 1] > tol

        if any(outlier0_idx):
            #print('X:\n', X[outlier0_idx, :])
            #print('Y0_pred:', self.predict(X[outlier0_idx, :])[:, 0])
            #print('Y0_true', Y0[outlier0_idx])
            Y0_pred_ = self.predict(X[outlier0_idx, :])[:, 0]
            Y0_pred = np.reshape(Y0_pred_, (Y0_pred_.size, 1))
            # make it 2-dim to match dim of X and Y0_pred
            Y0_true_ = Y0[outlier0_idx]
            Y0_true = np.reshape(Y0_true_, (Y0_true_.size, 1))
            epc0_ = error_pc[outlier0_idx, 0]
            epc0 = np.reshape(epc0_, (epc0_.size, 1))

            logger.debug('X - Y0_pred - Y0_true - error_pc')
            logger.debug(np.hstack((X[outlier0_idx, :], Y0_pred, Y0_true, epc0)))

        if any(outlier1_idx):
            Y1_pred_ = self.predict(X[outlier1_idx, :])[:, 1]
            Y1_pred = np.reshape(Y1_pred_, (Y1_pred_.size, 1))

            # make it 2-dim to match dim of X and Y1_pred
            Y1_true_ = Y1[outlier1_idx]
            Y1_true = np.reshape(Y1_true_, (Y1_true_.size, 1))
            epc1_ = error_pc[outlier1_idx, 1]
            epc1 = np.reshape(epc1_, (epc1_.size, 1))
            logger.debug('X - Y1_pred - Y1_true - error_pc')
            logger.debug(np.hstack((X[outlier1_idx, :], Y1_pred, Y1_true, epc1)))

        # plot the surface
        #print(xx)
        #print(xx2)
        #print(yy[:, 0])
        #embed()

        ################
        # First subplot
        ################

        fig = plt.figure()#figsize=plt.figaspect(2.))
        fig.suptitle(title)

        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.plot_surface(xx1, xx2, np.reshape(yy0, (xx1.shape)))
        ax.plot(X[:, 0], X[:, 1], Y0, 'y.')
        ax.plot(X[outlier0_idx, 0], X[outlier0_idx, 1], Y0[outlier0_idx], 'r.')
        ax.set_title('y0 vs x')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y0')

        #################
        # Second subplot
        #################
        ax = fig.add_subplot(2, 1, 2, projection='3d')
        ax.plot_surface(xx1, xx2, np.reshape(yy1, (xx1.shape)))
        ax.plot(X[:, 0], X[:, 1], Y1, 'y.')
        ax.plot(X[outlier1_idx, 0], X[outlier1_idx, 1], Y1[outlier1_idx], 'r.')
        ax.set_title('y1 vs x')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y1')

        #settings.plt_show()

# Old single figure plots
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         ax.plot_surface(xx1, xx2, np.reshape(yy0, (xx1.shape)))
#         ax.plot(X[:, 0], X[:, 1], Y0, 'y.')
#         ax.set_title('y0 vs x:' + title)
#         plt.show()

#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         ax.plot_surface(xx1, xx2, np.reshape(yy1, (xx1.shape)))
#         ax.plot(X[:, 0], X[:, 1], Y1, 'y.')
#         ax.set_title('y1 vs x:' + title)
#         plt.show()


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


class OLS(RegressionModel):
    def __init__(self, x, y, fit_intercept=True):
        super(self.__class__, self).__init__(x, y)
        # Copy must be on...arrays are getting reused!
        self.model = skl_lm.LinearRegression(copy_X=True, fit_intercept=fit_intercept, n_jobs=1, normalize=True)
        self.model.fit(x, y)
        self.fit_error_ = self.error(x, y)

    @property
    def A(self):
        return self.model.coef_

    @property
    def b(self):
        return self.model.intercept_

    def predict(self, X):
        Y = self.model.predict(X)
        return Y

    @property
    def fit_error(self):
        return self.fit_error_


class TLS(RegressionModel):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)
        # Copy must be on...arrays are getting reused!
        models, coefs, intercepts = [], [], []

        # for each coloumn of y
        for yi in y.T:
            model = skl_lm.TheilSenRegressor(copy_X=True, fit_intercept=True, n_jobs=1)
            #model = skl_lm.HuberRegressor(fit_intercept=True)
            model.fit(x, yi)
            models.append(model)
            coefs.append(model.coef_)
            intercepts.append(model.intercept_)

        self.coef = np.array(coefs)
        self.intercept = np.array(intercepts)

        self.fit_error_ = self.error(x, y)

    @property
    def fit_error(self):
        return self.fit_error_

    def predict(self, X):
        Y = np.dot(X, self.coef.T) + self.intercept
        return Y

    @property
    def A(self):
        return self.coef

    @property
    def b(self):
        return self.intercept


class RobustRegressionModel(RegressionModel):
    def __init__(self, x, y, fit=True):
        super(self.__class__, self).__init__(x, y)

        self.ransac_model = skl_lm.RANSACRegressor(residual_threshold=0.01)
        self.ransac_model.fit(x, y)
        self.fit_error = self.error(x, y)

#         print('='*20)
#         if np.any(self.model.coef_ != self.A):
#             err.warn_severe('Different')
#             print(self.model.coef_)
#             print(self.A)
#         print('='*20)

    @property
    def A(self):
        return self.ransac_model.estimator_.coef_

    @property
    def b(self):
        return self.ransac_model.estimator_.intercept_

    def predict(self, X):
        Y = self.ransac_model.predict(X)
        return Y


class LinReg(RegressionModel):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)
    pass


class KLinReg():

    IFILE = 'ip.data'
    MODELFILE = 'out.model'
    KLINREG = '/home/zutshi/work/RA/tools/klinreg/klinreg'
    K = 4

    def __init__(self, x, y):
        C = self.__class__
        models = []
        # for each coloumn of y
        for yi in y.T:

            # write the file
            xy = np.column_stack((x, np.ones(yi.size), yi))
            hdr = str(xy.shape).replace('(', '').replace(')', '').replace(',', '')
            s = str(xy).replace('[', '').replace(']', '')
            fops.write_data(C.IFILE, hdr+'\n'+s)
            # run klinreg
            U.strict_call([C.KLINREG, C.IFILE, C.K, '-d'])
            # read the output
            affine_funs = np.loadtxt(C.MODELFILE)


        for yi in y.T:
            model = skl_lm.TheilSenRegressor(copy_X=True, fit_intercept=True, n_jobs=1)
            #model = skl_lm.HuberRegressor(fit_intercept=True)
            model.fit(x, yi)
            models.append(model)
            coefs.append(model.coef_)
            intercepts.append(model.intercept_)

        self.coef = np.array(coefs)
        self.intercept = np.array(intercepts)

        # TODO: hack!
        self.fit_error_ = cons.zero2ic(x.ndim)

    @property
    def fit_error(self):
        return self.fit_error_

    def predict(self, X):
        Y = np.dot(X, self.coef.T) + self.intercept
        return Y

    @property
    def A(self):
        return self.coef

    @property
    def b(self):
        return self.intercept


class Poly(RegressionModel):
    def __init__(self, x, y, degree=2):
        super(self.__class__, self).__init__(x, y)
        self.poly_ = PolynomialFeatures(degree=degree, include_bias=True)
        x_poly = self.poly_.fit_transform(x)

        self.ols = OLS(x_poly, y, fit_intercept=False)
        self.coeffs = self.ols.A
        self.poly_rep = [power_array2Poly(self.poly_.powers_, coeff) for coeff in self.coeffs]

    @property
    def poly(self):
        return self.poly_rep

    def predict(self, X):
        return self.ols.predict(self.poly_.transform(X))

    @property
    def fit_error(self):
        return self.ols.fit_error_

    @property
    def A(self):
        #return self.ols.model.coef_
        raise NotImplementedError

    @property
    def b(self):
        #return self.ols.model.intercept_
        raise NotImplementedError
