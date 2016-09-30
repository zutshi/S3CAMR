from __future__ import print_function

import logging

from sklearn import linear_model as skl_lm
import numpy as np

#from IPython import embed

import constraints as cons
import err

import settings

logger = logging.getLogger(__name__)


# TODO: UNUSED
def factory(model_type):
    if model_type == 'approx':
        return RegressionModel()
    elif model_type == 'sound':
        return FlowStarModel()
    else:
        raise NotImplementedError


def warn_if_small_data_set(x):
    if len(x) <= 2:
        err.warn('Less than 2 training samples!')#: {}'.format(x))
    #elif len(x) <= 10:
        #err.warn('Less than 10 training samples!')#: {}'.format(x))


class RegressionModel(object):
    def __init__(self, x, y):
        warn_if_small_data_set(x)
        #self.clf_ = skl_lm.RidgeCV(alphas=[1.0, 1.0], fit_intercept=True)
        # Copy must be on...arrays are getting reused!
        self.clf_ = skl_lm.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
        self.clf_.fit(x, y)
        #if settings.debug:
        #    print clf_.coef_, clf_.intercept_
        self.fit_error = self.__error(x, y)

    @property
    def clf(self):
        return self.clf_

    @property
    def score(self):
        return self.clf.score()

    @property
    def A(self):
        return self.clf.coef_

    @property
    def b(self):
        return self.clf.intercept_

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
        #if settings.debug:
        #    print 'score: ', self.clf.score(X, Y)
        #    print abs((Y - Y_)/Y)*100
        return (abs((Y - Y_)/Y))*100

    def max_error_pc(self, X, Y):
        """max_error_pc

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
        #if settings.debug:
        #    print 'score: ', self.clf.score(X, Y)
        #    print abs((Y - Y_)/Y)*100
        return np.max((abs((Y - Y_)/Y))*100, axis=0)

    def __error(self, X, Y):
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

        settings.plt_show()

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


class RobustRegressionModel(object):
    def __init__(self, x, y):
        warn_if_small_data_set(x)
        #self.clf_ = skl_lm.RidgeCV(alphas=[1.0, 1.0], fit_intercept=True)
        # Copy must be on...arrays are getting reused!
        self.model = skl_lm.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
        #self.clf_.fit(x, y)
        self.clf_ = skl_lm.RANSACRegressor(self.model)
        self.clf_.fit(x, y)

        #if settings.debug:
        #    print clf_.coef_, clf_.intercept_

    @property
    def clf(self):
        return self.clf_

    @property
    def A(self):
        return self.clf.estimator_.coef_

    @property
    def b(self):
        return self.clf.estimator_.intercept_

    def predict(self, X):
        Y = self.clf.predict(X)
        return Y

    def max_error_pc(self, X, Y):
        """max_error_pc

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
        #if settings.debug:
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

"""robust linear regression RANSAC: example"""
# n_samples = 1000
# n_outliers = 50

# X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
#                                       n_informative=1, noise=10,
#                                       coef=True, random_state=0)

# # Add outlier data
# np.random.seed(0)
# X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
# y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# # Fit line using all data
# model = linear_model.LinearRegression()
# model.fit(X, y)

# # Robustly fit linear model with RANSAC algorithm
# model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
# model_ransac.fit(X, y)
# inlier_mask = model_ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# # Predict data of estimated models
# line_X = np.arange(-5, 5)
# line_y = model.predict(line_X[:, np.newaxis])
# line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])

# # Compare estimated coefficients
# print("Estimated coefficients (true, normal, RANSAC):")
# print(coef, model.coef_, model_ransac.estimator_.coef_)

# plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
# plt.plot(X[outlier_mask], y[outlier_mask], '.r', label='Outliers')
# plt.plot(line_X, line_y, '-k', label='Linear regressor')
# plt.plot(line_X, line_y_ransac, '-b', label='RANSAC regressor')
# plt.legend(loc='lower right')
# plt.show()
