import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel)


file_name = 'trial_1_vel'







# kernels = [, 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1), 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0)), ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
#            1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)]


def process_gaussian(x, y, fit, kernel, title):
    '''
    Processor.

    Args:
      x : 1D numpy array of x data
      y : 1D numpy array of y data
      fit : Whether or not to fit GP to data (boolean)
      kernel : specified kernel
      title : text
    '''
    gp = GaussianProcessRegressor(kernel=kernel)

    matplotlib.style.use('classic')
    plt.figure()
    plt.subplot(1, 1, 1)

    # plot mean
    if fit:
        gp.fit(x.reshape(-1,1), y.reshape(-1,1))

    x_int = np.linspace(start=0, stop=1, num=100, endpoint=True, retstep=False)
    y_mean, y_std = gp.predict(x_int.reshape(-1,1), return_std=True, return_cov=False)
    plt.plot(x_int, y_mean, 'black', lw=3)

    # plot std
    if y_mean.shape != (100,):
        print('reshaping!')
        print(y_mean)
        # y_mean.reshape(-1,2)
        y_mean = y_mean.flatten()
        print(y_mean)
        print('STD')
        print(y_std)
        print(x_int)
        plt.fill_between(x_int, y_mean - y_std, y_mean + y_std, alpha=0.2, color='black')
    else:
        print(y_mean.shape, y_std.shape)
        plt.fill_between(x_int, y_mean - y_std, y_mean + y_std, alpha=0.2, color='black')
    

    # plot sample lines
    y_samples = gp.sample_y(x_int.reshape(-1,1), n_samples=10)
    if y_samples.shape != (100,10):
        y_samples = y_samples.reshape(100,10)
        print('NEW SHAPE: {}'.format(y_samples.shape))

    plt.plot(x_int, y_samples, lw=1)

    # plot data points
    if fit:
        plt.scatter(x.reshape(-1,1), y.reshape(-1,1), c='red', s=50, zorder=10, edgecolors=(0,0,0))

    plt.xlim(-0.1, 1)
    plt.ylim(-0.1, 1.5)
    plt.title(title + '(kernel: {})\n Log-Likelihood: {}'.format(kernel, 'test'), fontsize=12)
# kernel, gp.log_marginal_likelihood()
    plt.tight_layout()
    plt.show()


# Main
data_X, data_Y = get_data(file_name)

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))


# plot prior
process_gaussian(x=None, y=None, fit=False, kernel=kernel, title='Prior')
# plot posterior
process_gaussian(x=data_X, y=data_Y, fit=True, kernel=kernel, title='Posterior')