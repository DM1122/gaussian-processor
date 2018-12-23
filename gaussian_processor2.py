import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel)

import libs
from libs import datalib

file_name = 'trial_1_vel'

kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-1, 10.0))

gp = GaussianProcessRegressor(kernel=kernel)

# plot prior
plt.figure()
plt.subplot(2, 1, 1)
x_int = np.linspace(start=0, stop=1, num=100, endpoint=True, retstep=False)
y_mean, y_std = gp.predict(x_int.reshape(-1,1), return_std=True, return_cov=False)
plt.plot(x_int, y_mean, 'black', lw=3)
plt.fill_between(x_int, y_mean - y_std, y_mean + y_std, alpha=0.2, color='black')
y_samples = gp.sample_y(x_int.reshape(-1,1), 10)
plt.plot(x_int, y_samples, lw=1)
plt.xlim(0, 1)
plt.ylim(0, 1.5)
plt.title('Prior (kernel: {})'.format(kernel), fontsize=12)

# get data
x, y = datalib.get_data(file_name)
gp.fit(x.reshape(-1,1), y)

# plot posterior
plt.subplot(2, 1, 2)
# x_int = np.linspace(start=0, stop=1, num=100, endpoint=True, retstep=False)
print(y_mean.shape)
y_mean, y_std = gp.predict(x_int.reshape(-1,1), return_std=True, return_cov=False)
# y_mean  = y_mean.flatten()
print(y_std.shape)
plt.plot(x_int, y_mean, 'black', lw=3)
plt.fill_between(x_int, y_mean - y_std, y_mean + y_std, alpha=0.25, color='blue')
# plt.fill(np.concatenate([x_int, x_int[::-1]]), np.concatenate([y_mean - y_std, (y_mean + y_std)[::-1]]), alpha=.2, fc='b', ec='None', label='95% confidence interval')
# plt.fill(np.concatenate([x_int, x_int[::-1]]), np.concatenate([y_mean - 1.9600 * y_std, (y_mean + 1.9600 * y_std)[::-1]]), alpha=.5, fc='b', ec='None', label='95% confidence interval')

# plot sample lines
y_samples = gp.sample_y(x_int.reshape(-1,1), 10)
print(y_samples.shape)
plt.plot(x_int, y_samples, lw=1)
plt.scatter(x.reshape(-1,1), y.reshape(-1,1), c='red', s=50, zorder=10, edgecolors=(0,0,0))
plt.xlim(-0.1, 1)
plt.ylim(-0.1, 1.5)
plt.title('Posterior (kernel: {})\n Log-Likelihood: {}'.format(gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)), fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1.5)
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()