import GPy
import numpy as np
from matplotlib import pyplot as plt
import libs
from libs import datalib

file_name = 'trial_1_vel'

sample_size = 5
X, Y = datalib.get_data(file_name)
X = X.reshape(-1,1)
# X = np.random.uniform(0, 1., (sample_size, 1))
# Y = np.sin(X) + np.random.randn(sample_size, 1)*0.05

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
model = GPy.models.GPRegression(X,Y,kernel, noise_var=1e-10)

testX = np.linspace(0, 1, 100).reshape(-1, 1)
posteriorTestY = model.posterior_samples_f(testX, full_cov=True, size=3)
simY, simMse = model.predict(testX)

plt.plot(testX, posteriorTestY.squeeze())
plt.plot(X, Y, 'ok', markersize=10)
plt.plot(testX, simY - 3 * simMse ** 0.5, '--g')
plt.plot(testX, simY + 3 * simMse ** 0.5, '--g')

plt.show()