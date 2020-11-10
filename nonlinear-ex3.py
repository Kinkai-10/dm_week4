import numpy as np
import datasets
import regression
import importlib

X, Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial3_features(X)

samples = np.arange(0, 4, 0.1)
x_samples = np.c_[ np.ones(len(samples)), samples ]
ex_x_samples = datasets.polynomial3_features(x_samples)

list = [0, 0.1, 0.5, 1.0, 10]

import matplotlib.pyplot as plt
plt.scatter(X[:,1], Y)

for x in list:
    model = regression.RidgeRegression(x)
    model.fit(ex_X, Y)
    plt.plot(samples, model.predict(ex_x_samples))

plt.show()
