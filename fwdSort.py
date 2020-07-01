import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state


theta = np.array([1.0, 9.0, 5.0, 3.0, 2.0, 4.0])
rho   = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
eps = (1.0 / 1000.0)

x = np.arange(len(theta))[::-1]
ir = IsotonicRegression()
z = rho/eps # -theta/eps
w = -np.sort(-theta)   # rho

sigma = np.argsort(-z)      #   <- play with this?
sigma_inv = np.argsort(sigma)

s = z[sigma]

v = ir.fit_transform(x,  s-w)
thetasort = z - v[sigma_inv]


print("sigma = ", sigma)
print("sigma_inv = ", sigma_inv)
print("s = ", s)
print("theta[sigma][sigma_inv] = ",  theta[sigma][sigma_inv] )
print("v = ", v)
print("w = ", w)
print("thetasort = ", thetasort)


