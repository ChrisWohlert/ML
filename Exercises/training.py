# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:23:27 2018
@author: Sila
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = 6 * np.random.rand(100, 1) - 3

plt.axis([-5,10,-5,10])

squared = lambda x: x ** 2
absolute = lambda x: abs(x)

plt.plot(X,squared(X), "b.")
plt.plot(X,absolute(X), "g.")
plt.show()