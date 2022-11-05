# -*- coding: utf-8 -*-
"""
Created on Mon September 20 17:01:45 2022

@author: sila
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
# Create Sample data
from sklearn.datasets import make_moons
X, y= make_moons(n_samples=500, shuffle=True, noise=0.1, random_state=20)


# The idea is this:
# - Any sample who has min_samples neigbours by the distance of epsilon is a core sample.
# - Any data sample which is not core, but has at least one core neighbor
# (with a distance less than eps), is a directly reachable sample and can be added to the cluster.
# - Any data sample which is not directly reachable nor a core, but has at least
#  one directly reachable neighbor (with a distance less than eps)
#  is a reachable sample and will be added to the cluster.
# - Any other examples are considered to be noise, outlier or
# whatever you want to name it.( and those will be labeled by -1)

from sklearn.cluster import DBSCAN
from sklearn import metrics
db = DBSCAN(eps=0.207, min_samples=10)
print(metrics.get_scorer_names())

grid = GridSearchCV(DBSCAN(), n_jobs=5, param_grid=[{"eps": [0.1, 0.15, 0.20]}, { "min_samples": [5, 6, 7, 8, 9, 10, 11, 12] }], scoring='precision')
grid.fit(X, y)

# Plotting the clusters
plt.scatter(x= X[:,0], y= X[:,1], c=grid.predict(X))

plt.show()