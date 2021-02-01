# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:55:34 2019

@author: User
"""

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

veri = pd.read_csv("breast-cancer-wisconsin.data")
"""
veri.replace('?', -99999, inplace='true')
veri.drop(['id'], axis=1)
"""

y = veri.values[:,7]
x = veri.values[:,0:6]+veri.values[:,7:8]

from sklearn.cluster.k_means_ import KMeans

kmeans=KMeans(n_clusters=3)
kmeans.fit(x,y)
print(kmeans.cluster_centers_)
print(pd.crosstab(y,kmeans.labels_))
"""
imp = Imputer(missing_values=-99999, strategy="mean",axis=0)
x = imp.fit_transform(x)

"""
""""
for z in range(25):
    z = 2*z+1
    print("En yakın",z,"komşu kullandığımızda tutarlılık oranımız")
    tahmin = KNeighborsClassifier(n_neighbors=z, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
    tahmin.fit(x,y)
    ytahmin = tahmin.predict(x)
    basari = accuracy_score(y, ytahmin, normalize=True, sample_weight=None)
    print(basari)
    
    
"""
"""

tahmin = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
tahmin.fit(x,y)
ytahmin = tahmin.predict(x)

basari = accuracy_score(y, ytahmin, normalize=True, sample_weight=None)
print("Yüzde",basari*100," oranında:" )

print(tahmin.predict([1,2,2,2,3,2,1,2,3,2]))"""