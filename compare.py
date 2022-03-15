#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:24:00 2019

@author: Ehsan
"""

filename = 'scene'
mat = sio.loadmat('../data/' + filename + '.mat')
data = mat['X'].astype(np.float64)
data /= np.max(data)
data -= np.mean(data,axis=0)
N = data.shape[0]
Y = data[np.random.permutation(N),:].T

k = 100

pca = PCA(n_components = k)
pca.fit(Y.T)
C = pca.components_.T
X = np.dot(C.T, Y)
print(np.mean(np.sum((Y - np.dot(C,X))**2, axis=0)))

C = inc(Y,k)
X = np.dot(C.T, Y)
print(np.mean(np.sum((Y - np.dot(C,X))**2, axis=0)))

eta_init = np.minimum(float(k), 10.0)
alpha = 0.8
C = krasulina(Y,k, eta0=eta_init, alpha=a)
X = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, Y))
print(np.mean(np.sum((Y - np.dot(C,X))**2, axis=0)))

eta_init = 10.0 * eta_init
C = implicit_krasulina(Y,k, eta0=eta_init, alpha=a)
X = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, Y))
print(np.mean(np.sum((Y - np.dot(C,X))**2, axis=0)))

