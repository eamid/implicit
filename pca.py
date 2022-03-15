#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 23:25:02 2019

@author: Ehsan
"""


import numpy as np
from scipy.linalg import qr
from scipy.linalg import qr_update
import scipy.io as sio
from sklearn.decomposition import PCA
#from scipy.linalg import lstsq
from scipy.linalg import inv
from numpy.linalg import eig
import time
import datetime
import sys


def oja(Y, k, eta0 = 1.0, alpha = 0.7, intermediate = False, step = 1000):
    dim, N = Y.shape
    C = np.random.normal(size=(dim,k))
    C, R = qr(C,mode='economic')
    idx = np.random.permutation(N)
    if intermediate:
        errors = np.zeros((N//step+1))
        X = np.dot(C.T, Y)
        err = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
        errors[0] = err
        print('initial  error ' + str(err))
    for ii in range(N):
        nn = idx[ii]
        eta = eta0/((float(ii+1))**alpha)
        C, R = qr_update(C, R, eta * Y[:,nn], np.dot(C.T, Y[:,nn]))
        if intermediate:
            if ((ii+1)%step == 0):
                X = np.dot(C.T, Y)
                err = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                errors[(ii+1)//step] = err
                print('iteration ' + str(ii+1) + ', error ' + str(err))
    if intermediate:
        return C, errors
    else:
        return C
    
    
def krasulina(Y, k, eta0 = 1.0, alpha = 0.7, intermediate = False, step = 1000):
    dim, N = Y.shape
    C = np.random.normal(size=(dim,k))
    C, _ = qr(C,mode='economic')
    idx = np.random.permutation(N)
    if intermediate:
        errors = np.zeros((N//step+1))
        X = np.dot(C.T, Y)
        err = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
        errors[0] = err
        print('initial  error ' + str(err))
    for ii in range(N):
        nn = idx[ii]
        eta = eta0/((float(ii+1))**alpha)
        x = np.dot(C.T, Y[:,nn])
        C, R = qr_update(C, np.eye(k), eta * (Y[:,nn] - np.dot(C, x)), x)
        if intermediate:
            if ((ii+1)%step == 0):
                X = np.dot(C.T, Y)
                err = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                errors[(ii+1)//step] = err
                print('iteration ' + str(ii+1) + ', error ' + str(err))
    if intermediate:
        return C, errors
    else:
        return C

def implicit_oja(Y, k, eta0 = 1.0, alpha = 0.7, intermediate = False, step = 1000):
    dim, N = Y.shape
    C = np.random.normal(size=(dim,k))
    C, _ = qr(C,mode='economic')
    idx = np.random.permutation(N)
    if intermediate:
        errors = np.zeros((N//step+1))
        X = np.dot(C.T, Y)
        err = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
        errors[0] = err
        print('initial  error ' + str(err))
    for ii in range(N):
        nn = idx[ii]
        eta = eta0/((float(ii+1))**alpha)
        eta = eta/(1.0 - eta * np.sum(Y[:,nn]**2))
        C, R = qr_update(C, np.eye(k), eta * Y[:,nn], np.dot(C.T, Y[:,nn]))
        if intermediate:
            if ((ii+1)%step == 0):
                X = np.dot(C.T, Y)
                err = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                errors[(ii+1)//step] = err
                print('iteration ' + str(ii+1) + ', error ' + str(err))
    if intermediate:
        return C, errors
    else:
        return C

def implicit_krasulina(Y, k, eta0 = 1.0, alpha = 0.7, intermediate = False, step = 1000):
    dim, N = Y.shape
    C = 0.1 * np.random.normal(size=(dim,k))
    Cinv = inv(np.dot(C.T, C))
    idx = np.random.permutation(N)
    if intermediate:
        errors = np.zeros((N//step+1))
        X = np.dot(np.dot(Cinv,C.T),Y)
#        X = lstsq(C, Y)[0]
        err = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
        errors[0] = err
        print('initial  error ' + str(err))
    for ii in range(N):
        nn = idx[ii]
        eta = eta0/((float(ii+1))**alpha)
        mu = np.dot(Cinv,np.dot(C.T,Y[:,nn]))
        mu2 = np.sum(mu**2)
        C = C + eta/(1 + eta * mu2) * np.outer(Y[:,nn] - np.dot(C, mu), mu)
#        C = eta /(1 + eta * mu2) * np.outer(Y[:,nn], mu) + C - eta * np.outer(np.dot(C, mu), mu)/(1 + eta * mu2)
        if intermediate:
            if ((ii+1)%step == 0):
#                X = lstsq(C, Y)[0]
                X = np.dot(np.dot(inv(np.dot(C.T,C)),C.T),Y)
                err = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                errors[(ii+1)//step] = err
                print('iteration ' + str(ii+1) + ', error ' + str(err))
        U = np.array([mu, eta * mu]).T
        V = np.array([eta * mu + eta**2 * np.dot(Y[:,nn], Y[:,nn]) * mu, mu]).T
        Cinv = Cinv - np.dot(Cinv, np.dot(np.dot(U, np.linalg.inv(np.eye(2) + np.dot(np.dot(V.T, Cinv), U))), np.dot(V.T, Cinv)))
    if intermediate:
        return C, errors
    else:
        return C
 
    
def inc(Y, k):
    d, N = Y.shape
    idx = np.random.permutation(N)
    D = 1e-3 * np.eye(k)
    C = np.eye(d)[:,:k]
    for ii in range(N):
        y = Y[:,ii]
        x = np.dot(C.T, y)[:,np.newaxis]
        yo = y[:,np.newaxis] - np.dot(C, x)
        D = D + np.outer(x,x)
        yo_norm = np.linalg.norm(yo)
        Q = np.hstack((D, yo_norm * x))
        Q = np.vstack((Q, np.append(yo_norm * x, yo_norm ** 2).T))
        D, U = eig(Q)
        idx = np.argsort(-D)
        D = np.real(np.diag(D[idx[range(k)]]))
        C = np.hstack((C, yo/yo_norm))
        C = np.real(np.dot(C, U)[:,idx[range(k)]])
    return C
    
    

#def showtime(t):
#    elapsed = str(datetime.clockdelta(seconds=t))
#    print("Elapsed time: %s" % (elapsed))

def main():
    filename = sys.argv[1]
    mat = sio.loadmat('../data/' + filename + '.mat')
    data = mat['X'].astype(np.float64)
    data /= np.max(data)
    data -= np.mean(data,axis=0)
    N = data.shape[0]
    Y = data[np.random.permutation(N),:].T
    
    a = 0.5
    num_iter = 10
#    k_values = [int(sys.argv[2])]
#    k_values = [1, 2, 5, 10, 20, 50, 80, 100, 150, 200]
    k_values = [25, 30, 35, 40]
#    eta_init_values = [0.01, 0.1, 0.5, 1.0, 2.0]
    eta_init_values = [1.0]
    
    error_all = np.zeros((6, len(k_values), len(eta_init_values), num_iter))
    time_all = np.zeros((6, len(k_values), len(eta_init_values), num_iter))
    
    for itr in range(num_iter):
        for i, k in enumerate(k_values):
            
            # PCA
            t = time.clock()
            pca = PCA(n_components = k)
            pca.fit(Y.T)
            time_all[0,i,:,:] = time.clock() - t
            C = pca.components_.T
            X = np.dot(C.T, Y)
            error_all[0,i,:,:] = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
            
            fid = open('./log/pca_log_' +  filename + '.txt', 'a+')
            for j, eta_init in enumerate(eta_init_values):
                
#                eta_init = eta_init/(float(k) ** 0.05)
                eta_init = np.minimum(float(k), 10.0)
                
                # Oja's
                t = time.clock()
                C = oja(Y,k, eta0=eta_init, alpha=a)
                time_all[1,i,j,itr] = time.clock() - t
                X = np.dot(C.T, Y)
                error_all[1,i,j,itr] = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                
                # Krasulina's
                t = time.clock()
                C = krasulina(Y,k, eta0=eta_init, alpha=a)
                time_all[2,i,j,itr] = time.clock() - t
                X = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, Y))
                error_all[2,i,j,itr] = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                
                # implicit Oja's
                t = time.clock()
                C = implicit_oja(Y,k, eta0=eta_init, alpha=a)
                time_all[3,i,j,itr] = time.clock() - t        
                X = np.dot(C.T, Y)
                error_all[3,i,j,itr] = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                
                # implicit Krasulina's
                t = time.clock()
                C = implicit_krasulina(Y,k, eta0 = 10.0 * eta_init, alpha=a)
                time_all[4,i,j,itr] = time.clock() - t
                X = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, Y))
                error_all[4,i,j,itr] = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                
                # Incremental
                t = time.clock()
                C = inc(Y,k)
                time_all[5,i,j,itr] = time.clock() - t
                X = np.dot(C.T, Y)
                error_all[5,i,j,itr] = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
                
            print('iteration = %d, k = %d' % (itr+1, k))
            fid.write('iteration = %d, k = %d\n' % (itr+1, k))
            
    
        sio.savemat('../results/rerun/pca_results_' +  filename + '.mat', {'error':error_all, 'runtime':time_all})
    fid.close()

if __name__ == "__main__":
    main()
