#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:09:48 2019

@author: Ehsan
"""

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
from scipy.linalg import inv
from numpy.linalg import eig
import time
import sys


def oja(Y, k, eta0 = 1.0, alpha = 1.0, intermediate = False, steps = [], Y_test = []):
    dim, N = Y.shape
    t = time.clock()
    C = np.random.normal(size=(dim,k))
    C, _ = qr(C,mode='economic')
    if intermediate:
        errors = np.zeros(len(steps)+1)
        times = np.zeros(len(steps)+1)
        X = np.dot(C.T, Y_test)
        err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
        errors[0] = err
        cc = 1
    for nn in range(N):
        eta = eta0/((float(nn+1))**alpha)
        C, _ = qr_update(C, np.eye(k), eta * Y[:,nn], np.dot(C.T, Y[:,nn]))
        if intermediate:
            if ((nn+1) in steps):
                times[cc] = time.clock() - t
                X = np.dot(C.T, Y_test)
                err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
                errors[cc] = err
                cc += 1
    if intermediate:
        return C, errors, times
    else:
        return C
    
    
def krasulina(Y, k, eta0 = 1.0, alpha = 0.5, intermediate = False, steps = [], Y_test = []):
    dim, N = Y.shape
    t = time.clock()
    C = np.random.normal(size=(dim,k))
    C, _ = qr(C,mode='economic')
    if intermediate:
        errors = np.zeros(len(steps)+1)
        times = np.zeros(len(steps)+1)
        X = np.dot(C.T, Y_test)
        err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
        errors[0] = err
        cc = 1
    for nn in range(N):
        eta = eta0/((float(nn+1))**alpha)
        x = np.dot(C.T, Y[:,nn])
        C, _ = qr_update(C, np.eye(k), eta * (Y[:,nn] - np.dot(C, x)), x)
        if intermediate:
            if ((nn+1) in steps):
                times[cc] = time.clock() - t
                X = np.dot(C.T, Y_test)
                err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
                errors[cc] = err
                cc += 1
    if intermediate:
        return C, errors, times
    else:
        return C


def implicit_oja(Y, k, eta0 = 1.0, alpha = 1.0, intermediate = False, steps = [], Y_test = []):
    dim, N = Y.shape
    t = time.clock()
    C = np.random.normal(size=(dim,k))
    C, _ = qr(C,mode='economic')
    if intermediate:
        errors = np.zeros(len(steps)+1)
        times = np.zeros(len(steps)+1)
        X = np.dot(C.T, Y_test)
        err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
        errors[0] = err
        cc = 1
    for nn in range(N):
        eta = eta0/((float(nn+1))**alpha)
        eta = eta/(1.0 - eta * np.sum(Y[:,nn]**2))
        C, _ = qr_update(C, np.eye(k), eta * Y[:,nn], np.dot(C.T, Y[:,nn]))
        if intermediate:
            if ((nn+1) in steps):
                times[cc] = time.clock() - t
                X = np.dot(C.T, Y_test)
                err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
                errors[cc] = err
                cc += 1
    if intermediate:
        return C, errors, times
    else:
        return C

def implicit_krasulina(Y, k, eta0 = 1.0, alpha = 0.5, intermediate = False, steps = [], Y_test = []):
    dim, N = Y.shape
    t = time.clock()
    C = 0.1 * np.random.normal(size=(dim,k))
    Cinv = inv(np.dot(C.T, C))
    if intermediate:
        errors = np.zeros(len(steps)+1)
        times = np.zeros(len(steps)+1)
        X = np.dot(Cinv, np.dot(C.T, Y_test))
        err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
        errors[0] = err
        cc = 1
    for nn in range(N):
        eta = eta0/((float(nn+1))**alpha)
        mu = np.dot(Cinv,np.dot(C.T,Y[:,nn]))
        mu2 = np.sum(mu**2)
        C = C + eta/(1 + eta * mu2) * np.outer(Y[:,nn] - np.dot(C, mu), mu)
        U = np.array([mu, eta * mu]).T
        V = np.array([eta * mu + eta**2 * np.dot(Y[:,nn], Y[:,nn]) * mu, mu]).T
        Cinv = Cinv - np.dot(Cinv, np.dot(np.dot(U, np.linalg.inv(np.eye(2) + np.dot(np.dot(V.T, Cinv), U))), np.dot(V.T, Cinv)))
        if intermediate:
            if ((nn+1) in steps):
                times[cc] = time.clock() - t
                X = np.dot(Cinv, np.dot(C.T, Y_test))
                err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
                errors[cc] = err
                cc += 1
    if intermediate:
        return C, errors, times
    else:
        return C
 
    
def inc(Y, k, intermediate = False, steps = [], Y_test = []):
    d, N = Y.shape
    t = time.clock()
    D = 1e-3 * np.eye(k)
    C = np.random.normal(size=(d,k))
    C, _ = qr(C,mode='economic')
    if intermediate:
        errors = np.zeros(len(steps)+1)
        times = np.zeros(len(steps)+1)
        X = np.dot(C.T, Y_test)
        err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
        errors[0] = err
        cc = 1
    for nn in range(N):
        y = Y[:,nn]
        x = np.dot(C.T, y)[:,np.newaxis]
        D = D + np.outer(x,x)
        yo = y[:,np.newaxis] - np.dot(C, x)
        yo_norm = np.linalg.norm(yo)
        Q = np.hstack((D, yo_norm * x))
        Q = np.vstack((Q, np.append(yo_norm * x, yo_norm ** 2).T))
        D, U = eig(Q)
        idx = np.argsort(-D)
        D = np.real(np.diag(D[idx[range(k)]]))
        C = np.hstack((C, yo/yo_norm))
        C = np.real(np.dot(C, U)[:,idx[range(k)]])
        if intermediate:
            if ((nn+1) in steps):
                times[cc] = time.clock() - t
                X = np.dot(C.T, Y_test)
                err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
                errors[cc] = err
                cc += 1
    if intermediate:
        return C, errors, times
    else:
        return C
    
def msg(Y, k, eta0 = 1.0, lam = 1.0, mu = 1.0, intermediate = False, steps = [], Y_test = []):
    d, N = Y.shape
    t = time.clock()
    D = np.zeros((k,k))
    C = np.random.normal(size=(d,k))
    C, _ = qr(C,mode='economic')
    if intermediate:
        errors = np.zeros(len(steps)+1)
        times = np.zeros(len(steps)+1)
        X = np.dot(C.T, Y_test)
        err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
        errors[0] = err
        cc = 1
    for nn in range(N):
        y = Y[:,nn]
        eta = eta0/((float(nn+1)) * lam)
        x = np.sqrt(eta) * np.dot(C.T, y)[:,np.newaxis]
        r = np.sqrt(eta) * y[:,np.newaxis] - np.dot(C, x)
        r_norm = np.linalg.norm(r)
        D = (1.0 - lam * eta) * D - mu * eta * np.eye(k) + np.outer(x,x)
        Q = np.hstack((D, r_norm * x))
        Q = np.vstack((Q, np.append(r_norm * x, r_norm ** 2).T))
        D, U = eig(Q)
        idx = np.argsort(-D)
        D = np.real(np.diag(D[idx[range(k)]]))
        C = np.hstack((C, r/r_norm))
        C = np.real(np.dot(C, U)[:,idx[range(k)]])
        if intermediate:
            if ((nn+1) in steps):
                times[cc] = time.clock() - t
                X = np.dot(C.T, Y_test)
                err = np.mean(np.sum((Y_test - np.dot(C,X))**2, axis=0))
                errors[cc] = err
                cc += 1
    if intermediate:
        return C, errors, times
    else:
        return C
    

def main():
    filename = sys.argv[1]
    method = sys.argv[2]
    mat = sio.loadmat('../data/' + filename + '.mat')
    data = mat['X'].astype(np.float64)
    data = data[:60000,:]
    data /= np.max(data)
    data -= np.mean(data,axis=0)
    N = data.shape[0]
    Y = data[np.random.permutation(N),:].T
    
    num_iter = 1
    k_values = [25, 30, 35, 40]
    eta_init_values = 10.0 ** np.r_[-3:4]
    lam_values = 10.0 ** np.r_[-3:4]
    mu_values = 10.0 ** np.r_[-3:4]
    
    steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000, 20000, 30000, 40000, 54000]
    num_steps = len(steps)
    error_all = np.zeros((len(k_values), num_iter, num_steps+1))
    time_all = np.zeros((len(k_values), num_iter, num_steps+1))
    num_train = 6000
    
    for itr in range(num_iter):
        idx_train_test = np.random.permutation(N)
        Y_train = Y[:, idx_train_test[:num_train]]
        Y_test = Y[:, idx_train_test[num_train:]]
        for i, k in enumerate(k_values):
            fid = open('./log/pca_log_' +  filename + '_' + method + '.txt', 'a+')
            
            if method == 'oja':
                error_val = np.zeros(len(eta_init_values))
                for j, eta_init in enumerate(eta_init_values):
                    C = oja(Y_train, k, eta0=eta_init, alpha=1.0)
                    X = np.dot(C.T, Y_train)
                    error_val[j] = np.mean(np.sum((Y_train - np.dot(C,X))**2, axis=0))
                idx_best = np.argmin(error_val)
                eta_best = eta_init_values[idx_best]
                C, errors, times = oja(Y_test,k, eta0=eta_best, alpha=1.0, 
                                       intermediate = True, steps=steps, Y_test = Y_test)
                time_all[i,itr,:] = times
                error_all[i,itr,:] = errors
                
            elif method == 'krasulina':
                error_val = np.zeros(len(eta_init_values))
                for j, eta_init in enumerate(eta_init_values):
                    C = krasulina(Y_train, k, eta0=eta_init, alpha=0.5)
                    X = np.dot(C.T, Y_train)
                    error_val[j] = np.mean(np.sum((Y_train - np.dot(C,X))**2, axis=0))
                    
                idx_best = np.argmin(error_val)
                eta_best = eta_init_values[idx_best]
                C, errors, times = krasulina(Y_test,k, eta0=eta_best, alpha=0.5, 
                                       intermediate = True, steps=steps, Y_test = Y_test)
                time_all[i,itr,:] = times
                error_all[i,itr,:] = errors
                
            elif method == 'implicit_oja':
                error_val = np.zeros(len(eta_init_values))
                for j, eta_init in enumerate(eta_init_values):
                    C = implicit_oja(Y_train, k, eta0=eta_init, alpha=1.0)
                    X = np.dot(C.T, Y_train)
                    error_val[j] = np.mean(np.sum((Y_train - np.dot(C,X))**2, axis=0))
                idx_best = np.argmin(error_val)
                eta_best = eta_init_values[idx_best]
                C, errors, times = implicit_oja(Y_test,k, eta0=eta_best, alpha=1.0, 
                                       intermediate = True, steps=steps, Y_test = Y_test)
                time_all[i,itr,:] = times
                error_all[i,itr,:] = errors
                
            elif method == 'implicit_krasulina':
                error_val = np.zeros(len(eta_init_values))
                for j, eta_init in enumerate(eta_init_values):
                    C = implicit_krasulina(Y_train, k, eta0=eta_init, alpha=0.5)
                    X = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, Y_train))
                    error_val[j] = np.mean(np.sum((Y_train - np.dot(C,X))**2, axis=0))
                idx_best = np.argmin(error_val)
                eta_best = eta_init_values[idx_best]
                C, errors, times = implicit_krasulina(Y_test,k, eta0=eta_best, alpha=0.5, 
                                       intermediate = True, steps=steps, Y_test = Y_test)
                time_all[i,itr,:] = times
                error_all[i,itr,:] = errors
                
            elif method == 'inc':
                C, errors, times = inc(Y,k,intermediate = True, steps=steps, Y_test = Y_test)
                time_all[i,itr,:] = times
                error_all[i,itr,:] = errors
            
            elif method == 'msg':
                if i == 0:
                    error_val = np.zeros((len(eta_init_values),
                                         len(lam_values), len(mu_values)))
                    for e, eta0 in enumerate(eta_init_values):
                        for l, lam in enumerate(lam_values):
                            for m, mu in enumerate(mu_values):
                                C, errors, _ = msg(Y_train[:,:1000], 40, eta0=eta0, lam=lam, mu=mu,
                                        intermediate = True, steps=[1,1000], Y_test = Y_train[:,1000:1100])
                                error_val[e,l,m] = errors[2] - errors[1]
                    
                    idx_min = np.where(error_val == np.min(error_val))
                    eta_best = eta_init_values[idx_min[0][0]]
                    lam_best = lam_values[idx_min[1][0]]
                    mu_best = mu_values[idx_min[2][0]]
                error_val = np.zeros(len(eta_init_values))
                for j, eta_init in enumerate(eta_init_values):
                    C = msg(Y_train, k, eta0=eta_init, lam=lam_best, mu=mu_best)
                    X = np.dot(C.T, Y_train)
                    error_val[j] = np.mean(np.sum((Y_train - np.dot(C,X))**2, axis=0))
                idx_best = np.argmin(error_val)
                eta_best = eta_init_values[idx_best]
                
                C, errors, times = msg(Y_test,k, eta0=eta_best, lam=lam_best, mu=mu_best, 
                                           intermediate = True, steps=steps, Y_test = Y_test)
                time_all[i,itr,:] = times
                error_all[i,itr,:] = errors
                
            print('iteration = %d, k = %d, method = %s' % (itr+1, k, method))
            fid.write('iteration = %d, k = %d\n' % (itr+1, k))
            fid.close()
            
    
        sio.savemat('../results/rerun/pca_results_' +  filename + '_' + method 
                    + '.mat', {'error':error_all, 'runtime':time_all})

if __name__ == "__main__":
    main()
