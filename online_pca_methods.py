import numpy as np
from numpy.linalg import norm
from numpy.linalg import eig
from scipy.sparse.linalg import eigs
import numba
from numba import jit
from scipy.linalg import qr
from scipy.linalg import qr_update
from scipy.linalg import inv
from scipy.linalg import pinv

def project_eigs(sig, k):
	d = len(sig)
	unique_eigs, kappa = np.unique(sig, return_counts=True)
	idx = np.argsort(unique_eigs)
	unique_eigs = unique_eigs[idx]
	kappa = kappa[idx]
	n = len(unique_eigs)
	i = j = 0
	si = sj = ci = cj = 0.0
	while i < n:
		if (i < j):
			S = (k - (sj - si) - (d - cj))/(cj - ci)
			if ((unique_eigs[i] + S) >= 0) and (unique_eigs[j-1] + S <= 1) and ((i < 1) or ((unique_eigs[i-1] + S) <= 0)) and ((j >= (n-1)) or (unique_eigs[j+1] >= 1)):
				break
		if (j < n) and ((unique_eigs[j] - unique_eigs[i]) <= 1):
			sj = sj + kappa[j] * unique_eigs[j]
			cj = cj + kappa[j]
			j = j + 1
		else:
			si = si + kappa[i] * unique_eigs[i]
			ci = ci + kappa[i]
			i = i + 1
	sig = np.maximum(0.0, np.minimum(sig + S, 1.0))
	return sig

# def project_eigs(sig, k):
# 	S = fsolve(lambda s: np.abs(np.sum(np.maximum(0.0, np.minimum(sig + s, 1.0))) - k), 0.0)
# 	sig = np.maximum(0.0, np.minimum(sig + S[0], 1.0))
# 	return sig

# @jit('float64[:](float64[:,:],float64[:])')
# def mvdot(A, b):
# 	c = np.zeros(shape=(len(A)))
# 	for i in range(len(A)):
# 		for j in range(len(b)):
# 			c[i] += A[i][j] * b[j]
# 	return c


def pinv_update(A, Ap, c, d, tol=1e-10):
	v = Ap @ c
	n = Ap.T @ d
	beta = 1.0 + v @ d
	w = c - A @ v
	m = d - A.T @ n
	w_norm = norm(w) ** 2
	m_norm = norm(m) ** 2
	is_zero_w = np.sqrt(w_norm/len(w)) < tol
	is_zero_m = np.sqrt(m_norm/len(m)) < tol
	is_zero_beta = np.abs(beta) < tol
	if not is_zero_w and not is_zero_m:
		G = -np.outer(v, w)/w_norm - np.outer(m, n)/m_norm + beta * np.outer(m, w)/(m_norm * w_norm)
	elif is_zero_w and not is_zero_m and is_zero_beta:
		G = -np.outer(v, Ap.T @ v)/(norm(v) ** 2) - np.outer(m, n)/m_norm
	elif is_zero_w and not is_zero_beta:
		v_norm = norm(v) ** 2
		G = np.outer(m, Ap.T @ v)/beta - beta/(beta ** 2 + m_norm * v_norm) * np.outer(v_norm/beta * m + v, m_norm/beta * Ap.T @ v + n)
	elif not is_zero_w and is_zero_m and is_zero_beta:
		G = -np.outer(Ap @ n, n)/(norm(n) ** 2) - np.outer(v, w)/w_norm
	elif is_zero_m and not is_zero_beta:
		n_norm = norm(n) ** 2
		G = np.outer(Ap @ n, w)/beta - beta/(beta ** 2 + w_norm * n_norm) * np.outer(w_norm/beta * (Ap @ n) + v, n_norm/beta * w + n)
	elif is_zero_w and is_zero_m and is_zero_beta:
		v_norm = norm(v) ** 2
		n_norm = norm(n) ** 2
		G = -np.outer(v, Ap.T @ v)/v_norm - np.outer(Ap @ n, n)/n_norm + np.dot(v, Ap @ n)/(v_norm * n_norm) * np.outer(v, n)
	return Ap + G

def oja(C, y, eta):
	k = C.shape[1]
	C, _ = qr_update(C, np.eye(k), eta * y, C.T @ y)
	return C

def krasulina(C, y, eta):
	k = C.shape[1]
	x = C.T @ y
	C, _ = qr_update(C, np.eye(k), eta * (y - C @ x), x)
	return C

def implicit_krasulina_pinv(C, Cp, y, eta):
	mu = Cp @ y
	mu2 = norm(mu) ** 2
	r = y - C @ mu
	Cp = pinv_update(C, Cp, eta/(1 + eta * mu2) * r, mu)
	C = C + eta/(1 + eta * mu2) * np.outer(r, mu)
	return C, Cp

def implicit_krasulina(C, Cp, y, eta):
	yc = C.T @ y
	mu = Cp @ yc
	mu2 = norm(mu) ** 2
	etax = eta/(1 + eta * mu2)
	r = etax * (y - C @ mu)
	Cr = C.T @ r
	C = C + np.outer(r, mu)
	U = np.vstack((Cr + (r @ r) * mu, mu))
	V = np.vstack((mu, Cr))
	VCp = V @ Cp
	Cp = Cp - ((Cp @ U.T) @ inv(np.eye(2) + VCp @ U.T)) @ VCp
	# Cp = inv(np.dot(C.T, C))
	return C, Cp


def sanger(C, Cp, y, eta):
	yc = C.T @ y
	mu = Cp @ yc
	r = eta * (y - C @ mu)
	Cr = C.T @ r
	C = C + np.outer(r, mu)
	# U = np.vstack((Cr + np.dot(r, r) * mu, mu))
	# V = np.vstack((mu, Cr))
	# Cp = Cp - np.dot(np.dot(np.dot(Cp, U.T), inv(np.eye(2) + np.dot(np.dot(V, Cp), U.T))), np.dot(V, Cp))
	Cp = inv(C.T @ C)
	return C, Cp


def uncentered_implicit_krasulina(C, Cp, y, m, eta):
	x = dot(Cp, dot(C.T, y - m))
	x2 = norm(x) ** 2
	yh = dot(C, x)
	# m = (m + eta * y)/(1 + eta)
	r = eta/(1 + eta * x2) * (y - m - yh)
	C = C + np.outer(r, x)
	Cr = np.dot(C.T, r)
	U = np.vstack((Cr + np.dot(r, r) * x, x))
	V = np.vstack((x, Cr))
	Cp = Cp - np.dot(np.dot(np.dot(Cp, U.T), inv(np.eye(2) + np.dot(np.dot(V, Cp), U.T))), np.dot(V, Cp))
	return C, Cp, m

def capped_msg(U, sig, y, eta, k, m):
	x = np.sqrt(eta) * dot(U.T, y)
	x_orth = (np.sqrt(eta) * y - dot(U, x))[:, np.newaxis]
	r = norm(x_orth)
	if r > 0:
		update = np.vstack((np.hstack((np.diag(sig) + np.outer(x, x), r * x[:, np.newaxis])), np.concatenate((r * x, [r ** 2]))))
		sig, V = eig(update)
		sig = np.real(sig)
		V = np.real(V)
		U = dot(np.hstack((U, x_orth/r)), V)
	else:
		sig, V = eig(np.diag(sig) + np.outer(x, x))
		sig = np.real(sig)
		V = np.real(V)
		U = dot(U, V)
	U = np.real(U)
	sig = np.real(sig)
	idx = np.argsort(-sig)
	sig = sig[:(k+m)]
	U = U[:,:(k+m)]
	sig = project_eigs(sig, k)
	return U, sig

def incremental(U, sig, y):
	k = U.shape[1]
	x = dot(U.T, y)
	yo = (y - dot(U, x))[:,np.newaxis]
	D = np.diag(sig) + np.outer(x,x)
	yo_norm = norm(yo)
	Q = np.hstack((D, yo_norm * x[:,np.newaxis]))
	Q = np.vstack((Q, np.append(yo_norm * x, yo_norm ** 2)))
	sig, V = eig(Q)
	idx = np.argsort(-sig)
	sig = np.real(sig[idx[:k]])
	U = np.hstack((U, yo/yo_norm))
	U = np.real(np.dot(U, V)[:,idx[:k]])
	return U, sig


def tune_params(Y, k, eta_vals):
	dim, N = Y.shape
	C_init = 0.00001 * np.random.normal(size=(dim,k))
	eta_opt = []
	for method in ['oja', 'krasulina', 'implicit_krasulina', 'capped_msg', 'sanger']:
		errors = np.zeros((len(eta_vals)))
		if method == 'oja':
			a = 0.9
			for i, eta0 in enumerate(eta_vals):
				C, _ = qr(C_init, mode='economic')
				for nn in range(N):
					eta = eta0/((nn+1) ** a)
					C = oja(C, Y[:,nn], eta)
				X = dot(C.T, Y)
				errors[i] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
			eta_opt.append(eta_vals[np.argmin(errors)])
		elif method == 'krasulina':
			a = 0.9
			for i, eta0 in enumerate(eta_vals):
				C, _ = qr(C_init, mode='economic')
				for nn in range(N):
					eta = eta0/((nn+1) ** a)
					C = krasulina(C, Y[:,nn], eta)
				X = dot(C.T, Y)
				errors[i] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
			eta_opt.append(eta_vals[np.argmin(errors)])
		elif method == 'implicit_krasulina':
			a = 0.8
			for i, eta0 in enumerate(eta_vals):
				C = C_init.copy()
				Cp = inv(np.dot(C.T, C))
				for nn in range(N):
					eta = eta0/((nn+1) ** a)
					C, Cp = implicit_krasulina(C, Cp, Y[:,nn], eta)
				X = dot(Cp, np.dot(C.T, Y))
				errors[i] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
			eta_opt.append(eta_vals[np.argmin(errors)])
		elif method == 'capped_msg':
			# a = 0.5
			# C = np.random.normal(size=(dim,k+1))
			# C = dot(C, C.T)
			# sig_init, U_init = eigs(C, k=k+1)
			# sig_init = project_eigs(sig_init, k)
			# for i, eta0 in enumerate(eta_vals):
			# 	U = U_init.copy()
			# 	sig = sig_init.copy()
			# 	for nn in range(N):
			# 		eta = eta0/((nn+1) ** a)
			# 		U, sig = capped_msg(U, sig, Y[:, nn], eta, k, 1)
			# 	U, _ = qr(U, mode='economic')
			# 	C = np.dot(U, U.T)
			# 	errors[i] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/N
			# eta_opt.append(eta_vals[np.argmin(errors)])
			eta_opt.append(0)
		elif method == 'sanger':
			a = 0.9
			for i, eta0 in enumerate(eta_vals):
				C = C_init.copy()
				Cp = inv(np.dot(C.T, C))
				for nn in range(N):
					eta = eta0/((nn+1) ** a)
					C, Cp = sanger(C, Cp, Y[:,nn], eta)
				X = dot(Cp, np.dot(C.T, Y))
				errors[i] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
			eta_opt.append(eta_vals[np.argmin(errors)])
	return eta_opt

def tune_params_sanger(Y, k, eta_vals):
	dim, N = Y.shape
	C_init = 0.0001 * np.random.normal(size=(dim,k))
	errors = np.zeros((len(eta_vals)))
	a = 0.8
	for i, eta0 in enumerate(eta_vals):
		C = C_init.copy()
		Cp = inv(C.T @ C)
		for nn in range(N):
			eta = eta0/((nn+1) ** a)
			C, Cp = sanger(C, Cp, Y[:,nn], eta)
		X = Cp @ (C.T @ Y)
		errors[i] = np.mean(np.sum((Y - C @ X)**2, axis=0))
	eta_opt = eta_vals[np.argmin(errors)]
	return eta_opt

def tune_params_implicit_krasulina(Y, k, eta_vals):
	dim, N = Y.shape
	C_init = 0.0001 * np.random.normal(size=(dim,k))
	errors = np.zeros((len(eta_vals)))
	a = 0.8
	for i, eta0 in enumerate(eta_vals):
		C = C_init.copy()
		Cp = inv(np.dot(C.T, C))
		for nn in range(N):
			eta = eta0/((nn+1) ** a)
			C, Cp = implicit_krasulina(C, Cp, Y[:,nn], eta)
		X = Cp @ (C.T @ Y)
		errors[i] = np.mean(np.sum((Y - C @ X)**2, axis=0))
	eta_opt = eta_vals[np.argmin(errors)]
	return eta_opt
