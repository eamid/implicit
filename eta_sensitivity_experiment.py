import scipy.io as sio
from scipy.linalg import qr
from sklearn.decomposition import PCA
from numpy import dot as dot
import time
import sys
from online_pca_methods import *


def main():
	filename = sys.argv[1]
	mat = sio.loadmat('../data/' + filename + '.mat')
	data = mat['X'].astype(np.float64)
	data = data
	data /= np.max(data)
	data -= np.mean(data,axis=0)

	data = data[np.random.permutation(data.shape[0]),:]
	num_valid = int(data.shape[0] * 0.1)
	Y_valid = data[:num_valid, :].T
	# data = data[num_valid+1:, :]
	N, dim = data.shape

	num_iter = 10
	k = 5
	# err_iters = np.r_[0:(N+1):5000][1:].astype(np.int32)
	err_iters = [N]
	# err_iters = np.concatenate(([1], np.r_[0:(N+1):100])[1:]).astype(np.int32)
	error_all = np.zeros((4, 3, len(err_iters)+1, num_iter))

	eta_vals = [0.1, 1.0, 2.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 100.0 * k, 1000.0 * k]
	eta0_oja, eta0_krasulina, eta0_imp_krasulina, eta0_msg = tune_params(Y_valid, k, eta_vals)

	# PCA
	Y = data.T
	pca = PCA(n_components = k)
	pca.fit(Y.T)
	C = pca.components_.T
	X = np.dot(C.T, Y)
	error_pca = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
	print("batch PCA error %f" % (error_pca))
	for itr in range(num_iter):
			Y = data[np.random.permutation(N),:].T
			C_init = 0.00001 * np.random.normal(size=(dim,k))
            # Oja's
			for i, factor in enumerate([0.1, 1.0, 10.0]):
				eta0 = eta0_oja * factor
				a = 0.9
				C, _ = qr(C_init.copy(), mode='economic')
				err_idx = 0
				# X = dot(C.T, Y)
				# error_all[0,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
				# err_idx += 1
				for nn in range(N):
					eta = eta0/((nn+1) ** a)
					C = oja(C, Y[:,nn], eta)
					if (nn+1) in err_iters:
						X = dot(C.T, Y)
						error_all[0,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
						err_idx += 1
				print("Oja error %f" % error_all[0,i,err_idx-1,itr])

			# Krasulina's
			for i, factor in enumerate([0.1, 1.0, 10.0]):
				eta0 = eta0_krasulina * factor
				a = 0.9
				C, _ = qr(C_init.copy(), mode='economic')
				err_idx = 0
				# X = dot(C.T, Y)
				# error_all[1,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
				# err_idx += 1
				for nn in range(N):
					eta = eta0/((nn+1) ** a)
					C = krasulina(C, Y[:,nn], eta)
					if (nn+1) in err_iters:
						X = dot(C.T, Y)
						error_all[1,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
						err_idx += 1
				print("Krasulina error %f" % error_all[1,i,err_idx-1,itr])

			# implicit Krasulina's
			for i, factor in enumerate([0.1, 1.0, 10.0]):
				eta0 = eta0_imp_krasulina * factor
				a = 0.8
				C = C_init.copy()
				Cp = inv(np.dot(C.T, C))
				err_idx = 0
				# X = dot(Cp, np.dot(C.T, Y))
				# error_all[2,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
				# err_idx += 1
				for nn in range(N):
					eta = eta0/((nn+1) ** a)
					C, Cp = implicit_krasulina(C, Cp, Y[:,nn], eta)
					if (nn+1) in err_iters:
						X = dot(Cp, np.dot(C.T, Y))
						error_all[2,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
						err_idx += 1
				print("Imp Krasulina error %f" % error_all[2,i,err_idx-1,itr])

			# # Capped MSG
			# eta0 = eta0_msg
			# a = 0.5
			# m = 1;
			# elapsed = 0.0
			# t = time.clock()
			# M = np.random.normal(size=(dim,k+m))
			# M = dot(M, M.T)
			# sig, U = eigs(M, k=k+m)
			# sig = project_eigs(sig, k)
			# elapsed += time.clock() - t
			# err_idx = 0
			# for nn in range(N):
			# 	t = time.clock()
			# 	eta = eta0/((nn+1) ** a)
			# 	U, sig = capped_msg(U, sig, Y[:, nn], eta, k, m)
			# 	elapsed += time.clock() - t
			# 	if (nn+1) in err_iters:
			# 		time_all[4,i,err_idx,itr] = elapsed
			# 		U, _ = qr(U, mode='economic')
			# 		C = np.dot(U, U.T)
			# 		error_all[4,i,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
			# 		err_idx += 1
			# print("Capped MSG error %f" % error_all[4,i,err_idx-1,itr])

			# Incremental
			err_idx = 0
			sig, U = eigs(np.dot(C_init, C_init.T), k=k)
			# U, _ = qr(U, mode='economic')
			# C = np.dot(U, U.T)
			# error_all[3,:,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
			# err_idx += 1
			for nn in range(N):
				U, sig = incremental(U, sig, Y[:,nn])
				if (nn+1) in err_iters:
					U, _ = qr(U, mode='economic')
					C = np.dot(U, U.T)
					error_all[3,:,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
					err_idx += 1
			print("Incremental error %f" % error_all[3,0,err_idx-1,itr])
			sio.savemat('../results/eta_sensitivity_results_' +  filename + '_k_'+ str(k) + '.mat', {'error_all':error_all, 'error_pca':error_pca})
			print("iter %d" % (itr+1))
			print()

if __name__ == "__main__":
	main()
