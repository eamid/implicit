import numpy as np
import scipy.io as sio
from scipy.linalg import qr
from sklearn.decomposition import PCA
from scipy.linalg import inv
from numpy import dot as dot
import sys
from online_pca_methods import *


def main():
	filename = sys.argv[1]
	mat = sio.loadmat('../data/' + filename + '.mat')
	data = mat['X'].astype(np.float64)
	data /= np.max(data)
	data -= np.mean(data,axis=0)
	data = data[np.random.permutation(data.shape[0]),:]
	num_valid = int(data.shape[0] * 0.1)
	Y_valid = data[:num_valid, :].T
	# data = data[num_valid+1:, :]
	N, dim = data.shape
	N, dim = data.shape
	M = 10
	num_example_per_machine = N//M
	num_steps = 1000

	num_iter = 10
	k = 5
	err_iters = np.r_[num_steps:(num_example_per_machine+1):num_steps].astype(np.int32)

	error_oja = np.zeros((M+1, len(err_iters)+1, num_iter))
	error_krasulina = np.zeros((M+1, len(err_iters)+1, num_iter))
	error_capped_msg = np.zeros((M+1, len(err_iters)+1, num_iter))
	error_imp_krasulina = np.zeros((M+1, len(err_iters)+1, num_iter))
	error_incremental = np.zeros((M+1, len(err_iters)+1, num_iter))

	pca = PCA(n_components = k)
	pca.fit(data)
	C = pca.components_.T
	X = np.dot(C.T, data.T)
	error_pca = np.mean(np.sum((data.T - np.dot(C,X))**2, axis=0))
	print("PCA error %f" % (error_pca))

	eta_vals = [0.1, 1.0, 2.0, 10.0, 100.0, 1000.0, 100.0 * k, 1000.0 * k]
	eta0_oja, eta0_krasulina, eta0_imp_krasulina, eta0_msg = tune_params(Y_valid, k, eta_vals)
	for itr in range(num_iter):
		Y = data[np.random.permutation(N),:].T
		C_init = 0.0001 * np.random.normal(size=(dim,k))
		C_init_orth, _ = qr(C_init, mode='economic')

		# Oja
		err_idx = 0
		C_all = [C_init_orth.copy() for _ in range(M)]

		X = dot(C_init_orth.T, Y)
		error_oja[:,err_idx,itr] = np.mean(np.sum((Y - dot(C_init_orth,X))**2, axis=0))
		err_idx += 1
		
		eta0 = eta0_oja
		a = 0.9
		for nn in range(num_example_per_machine):
			eta = eta0/((nn +1) ** a)
			for mm in range(M):
				C = C_all[mm]
				C = oja(C, Y[:, nn + (mm * num_example_per_machine)], eta)
				C_all[mm] = C
			if (nn+1) in err_iters:
				C = C_all[0].copy()
				X = dot(C.T, Y)
				error_oja[0,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))

				CT = C.copy()
				for mm in range(1, M):
					C = C_all[mm]
					X = dot(C.T, Y)
					error_oja[mm,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
					CT = CT + C
				C = CT/M
				C, _ = qr(C, mode='economic')
				C_all = [C.copy() for _ in range(M)]
				X = dot(C.T, Y)
				error_oja[M,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
				err_idx += 1
				print("Oja step %d/%d, k = %d, mean error %f, combined error %f" % (nn+1, num_example_per_machine, k, np.mean(error_oja[:M,err_idx-1,itr]), error_oja[M,err_idx-1,itr]))

		# Krasulina
		err_idx = 0
		C_all = [C_init_orth.copy() for _ in range(M)]

		X = dot(C_init_orth.T, Y)
		error_krasulina[:,err_idx,itr] = np.mean(np.sum((Y - dot(C_init_orth,X))**2, axis=0))
		err_idx += 1
		
		eta0 = eta0_krasulina
		a = 0.9
		for nn in range(num_example_per_machine):
			eta = eta0/((nn +1) ** a)
			for mm in range(M):
				C = C_all[mm]
				C = krasulina(C, Y[:, nn + (mm * num_example_per_machine)], eta)
				C_all[mm] = C
			if (nn+1) in err_iters:
				C = C_all[0].copy()
				X = dot(C.T, Y)
				error_krasulina[0,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))

				CT = C.copy()
				for mm in range(1, M):
					C = C_all[mm]
					X = dot(C.T, Y)
					error_krasulina[mm,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
					CT = CT + C
				C = CT/M
				C, _ = qr(C, mode='economic')
				C_all = [C.copy() for _ in range(M)]
				X = dot(C.T, Y)
				error_krasulina[M,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
				err_idx += 1
				print("Krasulina step %d/%d, k = %d, mean error %f, combined error %f" % (nn+1, num_example_per_machine, k, np.mean(error_krasulina[:M,err_idx-1,itr]), error_krasulina[M,err_idx-1,itr]))

		# Incremental
		err_idx = 0
		C = np.random.normal(size=(dim,k+1))
		C = dot(C, C.T)
		sig, U = eigs(C, k=k)

		U_all = [U.copy() for _ in range(M)]
		sig_all = [sig.copy() for _ in range(M)]

		U, _ = qr(U, mode='economic')
		C = np.dot(U, U.T)
		error_incremental[:,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
		err_idx += 1
		
		for nn in range(num_example_per_machine):
			eta = eta0/((nn +1) ** a)
			for mm in range(M):
				U = U_all[mm]
				sig = sig_all[mm]
				U, sig = incremental(U, sig, Y[:, nn + (mm * num_example_per_machine)])
				U_all[mm] = U
				sig_all[mm]= sig
			if (nn+1) in err_iters:
				U = U_all[0].copy()
				U, _ = qr(U, mode='economic')
				C = np.dot(U, U.T)
				error_incremental[0,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]

				UT = U.copy()
				for mm in range(1, M):
					U = U_all[mm].copy()
					U, _ = qr(U, mode='economic')
					C = np.dot(U, U.T)
					error_incremental[mm,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
					UT = UT + U
				U = UT/M
				U, _ = qr(U, mode='economic')
				C = np.dot(U, U.T)
				sig, U = eigs(C, k=k)
				U_all = [U.copy() for _ in range(M)]
				sig_all = [sig.copy() for _ in range(M)]
				error_incremental[M,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
				err_idx += 1
				print("Incremental step %d/%d, k = %d, mean error %f, combined error %f" % (nn+1, num_example_per_machine, k, np.mean(error_incremental[:M,err_idx-1,itr]), error_incremental[M,err_idx-1,itr]))

		# # Capped MSG
		# err_idx = 0
		# C = np.random.normal(size=(dim,k+1))
		# C = dot(C, C.T)
		# sig, U = eigs(C, k=k+1)
		# sig = project_eigs(sig, k)

		# U_all = [U.copy() for _ in range(M)]
		# sig_all = [sig.copy() for _ in range(M)]

		# U, _ = qr(U, mode='economic')
		# C = np.dot(U, U.T)
		# error_capped_msg[:,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
		# err_idx += 1
		
		# eta0 = 1.0
		# a = 0.5
		# for nn in range(num_example_per_machine):
		# 	eta = eta0/((nn +1) ** a)
		# 	for mm in range(M):
		# 		U = U_all[mm]
		# 		sig = sig_all[mm]
		# 		U, sig = capped_msg(U, sig, Y[:, nn + (mm * num_example_per_machine)], eta, k, 1)
		# 		U_all[mm] = U
		# 		sig_all[mm]= sig
		# 	if (nn+1) in err_iters:
		# 		U = U_all[0].copy()
		# 		U, _ = qr(U, mode='economic')
		# 		C = np.dot(U, U.T)
		# 		error_capped_msg[0,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]

		# 		UT = U.copy()
		# 		for mm in range(1, M):
		# 			U = U_all[mm].copy()
		# 			U, _ = qr(U, mode='economic')
		# 			C = np.dot(U, U.T)
		# 			error_capped_msg[mm,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
		# 			UT = UT + U
		# 		U = UT/M
		# 		U, _ = qr(U, mode='economic')
		# 		C = np.dot(U, U.T)
		# 		sig, U = eigs(C, k=k+1)
		# 		sig = project_eigs(sig, k)
		# 		U_all = [U.copy() for _ in range(M)]
		# 		sig_all = [sig.copy() for _ in range(M)]
		# 		error_capped_msg[M,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot(Y, Y.T)))/Y.shape[1]
		# 		err_idx += 1
		# 		print("Capped MSG step %d/%d, k = %d, mean error %f, combined error %f" % (nn+1, num_example_per_machine, k, np.mean(error_capped_msg[:M,err_idx-1,itr]), error_capped_msg[M,err_idx-1,itr]))



		# Implicit Krasulina
		err_idx = 0
		Cp_init = inv(np.dot(C_init.T, C_init))

		C_all = [C_init.copy() for _ in range(M)]
		Cp_all = [Cp_init.copy() for _ in range(M)]

		X = dot(Cp_init, np.dot(C_init.T, Y))
		error_imp_krasulina[:,err_idx,itr] = np.mean(np.sum((Y - dot(C_init,X))**2, axis=0))
		err_idx += 1

		eta0 = eta0_imp_krasulina
		a = 0.9
		for nn in range(num_example_per_machine):
			eta = eta0/((nn +1) ** a)
			for mm in range(M):
				C = C_all[mm]
				Cp = Cp_all[mm]
				C, Cp = implicit_krasulina(C, Cp, Y[:, nn + (mm * num_example_per_machine)], eta)
				C_all[mm] = C
				Cp_all[mm] = Cp
			if (nn+1) in err_iters:
				C = C_all[0].copy()
				Cp = Cp_all[0].copy()
				X = dot(Cp, np.dot(C.T, Y))
				error_imp_krasulina[0,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))

				CT = C.copy()
				for mm in range(1, M):
					C = C_all[mm]
					Cp = Cp_all[mm]
					X = dot(Cp, np.dot(C.T, Y))
					error_imp_krasulina[mm,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
					CT = CT + C
				C = CT/M
				Cp = inv(dot(C.T, C))
				C_all = [C.copy() for _ in range(M)]
				Cp_all = [Cp.copy() for _ in range(M)]
				X = dot(Cp, np.dot(C.T, Y))
				error_imp_krasulina[M,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
				err_idx += 1
				print("Imp. Krasulina step %d/%d, k = %d, mean error %f, combined error %f" % (nn+1, num_example_per_machine, k, np.mean(error_imp_krasulina[:M,err_idx-1,itr]), error_imp_krasulina[M,err_idx-1,itr]))

		sio.savemat('../results/distributed_pca_results_' +  filename + '_k_' + str(k) + '.mat', {'error_oja':error_oja, 'error_krasulina':error_krasulina, 'error_imp_krasulina':error_imp_krasulina, 'error_incremental':error_incremental, 'error_capped_msg':error_capped_msg, 'error_pca':error_pca})


if __name__ == "__main__":
	main()
