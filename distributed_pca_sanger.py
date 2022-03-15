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

	error_sanger = np.zeros((M+1, len(err_iters)+1, num_iter))

	eta_vals = [1e-6, 5e-6, 1e-5, 5e-5, 8e-5, 0.0001, 0.0002, 0.0005, 0.0008, 0.001, 00.005, 0.01, 0.002, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
	eta0_sanger = tune_params_sanger(Y_valid, k, eta_vals)
	for itr in range(num_iter):
		Y = data[np.random.permutation(N),:].T
		C_init = 0.0001 * np.random.normal(size=(dim,k))

		# Implicit Krasulina
		err_idx = 0
		Cp_init = inv(np.dot(C_init.T, C_init))

		C_all = [C_init.copy() for _ in range(M)]
		Cp_all = [Cp_init.copy() for _ in range(M)]

		X = dot(Cp_init, np.dot(C_init.T, Y))
		error_sanger[:,err_idx,itr] = np.mean(np.sum((Y - dot(C_init,X))**2, axis=0))
		err_idx += 1

		eta0 = eta0_sanger
		a = 0.8
		for nn in range(num_example_per_machine):
			eta = eta0/((nn +1) ** a)
			for mm in range(M):
				C = C_all[mm]
				Cp = Cp_all[mm]
				C, Cp = sanger(C, Cp, Y[:, nn + (mm * num_example_per_machine)], eta)
				C_all[mm] = C
				Cp_all[mm] = Cp
			if (nn+1) in err_iters:
				C = C_all[0].copy()
				Cp = Cp_all[0].copy()
				X = dot(Cp, np.dot(C.T, Y))
				error_sanger[0,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))

				CT = C.copy()
				for mm in range(1, M):
					C = C_all[mm]
					Cp = Cp_all[mm]
					X = dot(Cp, np.dot(C.T, Y))
					error_sanger[mm,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
					CT = CT + C
				C = CT/M
				Cp = inv(dot(C.T, C))
				C_all = [C.copy() for _ in range(M)]
				Cp_all = [Cp.copy() for _ in range(M)]
				X = dot(Cp, np.dot(C.T, Y))
				error_sanger[M,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
				err_idx += 1
				print("Sanger iter %d step %d/%d, k = %d, mean error %f, combined error %f" % (itr+1, nn+1, num_example_per_machine, k, np.mean(error_sanger[:M,err_idx-1,itr]), error_sanger[M,err_idx-1,itr]))

		sio.savemat('../results/distributed_pca_sanger_' +  filename + '_k_' + str(k) + '.mat', {'error_sanger':error_sanger})


if __name__ == "__main__":
	main()
