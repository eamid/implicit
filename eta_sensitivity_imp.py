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

	k = 5

	mat = sio.loadmat('../results/eta_sensitivity_results_' +  filename + '_k_'+ str(k) + '.mat')
	error_all = mat['error_all']
	error_pca = mat['error_pca']


	num_iter = 10
	err_iters = np.r_[0:(N+1):5000][1:].astype(np.int32)

	eta_vals = [0.1, 1.0, 2.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
	eta0_oja, eta0_krasulina, eta0_imp_krasulina, eta0_msg = tune_params(Y_valid, k, eta_vals)
	print(eta0_imp_krasulina)
	for itr in range(num_iter):
			Y = data[np.random.permutation(N),:].T
			C_init = 0.00001 * np.random.normal(size=(dim,k))

			# implicit Krasulina's
			for i, factor in enumerate([0.1, 1.0, 10.0]):
				eta0 = eta0_imp_krasulina * factor
				a = 0.8
				C = C_init.copy()
				Cp = inv(np.dot(C.T, C))
				err_idx = 0
				X = dot(Cp, np.dot(C.T, Y))
				error_all[2,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
				err_idx += 1
				for nn in range(N):
					eta = eta0/((nn+1) ** a)
					C, Cp = implicit_krasulina(C, Cp, Y[:,nn], eta)
					if (nn+1) in err_iters:
						X = dot(Cp, np.dot(C.T, Y))
						error_all[2,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
						err_idx += 1
				print("Imp Krasulina error %f" % error_all[2,i,err_idx-1,itr])
			sio.savemat('../results/eta_sensitivity_results_' +  filename + '_k_'+ str(k) + '.mat', {'error_all':error_all, 'error_pca':error_pca})

if __name__ == "__main__":
	main()
