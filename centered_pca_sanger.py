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


	num_iter = 5
	k_values = [5, 10, 20, 50]
	err_iters = np.r_[0:(N+1):5000][1:].astype(np.int32)
	# err_iters = np.concatenate(([1], np.r_[0:(N+1):100])[1:]).astype(np.int32)
	error_all = np.zeros((1, len(k_values), len(err_iters) + 1, num_iter))
	time_all = np.zeros((1, len(k_values),  len(err_iters) + 1, num_iter))

	
	for i, k in enumerate(k_values):
		eta_vals = [1e-6, 5e-6, 1e-5, 5e-5, 8e-5, 0.0001, 0.0002, 0.0005, 0.0008, 0.001, 00.005, 0.01, 0.002, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
		# eta_vals = [0.1, 1.0, 2.0, 10.0, 100.0, 1000.0]
		eta0_sanger = tune_params_sanger(Y_valid, k, eta_vals)

		Y = data.T
		for itr in range(num_iter):
			Y = data[np.random.permutation(N),:].T
			C_init = 0.0001 * np.random.normal(size=(dim,k))
            
			# Sanger's
			eta0 = eta0_sanger
			a = 0.8
			elapsed = 0.0
			t = time.clock()
			C = C_init.copy()
			Cp = inv(np.dot(C.T, C))
			elapsed += time.clock() - t
			err_idx = 0
			time_all[0,i,err_idx,itr] = elapsed
			X = dot(Cp, np.dot(C.T, Y))
			error_all[0,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
			err_idx += 1
			for nn in range(N):
				t = time.clock()
				eta = eta0/((nn+1) ** a)
				C, Cp = sanger(C, Cp, Y[:,nn], eta)
				elapsed += time.clock() - t
				if (nn+1) in err_iters:
					time_all[0,i,err_idx,itr] = elapsed
					X = dot(Cp, np.dot(C.T, Y))
					error_all[0,i,err_idx,itr] = np.mean(np.sum((Y - dot(C,X))**2, axis=0))
					err_idx += 1
			print("iteration = %d, k = %d, Sanger error %f" % (itr+1, k, error_all[0,i,err_idx-1,itr]))

		sio.savemat('../results/centered_pca_sanger_' +  filename + '.mat', {'error':error_all, 'runtime':time_all})

if __name__ == "__main__":
	main()

