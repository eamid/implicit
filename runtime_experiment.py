import scipy.io as sio
from scipy.linalg import qr
from sklearn.decomposition import PCA
from numpy import dot as dot
import time
import sys
from online_pca_methods_jit import *


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


	num_iter = 1
	k_values = [5, 10, 20, 50]

	
	for i, k in enumerate(k_values):
		eta_vals = [0.1, 1.0, 2.0, 10.0, 100.0, 1000.0, 100.0 * k, 1000.0 * k]
		eta0_oja, eta0_krasulina, eta0_imp_krasulina, eta0_msg = tune_params(Y_valid, k, eta_vals)

		Y = data.T
		# PCA
		t = time.clock()
		pca = PCA(n_components = k)
		pca.fit(Y.T)
		elapsed = time.clock() - t
		C = pca.components_.T
		X = np.dot(C.T, Y)
		print("batch PCA time %f" % elapsed)
		for itr in range(num_iter):
			C_init = 0.1 * np.random.normal(size=(dim,k))
			Y = data[np.random.permutation(N),:].T

            # Oja's
			eta0 = eta0_oja
			a = 0.9
			elapsed = 0.0
			t = time.clock()
			C, _ = qr(C_init.copy(), mode='economic')
			elapsed += time.clock() - t
			for nn in range(N):
				t = time.clock()
				eta = eta0/((nn+1) ** a)
				C = oja(C, Y[:,nn], eta)
				elapsed += time.clock() - t
			print("Oja time %f" % elapsed)
            
            # Krasulina's
			eta0 = eta0_krasulina
			a = 0.5
			elapsed = 0.0
			t = time.clock()
			C, _ = qr(C_init.copy(), mode='economic')
			elapsed += time.clock() - t
			for nn in range(N):
				t = time.clock()
				eta = eta0/((nn+1) ** a)
				C = krasulina(C, Y[:,nn], eta)
				elapsed += time.clock() - t
			print("Krasulina time %f" % elapsed)

            # implicit Krasulina's
			eta0 = eta0_imp_krasulina
			a = 0.9
			elapsed = 0.0
			t = time.clock()
			C = C_init.copy()
			Cp = pinv(C)
			elapsed += time.clock() - t
			for nn in range(N):
				t = time.clock()
				eta = eta0/((nn+1) ** a)
				implicit_krasulina_pinv(C, Cp, Y[:,nn], eta)
				elapsed += time.clock() - t
			print("Imp Krasulina time %f" % elapsed)

            # Incremental
			elapsed = 0.0
			err_idx = 0
			t = time.clock()
			sig, U = eigs(np.dot(C_init, C_init.T), k=k)
			elapsed += time.clock() - t
			for nn in range(N):
				t = time.clock()
				U, sig = incremental(U, sig, Y[:,nn])
				elapsed += time.clock() - t
			print("Incremental time %f" % elapsed)

if __name__ == "__main__":
	main()

