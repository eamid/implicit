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
	mu_batch = np.mean(data,axis=0)

	# data = data[:10000,:]
	data = data[np.random.permutation(data.shape[0]),:]
	num_valid = int(data.shape[0] * 0.1)
	Y_valid = data[:num_valid, :].T - mu_batch[:,np.newaxis]
	# data = data[num_valid+1:, :]
	N, dim = data.shape


	num_iter = 1
	k_values = [5, 10, 20, 50]
	err_iters = np.concatenate(([1], np.r_[0:(N+1):100])[1:]).astype(np.int32)
	error_all = np.zeros((6, len(k_values), len(err_iters), num_iter))
	error_mean = np.zeros((2, len(k_values), len(err_iters), num_iter))
	time_all = np.zeros((6, len(k_values),  len(err_iters), num_iter))

	mu_inint = 0.01 * np.random.normal(size=(dim))
	for i, k in enumerate(k_values):
		eta_vals = [0.1, 1.0, 2.0, 10.0, 100.0, 1000.0, 100.0 * k, 1000.0 * k]
		eta0_oja, eta0_krasulina, eta0_imp_krasulina, eta0_msg, eta0_sanger = tune_params(Y_valid, k, eta_vals)

		# # PCA
		# Y = (data.copy()).T
		# Y -= mu_batch[:, np.newaxis]
		# t = time.clock()
		# pca = PCA(n_components = k)
		# pca.fit(Y.T)
		# time_all[0,i,:,:] = time.clock() - t
		# C = pca.components_.T
		# X = np.dot(C.T, Y)
		# error_all[0,i,:,:] = np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))
		C_init = 0.1 * np.random.normal(size=(dim,k))
		# print("batch PCA error %f" % (np.mean(np.sum((Y - np.dot(C,X))**2, axis=0))))

		for itr in range(num_iter):
			Y = (data.copy())[np.random.permutation(N),:].T

   #          # Oja's
			# eta0 = eta0_oja
			# a = 0.9
			# elapsed = 0.0
			# t = time.clock()
			# C, _ = qr(C_init, mode='economic')
			# mu = mu_inint.copy()
			# elapsed += time.clock() - t
			# err_idx = 0
			# for nn in range(N):
			# 	t = time.clock()
			# 	eta = eta0/((nn+1) ** a)
			# 	C = oja(C, Y[:,nn] - mu, eta)
			# 	mu = (nn * mu + Y[:,nn])/(nn+1)
			# 	elapsed += time.clock() - t
			# 	if (nn+1) in err_iters:
			# 		time_all[1,i,err_idx,itr] = elapsed
			# 		X = dot(C.T, Y - mu[:,np.newaxis])
			# 		error_all[1,i,err_idx,itr] = np.mean(np.sum((Y - mu[:,np.newaxis] - dot(C,X))**2, axis=0))
			# 		error_mean[0,i,err_idx,itr] = np.sqrt(np.sum((mu - mu_batch) ** 2))
			# 		err_idx += 1
			# 		print("Oja iter %d, error %f, error mean %f" % (nn+1, error_all[1,i,err_idx-1,itr], error_mean[0,i,err_idx-1,itr]))
            
   #          # Krasulina's
			# eta0 = eta0_krasulina
			# a = 0.5
			# elapsed = 0.0
			# t = time.clock()
			# C, _ = qr(C_init, mode='economic')
			# mu = mu_inint.copy()
			# elapsed += time.clock() - t
			# err_idx = 0
			# for nn in range(N):
			# 	t = time.clock()
			# 	eta = eta0/((nn+1) ** a)
			# 	C = krasulina(C, Y[:,nn] - mu, eta)
			# 	mu = (nn * mu + Y[:,nn])/(nn+1)
			# mu = (nn * mu + Y[:,nn])/(nn+1)
			# 	elapsed += time.clock() - t
			# 	if (nn+1) in err_iters:
			# 		time_all[2,i,err_idx,itr] = elapsed
			# 		X = dot(C.T, Y - mu[:,np.newaxis])
			# 		error_all[2,i,err_idx,itr] = np.mean(np.sum((Y - mu[:,np.newaxis] - dot(C,X))**2, axis=0))
			# 		err_idx += 1
			# 		print("Krasulina iter %d, error %f" % (nn+1, error_all[2,i,err_idx-1,itr]))


   #          # implicit Krasulina's
			# eta0 = eta0_imp_krasulina
			# eta0_mean = 10.0
			# a_mean = 0.5
			# a = 0.9
			# elapsed = 0.0
			# t = time.clock()
			# C = C_init.copy()
			# Cp = inv(dot(C.T, C))
			# mu = mu_inint.copy()
			# elapsed += time.clock() - t
			# err_idx = 0
			# for nn in range(N):
			# 	t = time.clock()
			# 	eta = eta0/((nn+1) ** a)
			# 	# eta_mean = eta0_mean/((nn+1) ** a_mean)
			# 	# mu = (mu + eta_mean * (Y[:,nn] - dot(C, dot(Cp, dot(C.T, Y[:,nn]-mu)))))/(1 + eta_mean)
			# 	mu = (nn * mu + Y[:,nn])/(nn+1)
			# 	C, Cp = implicit_krasulina(C, Cp, Y[:,nn]-mu, eta)
			# 	# mu = (nn * mu + Y[:,nn])/(nn+1)
			# 	# x = dot(Cp, dot(C.T, Y[:,nn] - mu))
			# 	# x2 = norm(x) ** 2
			# 	# mu = (1 + eta * x2)/(1 + eta * (1 + x2)) * (mu + eta * (Y[:,nn]/(1 + eta * x2) - dot(C, x)))
			# 	# yh = dot(C, x)
			# 	# r = eta/(1 + eta * x2) * (Y[:,nn] - mu - yh)
			# 	# C = C + np.outer(r, x)
			# 	# Cr = np.dot(C.T, r)
			# 	# U = np.vstack((Cr + np.dot(r, r) * x, x))
			# 	# V = np.vstack((x, Cr))
			# 	# Cp = Cp - np.dot(np.dot(np.dot(Cp, U.T), inv(np.eye(2) + np.dot(np.dot(V, Cp), U.T))), np.dot(V, Cp))

			# 	# C, Cp, mu = uncentered_implicit_krasulina(C, Cp, Y[:,nn], mu, eta)
			# 	elapsed += time.clock() - t
			# 	if (nn+1) in err_iters:
			# 		time_all[3,i,err_idx,itr] = elapsed
			# 		X = dot(Cp, dot(C.T, Y - mu[:,np.newaxis]))
			# 		error_all[3,i,err_idx,itr] = np.mean(np.sum((Y - mu[:,np.newaxis] - dot(C,X))**2, axis=0))
			# 		error_mean[1,i,err_idx,itr] = np.sqrt(np.sum((mu - mu_batch) ** 2))
			# 		err_idx += 1
			# 		print("Imp. Krasulina iter %d, error %f, error mean %f" % (nn+1, error_all[3,i,err_idx-1,itr], error_mean[1,i,err_idx-1,itr]))

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
			# elapsed = 0.0
			# err_idx = 0
			# t = time.clock()
			# sig, U = eigs(np.dot(C_init, C_init.T), k=k)
			# mu = mu_inint.copy()
			# elapsed += time.clock() - t
			# for nn in range(N):
			# 	t = time.clock()
			# 	U, sig = incremental(U, sig, Y[:,nn] - mu)
			# 	mu = (nn * mu + Y[:,nn])/(nn+1)
			# 	elapsed += time.clock() - t
			# 	if (nn+1) in err_iters:
			# 		time_all[5,i,err_idx,itr] = elapsed
			# 		U, _ = qr(U, mode='economic')
			# 		C = np.dot(U, U.T)
			# 		error_all[5,i,err_idx,itr] = np.trace(np.dot(np.eye(Y.shape[0]) - C, np.dot((Y-mu[:,np.newaxis]), (Y-mu[:,np.newaxis]).T)))/Y.shape[1]
			# 		err_idx += 1
			# print("Incremental error %f" % error_all[5,i,err_idx-1,itr])
			# implicit Krasulina's
			eta0 = eta0_sanger
			a = 0.9
			elapsed = 0.0
			t = time.clock()
			C = C_init.copy()
			Cp = inv(dot(C.T, C))
			mu = mu_inint.copy()
			elapsed += time.clock() - t
			err_idx = 0
			for nn in range(N):
				t = time.clock()
				eta = eta0/((nn+1) ** a)
				mu = (nn * mu + Y[:,nn])/(nn+1)
				C, Cp = sanger(C, Cp, Y[:,nn]-mu, eta)
				elapsed += time.clock() - t
				if (nn+1) in err_iters:
					time_all[3,i,err_idx,itr] = elapsed
					X = dot(Cp, dot(C.T, Y - mu[:,np.newaxis]))
					error_all[3,i,err_idx,itr] = np.mean(np.sum((Y - mu[:,np.newaxis] - dot(C,X))**2, axis=0))
					error_mean[1,i,err_idx,itr] = np.sqrt(np.sum((mu - mu_batch) ** 2))
					err_idx += 1
					print("Imp. Krasulina iter %d, error %f, error mean %f" % (nn+1, error_all[3,i,err_idx-1,itr], error_mean[1,i,err_idx-1,itr]))

			print('iteration = %d, k = %d \n' % (itr+1, k))            
		sio.savemat('../results/rerun/centered_pca_results_' +  filename + '.mat', {'error':error_all, 'runtime':time_all})

if __name__ == "__main__":
	main()

