import math
import numpy as np
import random
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import math
import time

def obj_fun(w, X, y, C, z, l):	#Returns the objective function value for given w
	nll = 0
	reg = 0
	i = 0
	while i < l:
		yz = y[i]*z[i]
		if yz >= 0:
			nll += C * math.log(1 + math.exp(-yz))
		else:
			nll += C * (-yz + math.log(1 + math.exp(yz)))
		i += 1

	return nll + (np.dot(w,w)*1.0)/2

def grad_f_D(l, w, X, y, C, z):	#Computes gradient of f into g, and also D and returns them
	sigma = np.ndarray(l)
	D = np.ndarray(l)
	i = 0
	while i < l:
		sigma[i] = 1.0 / (1 + math.exp(-y[i] * z[i]))
		D[i] = sigma[i] * (1 - sigma[i])
		sigma[i] = C * (sigma[i] - 1) * y[i]
		i += 1

	g = sigma * X
	g = w + g
	del sigma
	return (g,D)

def H_d(X, D, C, d):	#returns Hessian*d
	wa = X * d
	wa = C * np.multiply(D, wa)
	return wa * X + d

def cgp(delta, g, D, C, X, n): #Conjugate Gradient procedure for approximately solving the trust region sub-problem, finds s and r
	zai = 0.1 	#Page 635
	tol = zai * np.linalg.norm(g,2)
	i = 0
	
	s = np.zeros(n)
	r = np.copy(-g)
	d = np.copy(r)

	rTr = np.dot(r,r)
	
	while 1:
		if np.linalg.norm(r,2) <= tol:
			break
		Hd = H_d(X, D, C, d)
		alpha = (rTr*1.0)/np.dot(d, Hd)
		s = s + alpha * d
		if np.linalg.norm(s,2) > delta:
			s = s - alpha * d
			std = np.dot(s, d)
			sts = np.dot(s, s)
			dtd = np.dot(d, d)
			delta_sq = delta * delta
			rad = math.sqrt(std * std + dtd * (delta_sq-sts))
			if std >= 0:
				alpha = ((delta_sq - sts)*1.0)/(std + rad)
			else:
				alpha = ((rad - std)*1.0)/dtd
			s = s + alpha * d
			r = r - alpha * Hd
			break
		r = r - alpha * Hd
		rnewTrnew = np.dot(r, r)
		beta = (rnewTrnew*1.0)/rTr
		d = beta * d
		d = d + r
		rTr = rnewTrnew
	del d
	del Hd
	return (s,r)


def tra(X, y, l, n, C):	#Returns w learnt on training dataset
	eta0 = 0.0001	#Page 635
	eta1 = 0.25	#Page 635
	eta2 = 0.75	#Page 635
	sigma1 = 0.25	#Page 635
	sigma2 = 0.5 #Page 635
	sigma3 = 4	#Page 635
	eps = 0.001 #Page 639

	flag = 1
	w = np.zeros(n)
	z = X*w
	f = obj_fun(w, X, y, C, z, l)
	g,D = grad_f_D(l, w, X, y, C, z)

	if np.linalg.norm(g, np.inf) < eps:
		flag = 0

	delta = np.linalg.norm(g,2)	#Page 635

	max_iter = 1000
	k = 0
	if flag == 1:
		while k < max_iter:
			if np.linalg.norm(g, np.inf) < eps:
				break

			s,r = cgp(delta, g, D, C, X, n)
			w_new = np.copy(w)
			w_new = w_new + s

			gs = np.dot(g, s)
			deno = -0.5 * (gs - np.dot(s, r))

			z = X * w_new
			fnew = obj_fun(w_new, X, y, C, z, l)
			nume = f - fnew
			snorm = np.linalg.norm(s,2)
			if k == 1:
				delta = min(delta, snorm)

			if fnew - f - gs <= 0:
				alpha = sigma3
			else:
				alpha = max(sigma1, -0.5 * (gs / (fnew - f - gs)));

			if nume < eta0 * deno:
				delta = min(max(alpha, sigma1) * snorm, sigma2 * delta)
			elif (nume < eta1 * deno):
				delta = max(sigma1 * delta, min(alpha * snorm, sigma2 * delta))
			elif (nume < eta2 * deno):
				delta = max(sigma1 * delta, min(alpha * snorm, sigma3 * delta))
			else:
				delta = max(delta, min(alpha * snorm, sigma3 * delta))
				      
			if nume > eta0 * deno:
				k += 1
				w = np.copy(w_new)
				f = fnew
				g,D = grad_f_D(l, w, X, y, C, z)
			del w_new
	del g
	del D
	return w	# We don't need to delete w as garbage collected would automatically reclaim the memory after w goes out of scope

def accuracy(w, X, l, n, y):
	z = X * w
	i = 0
	accu = 0
	while i < l:
		if z[i] > 0:
			if 1.0/(1 + math.exp(-z[i])) > 0.5 and y[i] == 1:
				accu += 1
		else:
			if 1.0/(1 + math.exp(z[i])) > 0.5 and y[i] == -1:
				accu += 1
		i += 1
	return (100*accu*1.0)/l

def CV(C, fileName):	#For dividing dataset for crossvalidation
	cv_folds = 5
	accu = np.zeros(len(C))
	time_stat = np.zeros(len(C))
	for fl in fileName:
		if fl == 'rcv1_test.binary':
			cv_folds = 2
		X_,y = load_svmlight_file(fl)
		l = X_.shape[0]
		S = X_.toarray()
		P = np.append(S.T,[np.ones(l)],0).T 	#Augmenting for bias
		X = csr_matrix(P)
		n = X.shape[1]
		randomized = np.arange(l)
		np.random.shuffle(randomized)

		for j in range(len(C)):
			accu[j] = 0
			time_stat[j] = 0

		for cv_iter in range(cv_folds):
			X_test = X[randomized[(cv_iter*l/cv_folds):((cv_iter+1)*l/cv_folds)]]
			y_test = y[randomized[(cv_iter*l/cv_folds):((cv_iter+1)*l/cv_folds)]]
			l_test = (cv_iter+1)*l/cv_folds - cv_iter*l/cv_folds
			X_train = X[list(randomized[0:(cv_iter*l/cv_folds)])+list(randomized[((cv_iter+1)*l/cv_folds):l])]
			y_train = y[list(randomized[0:(cv_iter*l/cv_folds)])+list(randomized[((cv_iter+1)*l/cv_folds):l])]
			l_train = l - l_test

			for j in range(len(C)):
				time_stat[j] = time_stat[j] - time.time()
				w = tra(X_train, y_train, l_train, n, C[j])
				time_stat[j] = time_stat[j] + time.time()

				accur = accuracy(w, X_test, l_test, n, y_test)
				accu[j] += accur * l_test

			del X_test
			del y_test
			del X_train
			del y_train
			
		for j in range(len(C)):
			accu[j] = (accu[j]*1.0)/l;
			print "For file %s at C = %lf, accu = %lf and time = %lf" %(fl, C[j], accu[j], time_stat[j])
		
		del X_
		del X
		del y
		del S
		del P
		del randomized


def main():
	C = [0.25, 1, 4, 16]
	fileName = ['a9a','real-sim.svml','news20.binary','rcv1_test.binary']
	CV(C, fileName)


if __name__ == "__main__":
	main()
