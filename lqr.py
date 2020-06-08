from scipy.linalg import eig,solve_continuous_are,inv
import numpy

def lqr(A,B,Q,R):
	try:
		X = numpy.matrix(solve_continuous_are(A, B, Q, R))
	except:
		print(Q)
		print(R)
	R = [[R]]
	K = numpy.matrix(inv(R)*(B.T*X))
	eigVals, eigVecs = eig(A-B*K)
	return K, X, eigVals