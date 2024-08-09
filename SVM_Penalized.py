from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import cvxpy as cp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import sklearn

import matplotlib.pyplot as plt



class SVM_Penalized(object):
	def __init__(self, C, K, tol = 0.0001):
		self.C = C
		self.K = K
		self.tol = tol
		self.alpha =[]
		self.coef_ = []           # w consists of 2 elements
		self.intercept_ = [] 

	def fit_SVM(self, A_train, C_train):

		svm_model = sklearn.svm.SVC(kernel='linear', C = self.C)
		svm_model.fit(A_train, C_train)

		self.coef_std = svm_model.coef_[0]
		self.intercept_std = svm_model.intercept_[0]


	def fit(self, A_train_original, C_train, dQ = None):
		if dQ is None:
			raise ValueError("must provide gradients")

		# Standardize the data
		scaler = StandardScaler()
		A_train = scaler.fit_transform(A_train_original)

		self.fit_SVM(A_train, C_train)


		n, p = np.array(A_train).shape
		gp_env = gp.Env(empty=True) 
		#suppress or show output
		gp_env.setParam("OutputFlag",0)
		gp_env.start()
		m = gp.Model("gp model",env=gp_env)
		m.params.NonConvex = 2

		alpha = m.addVars(n, vtype=GRB.CONTINUOUS, lb = 0, ub = self.C, name = 'alpha')
		m.addConstr(quicksum( alpha[j]*C_train[j] for j in range(n))==0 )


		w_1 = self.coef_std
		#standardize dQ
		dQ = [ gradient/np.linalg.norm(gradient) for gradient in dQ]
		mean_vector = np.mean(dQ , axis=0)

		# Calculate the length of the mean vector
		mean_vector_length = np.linalg.norm(mean_vector)
		#print(1 - mean_vector_length)

		w_2 = np.array(np.mean(dQ, axis = 0)/np.linalg.norm(np.mean(dQ, axis = 0),2))
		self.mean_gradient = w_2
		w_1_proj_2 =  w_1 @ w_2.T/(w_2 @ w_2.T) * w_2
		w_1_proj_2 = w_1_proj_2/np.linalg.norm(w_1_proj_2)

		penalty = self.K*( 1 - (w_1 @ w_2.T)**2/(w_1 @ w_1.T) )
		t = (quicksum( alpha[j]*(w_1 - w_1_proj_2)@A_train[j].T*C_train[j]  for j in range(n)) - (w_1 - w_1_proj_2)@w_1_proj_2.T - penalty)/((w_1 - w_1_proj_2)@(w_1 - w_1_proj_2).T)
		m.addConstr((quicksum( alpha[j]*(w_1 - w_1_proj_2)@A_train[j].T*C_train[j]  for j in range(n)) - (w_1 - w_1_proj_2)@w_1_proj_2.T)/((w_1 - w_1_proj_2)@(w_1 - w_1_proj_2).T) >= 0)
		m.addConstr((quicksum( alpha[j]*(w_1 - w_1_proj_2)@A_train[j].T*C_train[j]  for j in range(n)) - (w_1 - w_1_proj_2)@w_1_proj_2.T)/((w_1 - w_1_proj_2)@(w_1 - w_1_proj_2).T) <= 1)
		w_t = t*w_1 + (1 - t)*w_1_proj_2
		objective = quicksum(alpha[j] for j in range(n)) + 0.5*w_t@w_t.T + penalty*t - quicksum( alpha[j]*w_t@A_train[j].T*C_train[j] for j in range(n))


		m.setObjective(objective, GRB.MAXIMIZE)
		m.optimize()
		# Retrieve the solution
		
		if m.status == GRB.OPTIMAL:
			alpha_ = m.getAttr('x',alpha)
			self.alpha = alpha_
			m.dispose()
			gp_env.dispose()
		else:
			print("No optimal solution found")
			m.dispose()
			gp_env.dispose()


		t = (np.sum( [alpha_[j]*(w_1 - w_1_proj_2)@A_train[j].T*C_train[j]  for j in range(n)] ) - (w_1 - w_1_proj_2)@w_1_proj_2.T - penalty )/((w_1 - w_1_proj_2)@(w_1 - w_1_proj_2).T)

		w_penalized_std = t*w_1 + (1 - t)*w_1_proj_2

		# Identify support vectors
		support_vectors = []
		tol = 0.001

		for i in range(n):
			if tol < self.alpha[i] < self.C:
				support_vectors.append(i)
		
		# Calculate b using the average value of the support vectors
		if support_vectors:
			b_values = [C_train[i] - np.dot(A_train[i], w_penalized_std) for i in support_vectors]
			intercept_penalized_std = np.mean(b_values)
		else:
			intercept_penalized_std = 0


		self.coef_.append(w_penalized_std / scaler.scale_)           # w consists of 2 elements
		self.intercept_.append(intercept_penalized_std - np.dot(w_penalized_std, scaler.mean_ / scaler.scale_))


	def predict(self, A_train):
		return( np.array([1 if np.sum(self.coef_[0]*A_train[j]) + self.intercept_[0] >= 0 else -1   for j in range(len(A_train))]) )





if __name__ == '__main__':
	main()