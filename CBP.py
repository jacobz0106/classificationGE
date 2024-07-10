#import os
#import sys
import numpy as np
import pandas as pd
import random
#import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore', np.RankWarning)

from sklearn import cluster
from sklearn import svm
from PSVM import MagKmeans
import hierarchicalClustering
from SVM_Penalized import SVM_Penalized
# from sklearn.base import BaseEstimator, ClassifierMixin

class LabelEncode(object):
	def __init__(self, L):
		self.L1 = np.unique(np.array(L))
		self.L2 = None
		self.dicForward= None
		self.dicBackward = None

	def fit(self,Y):
		self.L2 = np.unique(np.array(Y))
		self.dicForward= {self.L1[i]:self.L2[i] for i in range(len(self.L1))}
		self.dicBackward = {self.L2[i]:self.L1[i] for i in range(len(self.L1))}

	def transform(self, Y):
		Y = np.array(Y).reshape(-1)
		return np.array([self.dicBackward.get(i) for i in Y])

	def transform_back(self,Y):
		Y = np.array(Y).reshape(-1)
		return np.array([self.dicForward.get(i) for i in Y])


def Euclidean_distance_vector(B, A_train):
				return np.array([np.linalg.norm(B - A_i) for A_i in A_train ]) 


class clusters(object):
	def __init__(self):
		self.labels_ = []
		self.clusterNum = 0

def find_k_nearest_points(k, point, points):
    """
    Find the k nearest points to a given point within a set of points.

    Args:
    k (int): The number of nearest points to return.
    point (ndarray): The reference point (1D array).
    points (ndarray): A 2D array of points to search within.

    Returns:
    ndarray: The k nearest points to the given point.
    """
    # Compute the Euclidean distances from the given point to all other points
    dists = np.array([np.linalg.norm(point - p) for p in points])

    try:
      index_self = np.where(np.all(points == point, axis=1))[0]
      # Set the distance of the point itself to a very high value
      dists[index_self] = 0
    except IndexError:
      pass  # The point is not in the list or cannot be exactly matched

    # Get the indices of the k smallest distances
    nearest_indices = np.argsort(dists)[:k]

    return nearest_indices


# ---------------------------------- Improved Gabriel editing algorithm ---------------------------------- 
# Code based on the pseudocode from the paper:
#   Title: "Application of proximity graphs to editing nearest neighbour decision rules"
#   Authors: Binary K.Bhattacharya, Ronald S.Poulsen, Godfried T.Toussaint
#   Published: 2010
# Class CBP implementation adapted from the paper's pseudocode.

class CBP(object):
	'''
	Characteristic Boundary Points(CBP) for a training data set.
	Contains a list of (i,j) indexing points
	'''
	def __init__(self, A_train, C_train, Euc_d = None):

		self.count = 0
		#pairs of CBP points 
		self.points = []
		self.midpoints = []
		self.margin = []
		if Euc_d is None:
			self._Euc(A_train, C_train)
		else:
			self.Euc_d = Euc_d

		self.transformer = LabelEncode([0,1])
		self.transformer.fit(C_train)
		self.C_train = self.transformer.transform(C_train)

		self._train( A_train, self.C_train)
		self.midpoints = np.array(self.midpoints)

	def _train(self, A_train, C_train):
		'''
		Train the model, input nXd matrix A_train and 1Xn matrix labels
		'''
		self.points = []
		self.midpoints = []
		for i in np.where(np.array(C_train) == 1)[0]:
			pointSet = np.where(np.array(C_train) == 0)[0]
			for j in pointSet:
				X_m = (np.array(A_train[i]) + np.array(A_train[j]) )/2
				for k in range(len(C_train)):
					if k == i or k ==j:
						continue 

					if self.Euc_d[i,j]**2 > self.Euc_d[i,k]**2 + self.Euc_d[j,k]**2:
						pointSet = pointSet[pointSet!=j]
						break
					else:
						#test whether p_k is inside pontSet:
						if k in pointSet:
							#test whether P_j lies inside disk P_ik:
							if self.Euc_d[i,k]**2 > self.Euc_d[i,j]**2 + self.Euc_d[j,k]**2:
								pointSet = pointSet[pointSet!=k]
			#end
			if len(pointSet) >= 1:
				for m in pointSet:
					self.points.append([i,m])
					self.midpoints.append( (A_train[i] + A_train[m])/2)
					self.margin.append(self.Euc_d[i,m])
					self.count += 1

	def _Euc(self, A_train, C_train):
		self.Euc_d = np.zeros(shape=(len(C_train),len(C_train)))
		self.Euc_d = np.array([Euclidean_distance_vector(A_i, A_train) for A_i in A_train])



# ---------------------------------- referenced method: linear ensemble ---------------------------------- 
# Code based on the pseudocode from the paper:
#   Title: "Geometry-Based Ensembles:Toward a Structural Characterization of the Classification Boundary"
#   Authors: Priol Pujol, David Masip
#   Published: IEEE Transactions on pattern analysis and machine learning, 2009, vol 31, number 6
# Class referenced_method implementation adapted from the paper's pseudocode.


class referenced_method(object):
	'''
	C_train must have label{1, -1}
	'''
	def __init__(self,alpha = 0.5, constLambda = 0.5):
		self.cbp = []
		self.alpha = alpha
		self.constLambda = constLambda
		self.weights = []

	def _train(self, A_train, C_train):
		self.cbp = CBP(A_train, C_train)

	def fit(self, A_train, C_train):
		self.cbp = []
		self.weights = []
		# Check if the input C_train is a numpy array contaning -1 and 1
		valid_values = {-1, 1} 
		if isinstance(C_train, (list, np.ndarray)):
			unique_values = set(C_train)
			if not (valid_values.issubset(unique_values) and len(unique_values) == 2):
				raise ValueError("Input array must contain both -1 and 1.")
		else:
			raise ValueError("Input must be a list or a numpy array.")
		self.transformer = LabelEncode([-1,1])
		self.transformer.fit(C_train)
		self.C_train = self.transformer.transform(C_train)
		self.A_train = A_train
		self._train(A_train, self.C_train)

		self.weights = np.zeros(self.cbp.count)
		A = np.zeros(shape = (len(self.A_train), self.cbp.count))
		for i in range(len(self.A_train)):
			A[i,:] = self.baseClassifier(self.A_train[i])
		initialWeights = np.repeat(1/self.cbp.count, self.cbp.count)
		self.weights = np.matmul(self.constLambda**2*initialWeights + np.matmul(np.transpose(A), self.C_train),
														 np.linalg.inv(np.matmul(np.transpose(A), A) + np.identity(self.cbp.count)*self.constLambda**2)
										)

	def baseClassifier(self,x):
		classifiers = np.zeros(self.cbp.count)
		for i in range(self.cbp.count):
			midPoint = self.cbp.midpoints[i]
			upperPoint = self.A_train[self.cbp.points[i][0]]
			lowerPoint = self.A_train[self.cbp.points[i][1]]
			disc= sum((x - midPoint)*(upperPoint - lowerPoint))
			if disc >= 0:
				classifiers[i] = 1
			else:
				classifiers[i] = -1
		return classifiers

	def ensemble(self,x):
		func = np.sum(self.weights*self.baseClassifier(x)) - self.alpha
		if func >= 0:
			return 1
		else:
			return -1

	def predict(self, x):
		ensembleVec = np.vectorize(self.ensemble, signature = '(n)->()')
		predict = ensembleVec(x)
		return self.transformer.transform_back(predict)

	def get_params(self,deep=True):
			return {"alpha" : self.alpha,
			"constLambda" : self.constLambda}

	def set_params(self, **parameters):
			# for parameter, value in parameters.items():
			#     setattr(self, parameter, value)
			self.__init__(**parameters)
			return self




# ---------------------------------- GPSVM Algorithm ---------------------------------- 



class GPSVM(object):
	def __init__(self, method, clusterNum = 1,ensembleNum=1,C = 0.1, CONST_C = 1):

		self.cbp = []
		self.clusterNum = clusterNum
		self.method = method
		self.clusterCentroids = []
		self.ensembleNum = ensembleNum
		self.C = C 
		self.CONST_C = CONST_C
		self.clusters = []
		self.SVM = []


	def _train(self, A_train, C_train):
		# Check if the input C_train is a numpy array contaning -1 and 1
		valid_values = {-1, 1} 
		if isinstance(C_train, (list, np.ndarray)):
			unique_values = set(C_train)
			if not (valid_values.issubset(unique_values) and len(unique_values) == 2):
				raise ValueError("Input array must contain both -1 and 1.")
		else:
			raise ValueError("Input must be a list or a numpy array.")
		#[0,1] -> [1, -1]
		self.transformer = LabelEncode([-1,1])
		self.transformer.fit(C_train)
		self.C_train = self.transformer.transform(C_train)

		self.A_train = A_train
		self.cbp = CBP(A_train, C_train)
		self.d = len(A_train[0])
		self.clusterLabel = []


	def fit(self, A_train, C_train, dQ = None):
		'''
		cluster = hierarchicalClustering or Kmeans
		'''
		self.SVM = []
		self.cbp = []
		self.clusters = []
		self.clusterCentroids = []
		self._train(A_train, C_train)
		if self.cbp.count < self.clusterNum:
			self.clusterNum = self.cbp.count
		if self.method == "hierarchicalClustering":
			self.clusters = hierarchicalClustering.hierarchicalClustering(n_clusters=self.clusterNum, CONST_C = self.CONST_C, random_state=0)
			

			# take gradient as the average of the Gabriel neighbors

			Gradient_i = np.array(dQ)[np.array(self.cbp.points)[:,0] ]
			Gradient_i_norm = np.array( [ i /np.linalg.norm(i) for i in Gradient_i   ] )

			Gradient_j = np.array(dQ)[np.array(self.cbp.points)[:,1] ]
			Gradient_j_norm = np.array( [ i /np.linalg.norm(i) for i in Gradient_j   ] )
			estimated_Gradient =   (Gradient_i_norm   +   Gradient_j_norm)/2
			self.clusters.fit(np.array(self.cbp.midpoints), estimated_Gradient) 
		else:
			self.clusters = cluster.KMeans(n_clusters = self.clusterNum, n_init = 'auto')
			self.clusters.fit(np.array(self.cbp.midpoints))
		self.clusterCentroids = self.clusters.cluster_centers_
		self.clusterLabel= np.unique(self.clusters.labels_)
		# check if all clusters are generated
		if self.clusterNum > len(np.unique(self.clusters.labels_)):
			self.clusterNum = len(np.unique(self.clusters.labels_))
		if self.ensembleNum > self.clusterNum:
			self.ensembleNum = self.clusterNum
		for i in range(self.clusterNum):
			midpoints_subset_index = self.clusters.labels_ == self.clusterLabel[i]
			Gabriel_pairs = np.array(self.cbp.points)[midpoints_subset_index]
			subset_index = Gabriel_pairs.reshape(-1)
			model = svm.SVC(kernel='linear', C = self.C)
			model.fit(self.A_train[subset_index], self.C_train[subset_index])
			self.SVM.append(model)

	def ensemble(self, x):
		I = Euclidean_distance_vector(x,self.clusterCentroids)
		index = I.argsort()[0:self.ensembleNum]
		classifier = 0.0
		for i in range(len(index)):
			j = index[i]

			weight = 1/I[index[i]]/(np.sum(1/I[index]))
			classifier = classifier + weight*self.SVM[j].predict([x])
		if classifier >= 0:
			return 1
		else:
			return -1
	def predict(self,x):
		ensembleVec = np.vectorize(self.ensemble, signature = '(n)->()')
		predict = np.array(ensembleVec(np.array(x).reshape(-1,self.d))).reshape(-1)
		#[1,-1] -> [0, 1]
		return self.transformer.transform_back(predict)

	def get_params(self,deep=True):
			return {"clusterNum" : self.clusterNum,
			"ensembleNum" : self.ensembleNum,
			"C": self.C,
			"method": self.method,
			"CONST_C":self.CONST_C}

	def set_params(self, **parameters):
			# for parameter, value in parameters.items():
			#     setattr(self, parameter, value)
			self.__init__(**parameters)
			return self



# ---------------------------------- GMSVM Algorithm with clusters_overlap---------------------------------- 

class GMSVM(object):
	def __init__(self, clusterSize = 3,ensembleNum=1,C = 0.1, CONST_C = 1, K = 1):

		self.cbp = []
		self.clusterSize = clusterSize
		self.clusterCentroids = []
		self.ensembleNum = ensembleNum
		self.clusterNum  = 0
		self.C = C 
		self.K = K
		self.clusters = []
		self.CONST_C = CONST_C
		self.clusters = []
		self.SVM = []


	def _train(self, A_train, C_train):
		# Check if the input C_train is a numpy array contaning -1 and 1
		valid_values = {-1, 1} 
		if isinstance(C_train, (list, np.ndarray)):
			unique_values = set(C_train)
			if not (valid_values.issubset(unique_values) and len(unique_values) == 2):
				raise ValueError("Input array must contain both -1 and 1.")
		else:
			raise ValueError("Input must be a list or a numpy array.")
		#[0,1] -> [1, -1]
		self.transformer = LabelEncode([-1,1])
		self.transformer.fit(C_train)
		self.C_train = self.transformer.transform(C_train)

		self.A_train = A_train
		self.cbp = CBP(A_train, C_train)
		self.d = len(A_train[0])
		self.clusterLabel = []


	def fit(self, A_train, C_train, dQ = None):
		'''
		cluster = hierarchicalClustering or Kmeans
		'''
		self.SVM = []
		self.cbp = []
		self.clusters = clusters()
		self.clusterCentroids = []
		self._train(A_train, C_train)
		self.clusterNum  = self.cbp.count
		self.clusters.labels_ = np.array(range(self.clusterNum))
		self.clusters.clusterNum = self.cbp.count
		self.clusterCentroids = self.cbp.midpoints
		# define the clusters 

		for midpoint in self.cbp.midpoints:
			nearest_index = find_k_nearest_points(self.clusterSize,midpoint,self.cbp.midpoints)
			GE_point_index = np.unique(np.array(self.cbp.points)[nearest_index].reshape(-1))
			model = SVM_Penalized(C = self.C, K = self.K)
			model.fit(self.A_train[GE_point_index], self.C_train[GE_point_index], dQ)
			self.SVM.append(model)

	def ensemble(self, x):
		I = Euclidean_distance_vector(x,self.clusterCentroids)
		index = I.argsort()[0:self.ensembleNum]
		classifier = 0.0
		for i in range(len(index)):
			j = index[i]

			weight = 1/I[index[i]]/(np.sum(1/I[index]))
			classifier = classifier + weight*self.SVM[j].predict([x])
		if classifier >= 0:
			return 1
		else:
			return -1
	def predict(self,x):
		ensembleVec = np.vectorize(self.ensemble, signature = '(n)->()')
		predict = np.array(ensembleVec(np.array(x).reshape(-1,self.d))).reshape(-1)
		#[1,-1] -> [0, 1]
		return self.transformer.transform_back(predict)

	def get_params(self,deep=True):
			return {"clusterSize" : self.clusterSize,
			"ensembleNum" : self.ensembleNum,
			"C": self.C,
			"CONST_C":self.CONST_C}

	def set_params(self, **parameters):
			# for parameter, value in parameters.items():
			#     setattr(self, parameter, value)
			self.__init__(**parameters)
			return self




# ---------------------------------- LSVM Algorithm ---------------------------------- 
# Code based on the pseudocode from the paper:
#   Title: "SVM-KNN: Discriminative nearest neighbor classification for visual category recognition"
#   Authors: Zhang, Hao and Berg, Alexander C and Maire, Michael and Malik, Jitendra
#   Published: IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2006


class LSVM(object):
	def __init__(self,K = 1, C = 1):
		self.K = K
		self.C = C

	def fit(self, A_train, C_train):
		self.A_train = A_train
		self.C_train = C_train


	def classifier(self,x):
		I = Euclidean_distance_vector(x,self.A_train)
		index = I.argsort()[0:self.K]
		label = np.unique(self.C_train[index])
		if len(label) == 1:
			return label[0]
		else:
			model = svm.SVC(kernel='linear', C = self.C)
			model.fit(self.A_train[index], self.C_train[index])
			return model.predict([x])

	def predict(self,x):
		classifiersVec = np.vectorize(self.classifier,  signature = '(n)->()')
		return classifiersVec(x)

	def get_params(self,deep=True):
			return {"K" : self.K,
			"C" : self.C
			}

	def set_params(self, **parameters):
			# for parameter, value in parameters.items():
			#     setattr(self, parameter, value)
			self.__init__(**parameters)
			return self







# ---------------------------------- PSVM Algorithm ---------------------------------- 
# Code based on the pseudocode from the paper:
#   Title: "Efficient Algorithm for Localized Support Vector Machine"
#   Authors: Cheng, Haibin and Tan, Pang-Ning and Jin, Rong
#   Published: IEEE Transactions on Knowledge and Data Engineering, 2009, vol 2, number 4

class SVM_Single(object):
	def __init__(self):
		self.coef_ = [[0,0]]         
		self.intercept_ = [[]]

	def fit(self, dfTrain,dfLabel):
		self.intercept_[0] = np.unique(dfLabel)

	def predict(self,dfTest):
		return np.full_like(len(dfTest), self.intercept_[0])


class PSVM(object):
	def __init__(self,clusterNum = 1,ensembleNum=1,C = 0.1, R = 0.5, max_iterations =  500):
		'''
		fir args = [cluster number, ensemble number,  C, R]
		'''
		self.clusterNum = clusterNum
		self.ensembleNum = ensembleNum
		self.max_iterations = max_iterations
		self.C = C
		self.R = R
		self.SVM = []
		self.MagKmeans = []


	def fit(self, A_train, C_train):
		self.A_train = A_train
		self.transformer = LabelEncode([-1,1])
		self.transformer.fit(C_train)
		self.C_train = self.transformer.transform(C_train)
		# Check if the input C_train is a numpy array contaning -1 and 1
		valid_values = {-1, 1} 
		if isinstance(C_train, (list, np.ndarray)):
			unique_values = set(C_train)
			if not (valid_values.issubset(unique_values) and len(unique_values) == 2):
				raise ValueError("Input array must contain both -1 and 1.")
		else:
			raise ValueError("Input must be a list or a numpy array.")

		# Fit K-Means to the data with two clusters
		self.MagKmeans = MagKmeans(n_clusters = self.clusterNum, max_iterations = self.max_iterations, random_state=0)
		# Standardize the features
		self.MagKmeans.fit(self.A_train, self.C_train, R = self.R)
		self.clusterCentroids = self.MagKmeans.cluster_centers_

		# check if all clusters are generated
		self.clusterNum = self.MagKmeans.K

		if self.ensembleNum > self.clusterNum:
			self.ensembleNum = self.clusterNum

		for i in range(self.clusterNum):
			cluster_points = self.A_train[self.MagKmeans.labels_ == i]

			model = svm.SVC(kernel='linear', C = self.C)
			# check if cluster has more than two label:
			if len( np.unique(self.C_train[self.MagKmeans.labels_ == i]) ) == 2:
				model.fit(cluster_points, self.C_train[self.MagKmeans.labels_ == i])
			else:
				model = SVM_Single()
				model.fit( cluster_points, self.C_train[self.MagKmeans.labels_ == i] )
			self.SVM.append(model)


	def ensemble(self, x):
		I = Euclidean_distance_vector(x,self.clusterCentroids)
		index = I.argsort()[0:self.ensembleNum]
		classifier = 0.0
		sum_ = np.sum(1/I[index])
		for i in range(len(index)):
			j = index[i]
			weight = (1/I[index[i]])/sum_
			classifier = classifier + weight*self.SVM[j].predict([x])
		if classifier >= 0:
			return 1
		else:
			return -1
	
	def predict(self,x):
		ensembleVec = np.vectorize(self.ensemble, signature = '(n)->()')
		predict = np.array(ensembleVec(np.array(x).reshape(-1,len(self.A_train[0])))).reshape(-1)
		#[1,-1] -> [0, 1]
		return self.transformer.transform_back(predict)

	def get_params(self,deep=True):
			return {"clusterNum" : self.clusterNum,
			"ensembleNum" : self.ensembleNum,
			"C": self.C,
			"R":self.R}

	def set_params(self, **parameters):
			# for parameter, value in parameters.items():
			#     setattr(self, parameter, value)
			self.__init__(**parameters)
			return self





