import numpy as np
from scipy.spatial import distance
import copy
from CBP import CBP
import matplotlib.pyplot as plt
import math



class hierarchicalClustering(object):
	def __init__(self, n_clusters, CONST_C,max_iterations  = 4000, random_state = 0, difftype = "cosine difference", mergeCriteria = "inertia", verbose = False):
		'''
		mergeCriteria: "inertia", 
		'''
		self.verbose = verbose
		self.type = difftype
		self.mergeCriteria = mergeCriteria
		self.n_clusters = n_clusters
		self.CONST_C = CONST_C
		self.n_iter_ = max_iterations
		self.distMatrix = []
		self.Gradient = []
		self.dTrain = []

		self.GeMatrix = []
		self.mergeMatrix = []


		# update below when ever merge
		#cluster membership matrix
		self.clusterMembership = []
		self.cluster_centers_ = []
		#inertia refers to the objective function
		self.inertia_ = 0
		self.currentNumClusters = 0
		self.labels_ = []



	def Initialized_Gabriel_edited_Matrix(self):
		def Gabriel_neighbors(indexi):
			inCluster = np.repeat(-1, self.n )
			inCluster[indexi] = 1
			Ge_model = CBP(self.dTrain,inCluster, Euc_d  = self.distMatrix)
			Gabriel_neighbors = np.array(Ge_model.points)[:,1]
			isGabrielNeighbors = np.repeat(0, self.n)
			isGabrielNeighbors[Gabriel_neighbors ] = 1
			return(isGabrielNeighbors)

		self.GeMatrix  = np.vstack([Gabriel_neighbors(indexi) for indexi in range(self.n)])


	def Initialize_InertiaInc_Matrix(self):
		# inertia increment will always be positive, inf indicates they are not Gabriel neighbors. 
		self.mergeMatrix = np.full((self.currentNumClusters, self.currentNumClusters), np.inf)

		for i in range(self.currentNumClusters):
			Gabriel_neighbors = self.Gabriel_neighbor_of_centroid(i)
			i_array = np.full(shape=(len(Gabriel_neighbors), 1), fill_value=i)
			# Pair i with the original array to create a 2D array
			index_pairs = np.hstack((i_array, Gabriel_neighbors.reshape(-1, 1)))
			for index_pair in index_pairs:
				# contains some redundant calculations/(i,j) and (j,i). 
				self.mergeMatrix[index_pair[0], index_pair[1]] = self.inertia_Increment_If_Merge(index_pair[0], index_pair[1])

	def Update_InertiaInc_Matrix(self,index_i, index_j):
		

		if index_i > index_j:
			index_i = index_i - 1

		# Delete the j-th column/row from the matrix
		self.mergeMatrix  = np.delete(self.mergeMatrix , index_j, axis=1)
		self.mergeMatrix  = np.delete(self.mergeMatrix , index_j, axis=0)

		# Recalculate i-th column/row since ith cluster is now merged.
		self.mergeMatrix[index_i,:] = np.inf
		self.mergeMatrix[:,index_i] = np.inf

		Gabriel_neighbors = self.Gabriel_neighbor_of_centroid(index_i)
		i_array = np.full(shape=(len(Gabriel_neighbors), 1), fill_value=index_i)
		# Pair i with the original array to create a 2D array
		index_pairs = np.hstack((i_array, Gabriel_neighbors.reshape(-1, 1)))
		for index_pair in index_pairs:
			# contains some redundant calculations/(i,j) and (j,i). 
			self.mergeMatrix[index_pair[0], index_pair[1]] = self.inertia_Increment_If_Merge(index_pair[0], index_pair[1])


	def merge(self):
		'''
		merge the cluster with smallest difference in Intra-cluster distance + gradients

		'''
		print(self.currentNumClusters)
		if self.verbose == True:
			print(self.currentNumClusters)
		if self.currentNumClusters <= self.n_clusters:
			raise ValueError("Current number of clusters is less than K!")
		
		# ------------------------ Merge and Update ------------------------------------

		# merge the smallest increment pair
		flat_index_of_smallest = self.mergeMatrix.argmin()
		index_of_smallest = np.unravel_index(flat_index_of_smallest, self.mergeMatrix.shape)
		index_toMerge_i = index_of_smallest[0]
		index_toMerge_j = index_of_smallest[1]

		if index_toMerge_i == index_toMerge_j:
			raise ValueError('wrong merge! cannot merge with itself')
		# update 
		if self.verbose == True:
			plt.scatter(self.dTrain[:,0], self.dTrain[:,1], c = self.labels_)
			plt.scatter(self.dTrain[self.clusterMembership[:,index_toMerge_i]==1,0], self.dTrain[self.clusterMembership[:,index_toMerge_i]==1,1], c = 'none', edgecolors='red',marker = 'H', s = 200)
			plt.scatter(self.dTrain[self.clusterMembership[:,index_toMerge_j]==1,0], self.dTrain[self.clusterMembership[:,index_toMerge_j]==1,1], c = 'none', edgecolors='black',marker = 'H', s = 100)
			plt.show()

		if self.inertia_Increment_If_Merge(index_toMerge_i, index_toMerge_j) <0:
			raise ValueError('negative increase?')

		self.inertia_ = self.inertia_ + self.inertia_Increment_If_Merge(index_toMerge_i, index_toMerge_j)

		self.currentNumClusters = self.currentNumClusters - 1
		# update cluster membership, delete cluster j, merge its members to cluster i.
		self.labels_[self.labels_ == index_toMerge_j] = index_toMerge_i
		# Create an empty mapping dictionary
		mapping = {}
		# Iterate through the unique labels and assign a new value iteratively
		unique_labels = set(self.labels_)
		for idx, label in enumerate(unique_labels):
			mapping[label] = idx
		# Transform the labels using the mapping
		self.labels_ = np.array([mapping[label] for label in self.labels_])
		self.clusterMembership = np.zeros((self.n, self.currentNumClusters),  dtype=int)
		for i in range(self.n):
			self.clusterMembership[i,self.labels_[i]] = 1
		self.cluster_centers_ = [ np.mean(np.array(self.dTrain[self.clusterMembership[:,i]==1,:]),axis = 0) for i in range(self.currentNumClusters)]
		# Update mergeMatrix 
		self.Update_InertiaInc_Matrix(index_of_smallest[0],index_of_smallest[1])




	def inertia_Increment_If_Merge(self,i,j):
		'''
		return the increment in inertia if merging cluster i and j.
		'''
		index = np.arange(self.n)
		pointsIndexes = np.append(index[self.clusterMembership[:,i]==1], index[self.clusterMembership[:,j]==1])
		clusterCenter = np.mean(self.dTrain[pointsIndexes,:],axis = 0)
		clusterCenterGradient = np.mean(self.Gradient[pointsIndexes], axis = 0)
		intraClusterDist = 0
		intraClusterGradient = 0
		for k in pointsIndexes:
			point = np.array(self.dTrain[k])
			intraClusterDist += np.linalg.norm(point - clusterCenter)
			if self.type == "norm2 gradient":
				intraClusterGradient += np.linalg.norm(self.Gradient[k,:] - clusterCenterGradient)
			elif self.type == "cosine difference":
				#Angular difference 
				intraClusterGradient += self.GradientDifference(self.Gradient[k,:], clusterCenterGradient)
		inertia_Merged = intraClusterDist  + self.CONST_C*intraClusterGradient


		return inertia_Merged - self.Innertia(i) - self.Innertia(j)

	def cosAngularDifference_If_Merge(self, i, j):
		index = np.arange(self.n)
		clusterGradient_i = np.mean(self.Gradient[ self.clusterMembership[:,i]==1],axis = 0)
		clusterGradient_j = np.mean(self.Gradient[ self.clusterMembership[:,j]==1],axis = 0)

		return self.GradientDifference(clusterGradient_i, clusterGradient_j, difftype = 'linear')

	def Innertia(self,i):
		'''
			inertia in cluster i = intra-cluster difference + CONST_C times difference in Gradient:
		'''
		index = np.arange(self.n)
		pointsIndexes = index[self.clusterMembership[:,i]==1]
		clusterCenter = np.array(self.cluster_centers_[i])
		# calculate the mean gradient as column mean
		clusterCenterGradient = np.mean(self.Gradient[pointsIndexes], axis = 0)
		intraClusterDist = 0
		intraClusterGradient = 0
		for k in pointsIndexes:
			point = np.array(self.dTrain[k])
			intraClusterDist += np.linalg.norm(point - clusterCenter)
			# sum of norm2 difference 
			if self.type == "norm2 gradient":
				intraClusterGradient += np.linalg.norm(self.Gradient[k,:] - clusterCenterGradient)
			elif self.type == "cosine difference":
				#Angular difference 
				intraClusterGradient += self.GradientDifference(self.Gradient[k,:], clusterCenterGradient) 
		return intraClusterDist  + self.CONST_C*intraClusterGradient

	def GradientDifference(self,vectorA,vectorB, difftype = 'linear'):
		cosAngularDifference = np.dot(vectorA,  vectorB)/  np.linalg.norm(vectorA)* np.linalg.norm(vectorB) 
		if difftype == 'linear':
			diff = np.abs(1 - cosAngularDifference)
		elif difftype == 'exponential':
			diff = math.exp(1  + np.abs(1 - cosAngularDifference) )
		return diff 



	def Gabriel_neighbor_of_centroid(self,centroidIndex):
		'''
		return the centroids that containing its Gabriel neighbors
		'''
		#find the points in the centroids
		index = np.arange(self.n)
		pointsIndexes = index[self.labels_ == centroidIndex]
		pointsToSearch = index[self.labels_ != centroidIndex]
		neighbors_unfiltered = np.unique(np.concatenate([np.where(np.array(self.GeMatrix)[i,:] == 1)[0] for i in pointsIndexes]))
		neighbors = neighbors_unfiltered[np.isin(neighbors_unfiltered, list(pointsToSearch))]

		return np.unique(self.labels_[neighbors])
		




	def fit(self,dTrain, Gradient):
		self.n = len(dTrain)
		self.currentNumClusters = self.n
		self.cluster_centers_ = dTrain
		self.dTrain = dTrain
		#normalized gradient
		self.Gradient = Gradient
		# create a distance matrix
		self.distMatrix = distance.cdist(dTrain, dTrain, 'euclidean')

		self.clusterMembership = np.identity(self.n,  dtype=int)
		self.labels_ =  np.argmax(self.clusterMembership, axis=1)
		self.inertia_ = 0

		self.Initialized_Gabriel_edited_Matrix()
		self.Initialize_InertiaInc_Matrix()

		

		while self.currentNumClusters > self.n_clusters:
			self.merge()













































