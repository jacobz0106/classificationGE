import numpy as np
from scipy.spatial import distance
import copy
from CBP import CBP
import matplotlib.pyplot as plt
import math



class hierarchicalClustering(object):
	def __init__(self, n_clusters, CONST_C,max_iterations  = 4000, random_state = 0, difftype = "cosine difference", mergeCriteria = "inertia"):
		'''
		mergeCriteria: "inertia", 
		'''
		self.type = difftype
		self.mergeCriteria = mergeCriteria
		self.n_clusters = n_clusters
		self.CONST_C = CONST_C
		self.n_iter_ = max_iterations
		self.distMatrix = []
		self.Gradient = []
		self.dTrain = []


		# update below when ever merge
		#cluster membership matrix
		self.clusterMembership = []
		self.cluster_centers_ = []
		#inertia refers to the objective function
		self.inertia_ = 0
		self.currentNumClusters = 0
		self.labels_ = []



	def merge(self):
		'''
		merge the cluster with smallest difference in Intra-cluster distance + gradients

		'''
		print(self.currentNumClusters)
		# print(self.labels_)
		if self.currentNumClusters <= self.n_clusters:
			raise ValueError("Current number of clusters is less than K!")

		# find the min Gabriel neighboring centroids
		differenceArray = np.zeros( self.currentNumClusters )
		# store the cluster with min inertia increment if merge with.
		indexArray = np.repeat(0,self.currentNumClusters )

		# check here!!!---------------------------------------------
		for i in range(self.currentNumClusters):
			GE_clusterIndex = self.Gabriel_neighbor_of_centroid(i)
			# print('cluster:',i,'with size',np.sum(self.clusterMembership[:,i] ),'\n')
			# print('GeIndex:',GE_clusterIndex, '\n')
			if self.mergeCriteria == "angularDiff":
				inertiaIncrementArray = np.array([self.cosAngularDifference_If_Merge(i,j) for j in GE_clusterIndex])
			else:
				inertiaIncrementArray = np.array([self.inertia_Increment_If_Merge(i,j) for j in GE_clusterIndex])
			# print('incresement:' ,inertiaIncrementArray, '\n')
			differenceArray[i] = np.min(inertiaIncrementArray)
			indexArray[i] = GE_clusterIndex[np.argmin(inertiaIncrementArray)]
		# merge the smallest increment pair
		
		index_toMerge_i = np.argmin(differenceArray)
		index_toMerge_j = indexArray[index_toMerge_i]

		if index_toMerge_i == index_toMerge_j:
			raise ValueError('wrone merge! cannot merge with itself')
		# update 
		# plt.scatter(self.dTrain[:,0], self.dTrain[:,1], c = self.labels_)
		
		# plt.scatter(self.dTrain[self.clusterMembership[:,index_toMerge_i]==1,0], self.dTrain[self.clusterMembership[:,index_toMerge_i]==1,1], c = 'none', edgecolors='red',marker = 'H', s = 200)
		# plt.scatter(self.dTrain[self.clusterMembership[:,index_toMerge_j]==1,0], self.dTrain[self.clusterMembership[:,index_toMerge_j]==1,1], c = 'none', edgecolors='black',marker = 'H', s = 100)
		# plt.show()

		self.inertia_ = self.inertia_ + self.inertia_Increment_If_Merge(index_toMerge_i, index_toMerge_j)
		self.currentNumClusters = self.currentNumClusters - 1

		# update cluster membership

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

	def cosAngularDifference_If_Merge(self, i, j, ):
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

	# def Gabriel_neighbor_of_point(self,i, pointsIndexes):
	# 	'''
	# 	find the Gbriel neighbors
	# 	'''
	# 	if i in pointsIndexes:
	# 		pointsIndexes = np.delete(pointsIndexes,i)

	# 	Gabriel_set = copy.deepcopy(pointsIndexes)
	# 	setU = copy.deepcopy(pointsIndexes)
	# 	for j in Gabriel_set:
	# 		# check whether i is Gabriel Neighbor
	# 		for k in setU:
	# 			if k == j:
	# 				continue
	# 			if self.distMatrix[i,j]**2 >= self.distMatrix[i,k]**2 + self.distMatrix[j,k]**2 :
	# 				# k is inside the sphere so i is not in GE set
	# 				Gabriel_set = np.delete(Gabriel_set, j)
	# 				break
	# 			else: 
	# 				# test if k is in pointSet,
	# 				if k in Gabriel_set:
	# 					# test if j lies inside disk P_ik
	# 					if self.Euc_d[i,k]**2 > self.Euc_d[i,j]**2 + self.Euc_d[j,k]**2:
	# 						pointSet = Gabriel_set[Gabriel_set != k]
	# 	return Gabriel_set				


	# def Gabriel_neighbor_of_centroid(self,centroidIndex):
	# 	'''
	# 		return the centroids that containing its Gabriel neighbors
	# 	'''
	# 	#find the points in the centroids
	# 	index = np.arange(len(self.dTrain))
	# 	pointsIndexes = index[self.clusterMembership[:,centroidIndex]]

	# 	pointsToSearch = index[~np.isin(index, pointsIndexes)]

	# 	GE = np.array([])
	# 	for i in pointsIndexes:
	# 		GE = np.unqiue(np.append(GE, self.Gabriel_neighbor_of_point(i,pointsToSearch)))

	# 	# find the related Gabriel neighboring centroids of the current centorid
	# 	return np.unique(self.labels_[GE])


	def Gabriel_neighbor_of_centroid(self,centroidIndex):
		'''
		return the centroids that containing its Gabriel neighbors
		'''
		#find the points in the centroids
		index = np.arange(self.n)


		pointsIndexes = index[self.labels_ == centroidIndex]
		pointsToSearch = index[self.labels_ != centroidIndex]

		inCluster = np.repeat(1, self.n )
		inCluster[pointsToSearch] = -1

		Ge_model = CBP(self.dTrain,inCluster, Euc_d  = self.distMatrix)
		Gabriel_neighbors = np.array(Ge_model.points)[:,1]
		# check choosed the right index:
		set1 = set(pointsIndexes)
		set2 = set(Gabriel_neighbors)
		if len(set1.intersection(set2)) > 0:
			raise ValueError('wrong set chosen')
		# print('Now looking for neighbors of centoid:',centroidIndex)
		# print('Found Gabriel neighbor centroid:',np.unique(self.labels_[Gabriel_neighbors]))
		return np.unique(self.labels_[Gabriel_neighbors])




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

		while self.currentNumClusters > self.n_clusters:
			self.merge()













































