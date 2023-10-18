import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
import random

# ---------------------------------- MagKmeans Algorithm ---------------------------------- 
# Code based on the pseudocode from the paper:
# 	Title: "POF-Darts: Geometric Adaptive Sampling for Probability of Failure"
# 	Authors: Ebeida, Mohamed and Mitchell, Scott and Swiler, Laura and Romero, Vicente and Rushdi, Ahmad
# 	Published: Reliability Engineering & System Safety, vol 155, 2016
# Class POFdarts implementation adapted from the paper's pseudocode.


class POFdarts(object):
	def __init__(self, function_y, gradient, CONST_a,  critical_value, max_iterations  = 1000, random_state = 0):
		self.function_y = function_y
		self.gradient = gradient
		self.CONST_a = CONST_a
		self.critical_value = critical_value
		self.max_iterations = max_iterations
		self.seed = random_state
		# data frame is a list object
		self.df = []
		self.radius = []
		self.y = []
		self.Q = []


	def contain(self,point):
		for i in range(len(self.df) ):
			if np.linalg.norm(np.array(point) - np.array(self.df[i]) ) <= self.radius[i]:
				return True
		return False

	def remove_overlap(self):
		index = np.array(self.y) < self.critical_value
		I = [i for i, x in enumerate(index) if x]
		indexInv = [not i for i in index]
		J= [i for i, x in enumerate(index) if not x]
		for i in I:
			for j in J:
				if np.linalg.norm( np.array(self.df[i]) - np.array(self.df[j]) ) < self.radius[i] + self.radius[j]:
					L = np.abs( np.array(self.y[i]) -  np.array(self.y[j]) )/np.linalg.norm(np.array(self.df[i]) - np.array(self.df[j]) )
					self.radius[i] =  min(self.radius[i],  np.abs( np.array(self.y[i]) - self.critical_value)/L)
					self.radius[j] =  min(self.radius[j],np.abs( np.array(self.y[j]) - self.critical_value)/L)
		return


	def add_point(self,point):
		Q = self.gradient(point[0], point[1])
		y = self.function_y(point)
		radius = np.abs(y - self.critical_value)/(self.CONST_a*np.sqrt(Q[0]**2 + Q[1]**2))
		self.df.append(point)
		self.radius.append(radius)
		self.y.append(y)
		self.Q.append(Q)
		return

	def Generate_2D(self, iniPoints, N,xlim,ylim):
		random.seed(self.seed)
		X = np.random.uniform(xlim[0], xlim[1], iniPoints)
		Y = np.random.uniform(ylim[0], ylim[1], iniPoints)
		self.df = [[X[i], Y[i]] for i in range(iniPoints)]
		self.y = pd.DataFrame(self.df).apply(self.function_y, axis = 1).values.tolist()
		# calculate gradient
		self.radius= np.zeros(iniPoints)
		for i in range(iniPoints):
			Q = self.gradient(X[i], Y[i])
			self.radius[i]= np.abs(self.y[i] - self.critical_value)/(self.CONST_a*np.sqrt(Q[0]**2 + Q[1]**2))
		self.radius = self.radius.tolist()
		self.remove_overlap()
		i = 0
		while i < N:
			x = np.random.uniform(xlim[0], xlim[1], 1)[0]
			y = np.random.uniform(ylim[0], ylim[1], 1)[0]
			counter = 0
			while self.contain([x,y]) and counter < self.max_iterations:
				x = np.random.uniform(xlim[0], xlim[1], 1)[0]
				y = np.random.uniform(ylim[0], ylim[1], 1)[0]
				counter = counter + 1
			# no points found to add
			if counter == self.max_iterations:
				print('decrease')
				self.CONST_a = self.CONST_a * 3/2
				self.radius = [r * 2/3 for r in self.radius]
			# else found point to add
			else:
				z = self.function_y(np.array([x,y]))
				self.y.append(z)
				Q = self.gradient(x, y)
				r = np.abs(z - self.critical_value)/(self.CONST_a*np.sqrt(Q[0]**2 + Q[1]**2))
				self.radius.append(r)
				self.df.append([x,y])
				self.remove_overlap()
				i = i+1
		return







