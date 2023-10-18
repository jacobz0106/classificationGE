import matplotlib.pyplot as plt
from matplotlib import cm
from CBP import GPSVM, PSVM
from PSVM import MagKmeans
from dataGeneration import *
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn import svm



def main():

	num_points = 300
	df = create_binary_class_boundary_spiral(num_points)
	GPSVM_model = GPSVM(df[['X','Y']].values, df['Label'].values)
	GPSVM_model.fit(args = [20, 1, 5])

	PSVM_model = PSVM(df[['X','Y']].values, df['Label'].values)
	PSVM_model.fit(args = [10, 1, 1, 5])
	print(PSVM_model.predict(df[['X','Y']].values))



if __name__ == '__main__':
	main()
