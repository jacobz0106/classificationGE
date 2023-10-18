import numpy as np
from dataGeneration import Brusselator_Data_2_with_f, example1_Data, example2_Data, create_binary_class_boundary_spiral
import matplotlib.pyplot as plt
from CBP import PSVM, GPSVM, LSVM
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler

# -------------------------------------- comparison between runnning time -----------------------------
# spiral data set

def Running_time(N,outfile,outfileTotal,repeat = 20):
	# create matrix to store running time
	matrix = np.zeros((repeat*3, len(N)))

	matrix_total = np.zeros((repeat*3, len(N)))
	for n, i in zip(N, range(len(N))):
		for j in range(repeat):
			print([n,j])
			# initialize data set from spiral example
			df = create_binary_class_boundary_spiral(n)
			scaler = StandardScaler()

			dfTrain  = scaler.fit_transform(df[['X','Y']].values)


			Label = df['Label'].values

			args_LSVM = [20,0.5]
			args_PSVM = [20, 1, 5, 0.5]
			args_GPSVM = [20, 1, 0.5]
			# training
			# K = 20, C = 0.5

			# Training:
			# ------ LSVM 
			# Record the start time
			start_time = time.time()
			# K = 20, C = 0.5
			LSVM_model = LSVM(K  = args_LSVM[0],C = args_LSVM[1])
			LSVM_model.fit(dfTrain, Label)
			# Record the end time for LSVM
			train_end_time = time.time()
			LSVM_model.predict(dfTrain)
			predict_end_time = time.time()

			train_elapsed_time_LSVM = train_end_time - start_time
			predict_elapsed_time_LSVM = predict_end_time - start_time

			# ------ PSVM 
			start_time = time.time()
			PSVM_model = PSVM(clusterNum = args_PSVM[0],ensembleNum=args_PSVM[1],C = args_PSVM[2], R = args_PSVM[3], max_iterations =  2000)
			PSVM_model.fit(dfTrain,Label)
			# Record the end time for PSVM
			train_end_time = time.time()
			PSVM_model.predict(dfTrain)

			train_elapsed_time_PSVM = train_end_time - start_time
			predict_elapsed_time_PSVM = predict_end_time - start_time
			predict_end_time = time.time()


			# ------ GPSVM 
			start_time = time.time()
			GPSVM_model = GPSVM(clusterNum = args_GPSVM[0],ensembleNum=args_GPSVM[1],C = args_GPSVM[2])
			GPSVM_model.fit(dfTrain,Label)
			# Record the end time for GPSVM
			train_end_time = time.time()
			GPSVM_model.predict(dfTrain)
			predict_end_time = time.time()

			train_elapsed_time_GPSVM = train_end_time - start_time
			predict_elapsed_time_GPSVM = predict_end_time - start_time

			matrix[j*3,i] = train_elapsed_time_LSVM
			matrix[j*3 + 1,i] = train_elapsed_time_PSVM
			matrix[j*3 + 2,i] = train_elapsed_time_GPSVM

			matrix_total[j*3,i] = predict_elapsed_time_LSVM
			matrix_total[j*3 + 1,i] = predict_elapsed_time_PSVM
			matrix_total[j*3 + 2,i] = predict_elapsed_time_GPSVM


	np.savetxt(outfile, matrix, delimiter=",", header = '' )
	np.savetxt(outfileTotal, matrix_total, delimiter=",", header = '' )






# -------------------------------------- Accuracy comparison  -----------------------------

# POF-Darts

# Pujol, LSVM, PSVM, GPSVM
# bench mark: nnet, rf, lSVM, kSVM, 

















def main():
	df = example1_Data(500,1)
	Running_time(N = [100,200,300,500,800,1000,1500,2000], outfile = 'Results/running_time.csv', outfileTotal = 'Results/running_time_total.csv')
	return





































if __name__ == '__main__':
	main()