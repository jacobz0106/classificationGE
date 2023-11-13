import time
from sklearn.preprocessing import StandardScaler
from dataGeneration import *
import pandas as pd


def timeComparisonPOF(outfile, repeat = 20, N =[100,200,400,600,1000,1500], iniPoints = 10):
	'''
	CPU time comparison between kd-darts and accept-reject
	'''
	# create matrix to store running time
	matrix = np.zeros((repeat*len(N), 3))
	domains = [[0.7,1.5], [2.75,3.25], [0,2]]
	for n, i in zip(N, range(len(N))):
		for j in range(repeat):
			print([n,j])
			
			matrix[i*len(N) + j,0] = n
			dataSIP_acceptreject = SIP_Data(integral_3D, DQ_Dlambda_3D, 3.75, len(domains) , *domains)
			dataSIP_kdDart = SIP_Data(integral_3D, DQ_Dlambda_3D, 3.75, len(domains) , *domains)

			start_time = time.time()
			dataSIP_acceptreject.generate_POF(n = N[i], CONST_a = 1.5 ,iniPoints = iniPoints, sampleCriteria = 'accept-reject')
			end_time = time.time()

			matrix[i*len(N) + j,1] = end_time - start_time

			start_time = time.time()
			dataSIP_kdDart.generate_POF(n = N[i], CONST_a = 1.5 ,iniPoints = iniPoints, sampleCriteria = 'k-dDarts')
			end_time = time.time()

			matrix[i*len(N) + j,2] = end_time - start_time

	df = pd.DataFrame(matrix, columns=['n','accept-reoject', 'k-dDarts'])
	df.to_csv(outfile, index=False)

def main():
	timeComparisonPOF(outfile = 'Results/timeComparisonPofAcceptReject.py',repeat = 20, N =[100,200,400,600,1000,1500], iniPoints = 10)












if __name__ == '__main__':
	main()