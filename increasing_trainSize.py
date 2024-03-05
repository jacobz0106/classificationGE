from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataGeneration import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from CBP import referenced_method, LSVM, PSVM, GPSVM
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold


from best_model_gridsearchCV_bruss import Accuracy_comparison_CV




def main():
	if len(sys.argv) != 3:
		raise Valuerror('not enough argument')

	nTrain_list = [20,40,60,80, 130, 160,200,400,600,800,1000]
	test_size = 1000 
	example_name, sample_method = sys.argv[1:3]
	for index,train_size in enumerate(nTrain_list):
		accuracyTrain, accuracyPrediction = Accuracy_comparison_CV(int(train_size), int(test_size), str(example_name), str(sample_method))
		filenameTrain = f'../Results/incTrainSizeCVresults/Train_accuracy_{train_size}_{test_size}_{example_name}_{sample_method}.csv'
		filenamePredict = f'../Results/incTrainSizeCVresults/Prediction_accuracy_{train_size}_{test_size}_{example_name}_{sample_method}.csv'
		np.savetxt(filenameTrain, accuracyTrain, delimiter=",", header = '')
		np.savetxt(filenamePredict, accuracyPrediction, delimiter=",", header = '')










if __name__ == '__main__':
	main()