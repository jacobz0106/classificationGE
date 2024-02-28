import time
start_time = time.time()
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataGeneration import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from xgboost import XGBClassifier
from CBP import refrenced_method, LSVM, PSVM, GPSVM
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

#### Parameter definition:
## - Random Forest:
param_grid_rf = {
  'n_estimators': [100, 200],         # Number of trees in the forest (integer)
  'max_depth': [None, 10, 20, 30],       # Maximum depth of the trees (integer or None)
  'min_samples_split': [2, 5, 10],      # Minimum samples required to split an internal node (integer)
  'min_samples_leaf': [1, 2, 4],        # Minimum number of samples required to be at a leaf node (integer)

}

## - MLPClassifier:
param_grid_MLP = {
  'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],  # Architecture of hidden layers
  'activation': ['logistic', 'tanh', 'relu'],  # Activation function
  'alpha': [0.0001, 0.001, 0.01],  # L2 regularization term
  'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
  'max_iter': [3000, 5000, 10000],  # Maximum number of iterations
}
## - Xgbboost:
param_grid_xgb = {
  'n_estimators': [100, 200, 300],         # Number of boosting rounds (integer)
  'max_depth': [3, 4, 5],                # Maximum tree depth (integer)
  'learning_rate': [0.01, 0.1, 0.2],    # Learning rate (real-valued)
  'min_child_weight': [1, 3, 5],         # Minimum sum of instance weight needed in a child (integer)
  'subsample': [0.7, 0.8, 0.9],         # Fraction of samples used for fitting the trees (real-valued)
  'gamma': [0, 0.1, 0.2],               # Minimum loss reduction required to make a further partition on a leaf node (real-valued)
  'lambda': [0.0, 1.0, 2.0],                  # L2 regularization term (real-valued)
  'alpha': [0.0, 1.0, 2.0],                   # L1 regularization term (real-valued)
}

## - SVM kernel
param_grid_SVM = {'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'poly'],
        }
 


## - referenced_method:
param_grid_pujol = {
  'alpha': [0, 0.3, 0.6],        
  'constLambda': [0.1, 0.5, 1, 1.5],               
}



## - LSVM:
param_grid_LSVM = {
  'K':  [ 5,10,15,20],        
  'C': [0.1, 1.5, 9.5],               
}


## - PSVM:

param_grid_PSVM = {
  'clusterNum':  [ 5,10,15,20],        
  'ensembleNum': [1, 5, 10], 
  'C':[0.1,0.5,1,5,10],   
  'R':[0, 0.1,0.5,1,5,10],
  'max_iterations': [100],            
}

## - GPSVM : 
param_grid_GPSVM_Kmeans = {
  'method' : ["KMeans"], 
  'clusterNum':  [ 5,10,15,20],        
  'ensembleNum': [1, 5, 10], 
  'C':[0.1,0.5,1,5,10],      
}

param_grid_GPSVM_Hier = {
  'clusterNum':  [ 5,10,15,20],        
  'ensembleNum': [1, 5, 10], 
  'C':[0.1,0.5,1,5,10], 
  'method' : ["hierarchicalClustering"],      
}



print("--- %s seconds ---" % (time.time() - start_time))

def perform_grid_search_cv(model, param_grid, X, y, cv=5):
  """
  Perform hyperparameter tuning using GridSearchCV and cross-validation.

  Parameters:
  - model: Estimator object (e.g., a classifier or regressor).
  - param_grid: Dictionary of hyperparameters to search.
  - X: Feature matrix.
  - y: Target vector.
  - cv: Number of cross-validation folds (default is 5).

  Returns:
  - best_model: The best model with tuned hyperparameters.
  """
  # Create a GridSearchCV object
  print('start grid search')
  grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
  print('found optimal solution')
  # Fit the grid search to the data
  grid_search.fit(X, y)
  # Get the best model with tuned hyperparameters
  best_model = grid_search.best_estimator_

  return best_model









def Accuracy_comparison_CV(n , nTest, repeat = 20):
  #Classifier = [refrenced_method(), LSVM(), GPSVM(method = "KMeans"),GPSVM(method = 'hierarchicalClustering'),  RandomForestClassifier(), MLPClassifier(), XGBClassifier(use_label_encoder=False, eval_metric = 'logloss'), SVC(), PSVM()]
  print(1)
  Classifier = [svm.SVC()]
  #paras = [param_grid_rf, param_grid_MLP, param_grid_xgb, param_grid_SVM, param_grid_pujol, param_grid_LSVM, param_grid_PSVM , param_grid_GPSVM]
  paras = [param_grid_SVM]
  print(2)

  accuracyMatrixTrain = np.zeros( shape = (repeat, len(Classifier)) )
  accuracyMatrixPrediction = np.zeros( shape = (repeat, len(Classifier)) )
  for i in range(repeat):
    print('Epoch %d' %i + '--------------------------------------' + '\n')
    domains = [[1,5], [0.1,0.3], [0,1],[0,2]]
    dataSIP = SIP_Data(elliptic_function, elliptic_Gradient, 0, len(domains) , *domains)
    dataSIP.generate_POF(n = 100, CONST_a = 1.5 ,iniPoints = 10, sampleCriteria = 'k-dDarts')



    Label = dataSIP.df['Label'].values
    dfTrain = dataSIP.df[['X1','X2','X3', 'X4']].values

    # # Generating indices
    # indices = np.arange(len(Label))
    # train_indices, test_indices = train_test_split(indices, test_size=0.25, random_state=42)


    dQ = dataSIP.POFdarts.Q

    X_train = dfTrain
    y_train = Label

    dataSIP.generate_Uniform(nTest)

    X_test = dataSIP.df[['X1','X2','X3', 'X4']].values
    y_test = dataSIP.df['Label'].values
    for model, para, k in zip(Classifier, paras, range(len(Classifier))):
      print('Tunning model:' + str(model)+ '\n')
      print('with parameters:' +  str(para) + '\n')

      # encode for Xgboost
      if isinstance(model, XGBClassifier):
        xgb_y_train = [0 if label == -1 else 1 for label in y_train]
        xgb_y_test= [0 if label == -1 else 1 for label in y_test]
        best_model = perform_grid_search_cv(model, para, X_train,xgb_y_train)
        
        trainAccuracy = np.sum(best_model.predict(X_train) == xgb_y_train)/len(xgb_y_train)
        predictionAccuracy = np.sum(best_model.predict(X_test) == xgb_y_test)/len(xgb_y_test)
        accuracyMatrixTrain[i, k] =  trainAccuracy
        accuracyMatrixPrediction[i, k] = predictionAccuracy
        print('training samples, best model has train accuracy: %f' %trainAccuracy + ' prediction accuracy:%f' %predictionAccuracy + '\n')
      elif isinstance(model, GPSVM):
        if model.method == "hierarchicalClustering":
          # Create a GridSearchCV object
          fit_para = {'dQ':dQ}
          grid_search = GridSearchCV(model, para, cv=5, scoring='accuracy')

          # Fit the grid search to the data
          grid_search.fit(X_train, y_train, **fit_para)

          # Get the best model with tuned hyperparameters
          best_model = grid_search.best_estimator_

          trainAccuracy = np.sum(best_model.predict(X_train) == y_train)/len(y_train)
          predictionAccuracy = np.sum(best_model.predict(X_test) == y_test)/len(y_test)
          accuracyMatrixTrain[i, k] =  trainAccuracy
          accuracyMatrixPrediction[i, k] = predictionAccuracy
          print('training samples, best model has train accuracy: %f' %trainAccuracy + ' prediction accuracy:%f' %predictionAccuracy + '\n')
        else:
          model = GPSVM(method = "KMeans")
          best_model = perform_grid_search_cv(model, para, X_train,y_train)
          trainAccuracy = np.sum(best_model.predict(X_train) == y_train)/len(y_train)
          predictionAccuracy = np.sum(best_model.predict(X_test) == y_test)/len(y_test)
          accuracyMatrixTrain[i, k] =  trainAccuracy
          accuracyMatrixPrediction[i, k] = predictionAccuracy
          print('training samples, best model has train accuracy: %f' %trainAccuracy + ' prediction accuracy:%f' %predictionAccuracy + '\n')
      else:
        best_model = perform_grid_search_cv(model, para, X_train,y_train)
        trainAccuracy = np.sum(best_model.predict(X_train) == y_train)/len(y_train)
        predictionAccuracy = np.sum(best_model.predict(X_test) == y_test)/len(y_test)
        accuracyMatrixTrain[i, k] =  trainAccuracy
        accuracyMatrixPrediction[i, k] = predictionAccuracy
        print('training samples, best model has train accuracy: %f' %trainAccuracy + ' prediction accuracy:%f' %predictionAccuracy + '\n')
  return accuracyMatrixTrain, accuracyMatrixPrediction


def main():
  accuracyTrain, accuracyPrediction = Accuracy_comparison_CV(100, 500)



if __name__ == '__main__':
  main()

