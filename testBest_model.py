from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
import pandas as pd
import numpy as np
from dataGeneration import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
from itertools import product
# Suppress specific UserWarning about use_label_encoder deprecation
warnings.filterwarnings("ignore", message="`use_label_encoder` is deprecated in 1.7.0.")

from xgboost import XGBClassifier
from CBP import referenced_method, LSVM, PSVM, GPSVM
from sklearn.base import BaseEstimator, ClassifierMixin
import sys


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
param_grid_SVM = {'C': [0.1, 0.5,1],
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
  'C':[0.1,0.5,1],   
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

def perform_grid_search_cv(estimator, param_grid, X, y, cv=5, scoring='accuracy'):
    """
    A simplified custom grid search CV function without using cross_val_score.

    Parameters:
    - estimator: The machine learning model to tune.
    - param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    - X: Input features (numpy array or pandas DataFrame).
    - y: Target variable (numpy array or pandas Series).
    - cv: Number of cross-validation folds.
    - scoring: A callable to evaluate the predictions on the test set. For simplicity, this will use accuracy.

    Returns:
    - best_params: The parameter setting that gave the best results on the hold out data.
    - best_score: Mean cross-validated score of the best_estimator.
    """
    best_score = -np.inf
    best_params = None
    
    # Generate all combinations of parameters
    from itertools import product
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Define KFold cross-validator
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Iterate over all combinations of parameters
    for params in param_combinations:
        scores = []
        
        # Perform cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Set the parameters and fit the model
            estimator.set_params(**params)
            estimator.fit(X_train, y_train)
            
            # Make predictions and evaluate
            y_pred = estimator.predict(X_test)
            score = np.mean(y_pred == y_test)

                
            scores.append(score)
        
        # Compute the mean score
        mean_score = np.mean(scores)
        
        # Update the best parameters if the current mean score is greater
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    estimator.set_params(**best_params)
    estimator.fit(X,y)
    return estimator



# def perform_grid_search_cv(model, param_grid, X, y, cv=5):
#   """
#   Perform hyperparameter tuning using GridSearchCV and cross-validation.

#   Parameters:
#   - model: Estimator object (e.g., a classifier or regressor).
#   - param_grid: Dictionary of hyperparameters to search.
#   - X: Feature matrix.
#   - y: Target vector.
#   - cv: Number of cross-validation folds (default is 5).

#   Returns:
#   - best_model: The best model with tuned hyperparameters.
#   """
#   #Create a GridSearchCV object
#   grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', verbose = 1, n_jobs = -1)
#   # Fit the grid search to the data
#   grid_search.fit(X, y)
#   # Get the best model with tuned hyperparameters
#   best_model = grid_search.best_estimator_


#   return best_model









def Accuracy_comparison_CV(n , nTest, example, sample_crite = 'POF', repeat = 20):
  reference_classifier = referenced_method()
  # linear_svm = LSVM()
  # kmeans_based_GPSVM = GPSVM(method="KMeans")
  # hierarchical_clustering_GPSVM = GPSVM(method='hierarchicalClustering')
  # random_forest = RandomForestClassifier()
  # mlp_classifier = MLPClassifier()
  # xgboost_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
  # support_vector_classifier = SVC()
  profile_svm = PSVM()  # Assuming PSVM is a placeholder for a specific SVM variant

  Classifier = [
      reference_classifier,
      # linear_svm,
      # kmeans_based_GPSVM,
      # hierarchical_clustering_GPSVM,
      # random_forest,
      # mlp_classifier,
      # xgboost_classifier,
      # support_vector_classifier,
      profile_svm
  ]
  paras = [
  param_grid_pujol, 
  # param_grid_LSVM, 
  # param_grid_GPSVM_Kmeans, 
  # param_grid_GPSVM_Hier, 
  # param_grid_rf, 
  # param_grid_MLP, 
  # param_grid_xgb, 
  # param_grid_SVM,
  param_grid_PSVM
  ]
  
  accuracyMatrixTrain = np.zeros( shape = (repeat, len(Classifier)) )
  accuracyMatrixPrediction = np.zeros( shape = (repeat, len(Classifier)) )
  for i in range(repeat):
    print('Epoch %d' %i + '--------------------------------------' + '\n')
    if example == 'Brusselator':
      domains = [[0.7,1.5], [2.75,3.25], [0,2]]
      dataSIP = SIP_Data(integral_3D, DQ_Dlambda_3D, 3.75, len(domains) , *domains)
    elif example == 'Elliptic':
      domains = [[1,5], [0.1,0.3], [0,1],[0,2]]
      dataSIP = SIP_Data(elliptic_function, elliptic_Gradient, 0, len(domains) , *domains)

    elif example == 'Function1': 
      domains = [[0,2], [0,2] ]
      dataSIP = SIP_Data(function1, Gradient_f1, 0.5, len(domains) , *domains)

    elif example == 'Function2': 
      domains = [[0,1], [0,1] ]
      dataSIP = SIP_Data(function2, Gradient_f2, 1, len(domains) , *domains)
    else:
      raise Valuerror("not a valid example")

    if sample_crite == 'POF':
      dataSIP.generate_POF(n = n, CONST_a = 1.5 ,iniPoints = 10, sampleCriteria = 'k-dDarts')
    else:
      dataSIP.generate_Uniform(n)


    Label = dataSIP.df['Label'].values
    dfTrain = dataSIP.df.iloc[:, :-2].values
    dQ = dataSIP.Gradient

    X_train = dfTrain
    y_train = Label

    dataSIP.generate_Uniform(nTest)
    X_test = dataSIP.df.iloc[:, :-2].values
    y_test = dataSIP.df['Label'].values

    for model, para, k in zip(Classifier, paras, range(len(Classifier))):
      print('-------------------------------------------------')
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
  if len(sys.argv) != 5:
    raise Valuerror('not enough argument')

  # train size, test size, example name = [Brusselator, Elliptic, Function1, Function2], sample method 
  train_size, test_size, example_name, sample_method = sys.argv[1:5]

  accuracyTrain, accuracyPrediction = Accuracy_comparison_CV(int(train_size), int(test_size), str(example_name), str(sample_method))


  filenameTrain = f'../Results/CVresults/Train_accuracy_{train_size}_{test_size}_{example_name}_{sample_method}.csv'
  filenamePredict = f'../Results/CVresults/Prediction_accuracy_{train_size}_{test_size}_{example_name}_{sample_method}.csv'
  np.savetxt(filenameTrain, accuracyTrain, delimiter=",", header = '')
  np.savetxt(filenamePredict, accuracyPrediction, delimiter=",", header = '')



if __name__ == '__main__':
  main()

