import os
import warnings
# Set environment variable to suppress warnings at Python level
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress all warnings (must be before other imports)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
# Specifically suppress FutureWarnings from pandas
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress warnings from imported modules
warnings.filterwarnings("ignore", module="dataGeneration")
# Suppress joblib/loky resource_tracker warnings
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", module="joblib.externals.loky.backend.resource_tracker")
warnings.filterwarnings("ignore", module="multiprocessing.resource_tracker")
warnings.filterwarnings("ignore", message="There appear to be.*leaked.*objects")
warnings.filterwarnings("ignore", message=".*FileNotFoundError.*")
# Suppress specific UserWarning about use_label_encoder deprecation
warnings.filterwarnings("ignore", message="`use_label_encoder` is deprecated in 1.7.0.")

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
import pandas as pd
# Suppress pandas FutureWarnings
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

import numpy as np
from dataGeneration import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from CBP import referenced_method, LSVM, PSVM, GPSVM, GMSVM, GMSVM_reduced
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
import sys
# ADD near other imports
from joblib import Parallel, delayed
from tqdm import tqdm





#### Parameter definition:
# Working on:
# normalize the SVM,
# linear SVM with penalty terms


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
  # 'gamma': [0, 0.1, 0.2],               # Minimum loss reduction required to make a further partition on a leaf node (real-valued)
  # 'lambda': [0.0, 1.0, 2.0],                  # L2 regularization term (real-valued)
  # 'alpha': [0.0, 1.0, 2.0],                   # L1 regularization term (real-valued)
}

## - SVM kernel
param_grid_SVM = {'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'poly'],
        }
 
param_grid_knn = {
  'n_neighbors': [1,3,5,7],
}


## - referenced_method:
param_grid_pujol = {
  'alpha': [0, 0.3, 0.6],        
  'constLambda': [0.1, 0.5, 1, 1.5],               
}



## - LSVM:
param_grid_LSVM = {
  'K':  [ 5,10,15,20],        
  'C': [1,5,10,100, 500, 1000,2000],               
}


## - PSVM:

param_grid_PSVM = {
  'clusterNum':  [ 5,8,12, 15],        
  'ensembleNum': [1, 3, 5, 7], 
  'C':[0.1,1,10,100, 1000],   
  'R':[0, 0.1,0.5,1,5],
  'max_iterations': [100],            
}

## - GPSVM : 
param_grid_GPSVM_Kmeans = {
  'method' : ["KMeans"], 
  'clusterNum':  [ 5,10,12, 15,20],        
  'ensembleNum': [1, 3, 5, 7], 
  'C':[0.1,1,10,100,1000],      
}

# param_grid_GPSVM_Hier = {
#   'clusterNum':  [ 5,10,12, 15,20],        
#   'ensembleNum': [1, 3, 5, 7], 
#   'C':[0.05, 0.1, 0.5,1,5], 
#   'method' : ["hierarchicalClustering"], 
#   'CONST_C': [0.1,0.5,1,5, 10],     
# }


param_grid_GMSVM = { 
'clusterSize': [2,3,4,6,8],     
'ensembleNum': [1, 3, 5], 
'C':[0.1,1, 10],     
'K':[0,0.5,1,10] 
  }

param_grid_GMSVM_reduced = {  
'clusterSize': [2,3,4,6,8],    
'ensembleNum': [1, 3, 5], 
'C':[0.1,1, 10],     
'K':[0,0.5,1,10]
  }


def perform_grid_search_cv(model, param_grid, X, y, cv=5, n_jobs=1, **fit_params):
  """
  Perform hyperparameter tuning using GridSearchCV and cross-validation.
  """
  grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=n_jobs)
  grid_search.fit(X, y, **fit_params)
  best_model = grid_search.best_estimator_
  #print(best_model.get_params())
  return best_model






def Accuracy_comparison_CV(n, nTest, example, sample_crite='POF', repeat=20,
                           n_jobs_outer=4, n_jobs_cv=1, xgb_n_jobs=1, verbose=True):
  # construct fresh estimators
  reference_classifier   = referenced_method()
  localized_linear_svm   = LSVM()
  kmeans_based_GPSVM     = GPSVM(method="KMeans")
  GMSVM_model_reduced    = GMSVM_reduced()
  random_forest          = RandomForestClassifier()
  mlp_classifier         = MLPClassifier()
  xgboost_classifier     = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=xgb_n_jobs)
  support_vector_classifier = SVC()
  profile_svm            = PSVM()
  knn                    = KNeighborsClassifier()

  Classifier = [
      knn,
      reference_classifier,
      localized_linear_svm,
      kmeans_based_GPSVM,
      GMSVM_model_reduced,
      random_forest,
      mlp_classifier,
      xgboost_classifier,
      support_vector_classifier,
      profile_svm
  ]
  paras = [
      param_grid_knn,
      param_grid_pujol,
      param_grid_LSVM,
      param_grid_GPSVM_Kmeans,
      param_grid_GMSVM_reduced,
      param_grid_rf,
      param_grid_MLP,
      param_grid_xgb,
      param_grid_SVM,
      param_grid_PSVM
  ]

  # ----- helpers -----
  def _make_data():
    if example == 'Brusselator':
      domains = [[0.7, 1.5], [2.75, 3.25], [0, 2]]
      dataSIP = SIP_Data(integral_3D, DQ_Dlambda_3D, 3.47, len(domains), *domains)
    elif example == 'Elliptic':
      domains = [[1, 5], [0.1, 0.3], [0, 1], [0, 2]]
      dataSIP = SIP_Data(elliptic_function, elliptic_Gradient, 0, len(domains), *domains)
    elif example == 'Function1':
      domains = [[0, 2], [0, 2]]
      dataSIP = SIP_Data(function1, Gradient_f1, 0.5, len(domains), *domains)
    elif example == 'Function2':
      domains = [[-1, 1], [-1, 1]]
      dataSIP = SIP_Data(function2, Gradient_f2, 1, len(domains), *domains)
    else:
      raise ValueError("not a valid example")

    if sample_crite == 'POF':
      dataSIP.generate_POF(n=n, CONST_a=1, iniPoints=10, sampleCriteria='k-dDarts')
    else:
      dataSIP.generate_Uniform(n)

    y_train = dataSIP.df['Label'].values
    X_train = dataSIP.df.iloc[:, :-2].values
    dQ = dataSIP.Gradient

    dataSIP.generate_Uniform(nTest)
    X_test = dataSIP.df.iloc[:, :-2].values
    y_test = dataSIP.df['Label'].values
    return X_train, y_train, X_test, y_test, dQ

  def _run_one_epoch(epoch_idx):
    if verbose:
      print(f'[Epoch {epoch_idx}] --------------------------------------\n')

    X_train, y_train, X_test, y_test, dQ = _make_data()
    train_row = np.zeros(len(Classifier))
    pred_row  = np.zeros(len(Classifier))

    for k, (model, para) in enumerate(zip(Classifier, paras)):
      if verbose:
        print(f'[Epoch {epoch_idx}] Tuning model: {model}')
        print(f'[Epoch {epoch_idx}] with parameters: {para}\n')

      # XGBoost needs {0,1} labels
      if isinstance(model, XGBClassifier):
        xgb_y_train = np.array([0 if label == -1 else 1 for label in y_train])
        xgb_y_test  = np.array([0 if label == -1 else 1 for label in y_test])
        best_model  = perform_grid_search_cv(model, para, X_train, xgb_y_train, cv=5, n_jobs=n_jobs_cv)
        train_acc   = (best_model.predict(X_train) == xgb_y_train).mean()
        pred_acc    = (best_model.predict(X_test)  == xgb_y_test ).mean()

      elif isinstance(model, GPSVM):
        fit_para = {'dQ': dQ}
        if getattr(model, 'method', None) == "hierarchicalClustering":
          grid_search = GridSearchCV(model, para, cv=5, scoring='accuracy', verbose=1, n_jobs=n_jobs_cv)
          grid_search.fit(X_train, y_train, **fit_para)
          best_model = grid_search.best_estimator_
        else:
          # force KMeans variant
          model = GPSVM(method="KMeans")
          best_model = perform_grid_search_cv(model, para, X_train, y_train, cv=5, n_jobs=n_jobs_cv, **fit_para)

        train_acc = (best_model.predict(X_train) == y_train).mean()
        pred_acc  = (best_model.predict(X_test)  == y_test ).mean()

      elif isinstance(model, GMSVM_reduced):
        fit_para   = {'dQ': dQ}
        grid_search = GridSearchCV(model, para, cv=5, scoring='accuracy', verbose=1, n_jobs=n_jobs_cv)
        grid_search.fit(X_train, y_train, **fit_para)
        best_model = grid_search.best_estimator_
        train_acc  = (best_model.predict(X_train) == y_train).mean()
        pred_acc   = (best_model.predict(X_test)  == y_test ).mean()

      else:
        best_model = perform_grid_search_cv(model, para, X_train, y_train, cv=5, n_jobs=n_jobs_cv)
        train_acc  = (best_model.predict(X_train) == y_train).mean()
        pred_acc   = (best_model.predict(X_test)  == y_test ).mean()

      train_row[k] = train_acc
      pred_row[k]  = pred_acc
      if verbose:
        print(f'[Epoch {epoch_idx}] train acc: {train_acc:.6f}  pred acc: {pred_acc:.6f}\n')

    return train_row, pred_row

  # ---- run epochs in parallel (process-based; estimators must be picklable) ----
  results = Parallel(n_jobs=n_jobs_outer, backend="loky", verbose=0)(
      delayed(_run_one_epoch)(i) for i in tqdm(range(repeat), desc=f"Epochs (n={n})", disable=not verbose)
  )

  accuracyMatrixTrain      = np.vstack([r[0] for r in results])
  accuracyMatrixPrediction = np.vstack([r[1] for r in results])
  return accuracyMatrixTrain, accuracyMatrixPrediction



def _run_single_train_size(train_size, test_size, example_name, sample_method, 
                           n_jobs_outer=4, n_jobs_cv=1, xgb_n_jobs=2, verbose=True):
  """
  Run a single train_size experiment. This function is designed to be called in parallel.
  """
  print(f'\n================ Running train_size={train_size} ================\n')

  accuracyTrain, accuracyPrediction = Accuracy_comparison_CV(
      int(train_size), test_size, str(example_name), str(sample_method),
      repeat=20, n_jobs_outer=n_jobs_outer, n_jobs_cv=n_jobs_cv, xgb_n_jobs=xgb_n_jobs, verbose=verbose
  )

  filenameTrain   = f'../Results/CVresults/Train_accuracy_{train_size}_{test_size}_{example_name}_{sample_method}.csv'
  filenamePredict = f'../Results/CVresults/Prediction_accuracy_{train_size}_{test_size}_{example_name}_{sample_method}.csv'
  np.savetxt(filenameTrain, accuracyTrain, delimiter=",", header='')
  np.savetxt(filenamePredict, accuracyPrediction, delimiter=",", header='')
  print(f'[train_size={train_size}] done.')
  return train_size


def main():
  TRAIN_SIZES = [20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 750, 1000]

  # Expect only: test_size, example_name, sample_method
  if len(sys.argv) != 4:
    print(sys.argv)
    print('argument:', len(sys.argv))
    raise ValueError('usage: python script.py <test_size> <example_name> <sample_method>')

  test_size, example_name, sample_method = sys.argv[1:4]
  test_size = int(test_size)

  # Parallel knobs (tune as you like)
  N_OUTER = 2      # epochs in parallel (within each train_size experiment)
  N_CV    = 1      # threads per GridSearchCV
  XGB_J   = 1      # threads inside XGBoost
  N_TRAIN_SIZES = 5  # number of train_size experiments to run in parallel

  # Run train_size experiments in parallel
  print(f'\n========== Running {len(TRAIN_SIZES)} train_size experiments ==========')
  print(f'Running up to {N_TRAIN_SIZES} train_size experiments in parallel\n')
  
  # Use tqdm to show progress for train_size experiments
  results = Parallel(n_jobs=N_TRAIN_SIZES, backend="loky", verbose=0)(
      delayed(_run_single_train_size)(
          train_size, test_size, example_name, sample_method,
          n_jobs_outer=N_OUTER, n_jobs_cv=N_CV, xgb_n_jobs=XGB_J, verbose=True
      ) for train_size in tqdm(TRAIN_SIZES, desc="Train sizes", position=0, leave=True)
  )
  
  print(f'\n========== All {len(TRAIN_SIZES)} train_size experiments completed ==========')


if __name__ == '__main__':
  main()

