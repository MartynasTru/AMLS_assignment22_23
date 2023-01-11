import numpy as np
import pandas as pd
from cv2 import IMREAD_GRAYSCALE
import time
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
import random
from sklearn.ensemble import RandomForestClassifier
import dlib
from landmarks import extract_features_labels, get_data
from output import logging
from output import model_classification
from keras.preprocessing import image
from graph_drawing import plot_grid_search
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier



global start_time 



start_time = time.time()
features_extracted = True

#defining parameter range
def grid_fitting_svm(X_train, y_train):

    param_grid = {'C': [0.01, 0.1, 1, 10], 
                'gamma': [1, 0.1, 0.01],
                'kernel': ['linear']} 

    grid = GridSearchCV(SVC(), param_grid,  refit = True, verbose = 3, cv = 5)
    grid.fit(X_train, y_train)


    results = grid.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"mean_score: {mean_score:.3f}, params: {params}")
    return grid, param_grid

def grid_fitting_knn(X_train, y_train):

    k_range = list(range(1, 300))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid,  cv = 5, verbose = 3)
    grid.fit(X_train, y_train)


    results = grid.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"mean_score: {mean_score:.3f}, params: {params}")
    return grid, param_grid

def grid_fitting_forest(X_train, y_train):

    k_range = list(range(10, 600))
    param_grid = dict(n_estimators=k_range)
    grid = GridSearchCV(RandomForestClassifier(), param_grid,  cv = 3, verbose = 3)
    grid.fit(X_train, y_train)


    results = grid.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"mean_score: {mean_score:.3f}, params: {params}")
    return grid, param_grid

def model_test(training_images, training_labels, test_images, test_labels):
    model = svm.SVC(C = 0.01, gamma=1, kernel='linear')
    model.fit(training_images, training_labels)
    y_pred = model.predict(test_images)
    class_rep = classification_report(test_labels, y_pred)
    model_classification(class_rep)


def img_SVM(training_images, training_labels, test_images, test_labels):
    model = "SVM"

    grid, param_grid = grid_fitting_svm(training_images, training_labels)

    print ("best parameter after tuning")
    print(grid.best_params_)
    best_parameter = grid.best_params_
    print("test_score with best model:")
    print(grid.best_estimator_)
    # Evaluate the best model on the test set
    best_est = grid.best_estimator_
    best_model = best_est.score(test_images, test_labels)
    print("best-model", best_model)
    print ("Results")
    print(grid.cv_results_)
    report = grid.cv_results_
    print("More results:\n")
    results = grid.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"mean_score: {mean_score:.3f}, params: {params}")

    
    end_time = time.time()
    total_runtime = end_time - start_time

    #graph(scores, param_grid)
    plot_grid_search(grid.cv_results_, param_grid)
    logging(total_runtime, model, report, best_parameter, best_model)
    
def img_KNN(training_images, training_labels, test_images, test_labels):
    model = "KNN"

    grid, param_grid = grid_fitting_knn(training_images, training_labels)

    print ("best parameter after tuning")
    print(grid.best_params_)
    best_parameter = grid.best_params_
    print("test_score with best model:")
    print(grid.best_estimator_)
    # Evaluate the best model on the test set
    best_est = grid.best_estimator_
    best_model = best_est.score(test_images, test_labels)
    print("best-model", best_model)
    print ("Results")
    print(grid.cv_results_)
    report = grid.cv_results_
    print("More results:\n")
    results = grid.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"mean_score: {mean_score:.3f}, params: {params}")

    end_time = time.time()
    total_runtime = end_time - start_time

    #graph(scores, param_grid)
    plot_grid_search(grid.cv_results_, param_grid)
    logging(total_runtime, model, report, best_parameter, best_model)    

def img_FOREST(training_images, training_labels, test_images, test_labels):
    model = "Random Forest Classifier"

    grid, param_grid = grid_fitting_forest(training_images, training_labels)

    print ("best parameter after tuning")
    print(grid.best_params_)
    best_parameter = grid.best_params_
    print("test_score with best model:")
    print(grid.best_estimator_)
    # Evaluate the best model on the test set
    best_est = grid.best_estimator_
    best_model = best_est.score(test_images, test_labels)
    print("best-model", best_model)
    print ("Results")
    print(grid.cv_results_)
    report = grid.cv_results_
    print("More results:\n")
    results = grid.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"mean_score: {mean_score:.3f}, params: {params}")

    end_time = time.time()
    total_runtime = end_time - start_time

    #graph(scores, param_grid)
    plot_grid_search(grid.cv_results_, param_grid)
    logging(total_runtime, model, report, best_parameter, best_model)

if(features_extracted == False):
    tr_X, tr_Y, te_X, te_Y = get_data()
    print("Shape tr_X:" ,np.shape(tr_X))
    print("Shape tr_Y:" ,np.shape(tr_Y))
    print("Shape te_X:" ,np.shape(te_X))
    print("Shape te_Y:" ,np.shape(te_Y))
    np.savez('A2/extracted_features', tr_X=tr_X, tr_Y=tr_Y, te_X=te_X, te_Y=te_Y)
    pred=img_SVM(tr_X.reshape((np.shape(tr_X)[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 68*2)), list(zip(*te_Y))[0])

else:

    loaded_data = np.load('A2/extracted_features.npz')
    tr_X = loaded_data['tr_X']
    tr_Y = loaded_data['tr_Y']
    te_X = loaded_data['te_X']
    te_Y = loaded_data['te_Y']

    print("Shape tr_X:" ,np.shape(tr_X))
    print("Shape tr_Y:" ,np.shape(tr_Y))
    print("Shape te_X:" ,np.shape(te_X))
    print("Shape te_Y:" ,np.shape(te_Y))
    print("1 tr_X:" , tr_X[1])
    print("2 tr_Y:" , tr_Y[2])
    print("3 te_X:" , te_X[3])
    print("4 te_Y:" , te_Y[4])

    #model_test(tr_X.reshape((np.shape(tr_X)[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 68*2)), list(zip(*te_Y))[0])
    #pred=img_SVM(tr_X.reshape((np.shape(tr_X)[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 68*2)), list(zip(*te_Y))[0])
    #pred=img_KNN(tr_X.reshape((np.shape(tr_X)[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 68*2)), list(zip(*te_Y))[0])
    #pred=img_FOREST(tr_X.reshape((np.shape(tr_X)[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 68*2)), list(zip(*te_Y))[0])










