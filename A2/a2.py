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
from keras.preprocessing import image


global start_time 
basedir = 'Datasets/'
images_dir = os.path.join(basedir,'celeba/img/')
labels_filename = 'celeba/labels.csv'


start_time = time.time()
features_extracted = False

#defining parameter range
def grid_fitting(X_train, y_train):

    param_grid = {'C': [0.01, 0.1, 1, 10], 
                'gamma': [10, 1, 0.1, 0.01],
                'kernel': ['rbf', 'linear']} 

    grid = GridSearchCV(SVC(), param_grid,  refit = True, verbose = 3, cv = 5)
    grid.fit(X_train, y_train)


    results = grid.cv_results_
    for mean_score, params in zip(results["mean_test_score"], results["params"]):
        print(f"mean_score: {mean_score:.3f}, params: {params}")
    return grid



def img_SVM(training_images, training_labels, test_images, test_labels):
    model = "SVC"

    grid = grid_fitting(training_images, training_labels)

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

    end_time = time.time()
    total_runtime = end_time - start_time
    
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

    # # Zip the data together into a single list
    # data = list(zip(tr_X, tr_Y))
    # data = list(zip(tr_X, tr_Y))
    # # Shuffle the data
    # random.shuffle(data)

    # # Unzip the data back into separate arrays
    # tr_X, tr_Y, te_X, te_Y = zip(*data)
    # tr_X = np.array(tr_X)
    # te_X = np.array(te_X)
    # tr_Y = np.array(tr_Y)
    # te_Y = np.array(te_Y)

    pred=img_SVM(tr_X.reshape((np.shape(tr_X)[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 68*2)), list(zip(*te_Y))[0])


#tunning model







