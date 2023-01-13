import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from landmarks import get_data
from output import logging, model_classification
from graph_drawing import plot_grid_search

global start_time

start_time = time.time()
features_extracted = True

def SVM_fit(X_train, y_train):

    # Choosing parameter for SVM model
    param_grid = {"C": [0.01, 0.1, 1], "gamma": [1, 0.1, 0.01], "kernel": ["linear"]}

    grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=3)
    grid.fit(X_train, y_train)

    return grid

def KNN_fit(X_train, y_train):

    #Choosing best parameter for KNN model
    k_range = list(range(1, 150))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, verbose=3)
    grid.fit(X_train, y_train)

    return grid

def RF_fit(X_train, y_train):

    #Choosing best parameter for Random Forest model
    k_range = list(range(10, 300))
    param_grid = dict(n_estimators=k_range)
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, verbose=3)
    grid.fit(X_train, y_train)

    return grid

def model_train(training_images, training_labels, test_images, test_labels):

    #Random Forest model was used for this task
    model = RandomForestClassifier(n_estimators=51)
    model.fit(training_images, training_labels)
    y_pred = model.predict(test_images)
    class_rep = classification_report(test_labels, y_pred)
    # Logging data
    model_classification(class_rep)

def SVM_main(training_images, training_labels, test_images, test_labels):

    model_name = "SVM"
    model = SVM_fit(training_images, training_labels)

    best_parameter = model.best_params_
    best_est = model.best_estimator_
    best_model = best_est.score(test_images, test_labels)
    report = model.cv_results_

    y_pred = model.predict(test_images)
    conf_matrix = confusion_matrix(test_labels, y_pred)
    
    # Ending Timer
    end_time = time.time()
    total_runtime = end_time - start_time

    # Sending data to plot
    plot_grid_search(model.cv_results_, model_name, conf_matrix)

    # Saving all data into logs.txt file for later reference
    logging(total_runtime, model_name, report, best_parameter, best_model)

def KNN_main(training_images, training_labels, test_images, test_labels):

    model_name = "KNN"
    model = KNN_fit(training_images, training_labels)

    best_parameter = model.best_params_
    best_est = model.best_estimator_
    best_model = best_est.score(test_images, test_labels)
    report = model.cv_results_

    y_pred = model.predict(test_images)
    conf_matrix = confusion_matrix(test_labels, y_pred)

    # Ending Timer
    end_time = time.time()
    total_runtime = end_time - start_time

    # Sending data to plot
    plot_grid_search(model.cv_results_, model_name, conf_matrix)

    # Saving all data into logs.txt file for later reference
    logging(total_runtime, model_name, report, best_parameter, best_model)

def RF_main(training_images, training_labels, test_images, test_labels):
    
    #Random Forest model was used for this task
    model_name = "RF"
    model = RF_fit(training_images, training_labels)

    best_parameter = model.best_params_
    best_est = model.best_estimator_
    best_model = best_est.score(test_images, test_labels)
    report = model.cv_results_

    y_pred = model.predict(test_images)
    conf_matrix = confusion_matrix(test_labels, y_pred)

    # Ending Timer
    end_time = time.time()
    total_runtime = end_time - start_time

    train_sizes, train_scores, val_scores = learning_curve(
        model.best_estimator_, training_images, training_labels, cv=5
    )

    # Sending data to plot
    plot_grid_search(model.cv_results_, model_name, conf_matrix, train_sizes, train_scores, val_scores)

    # Saving all data into logs.txt file for later reference
    logging(total_runtime, model_name, report, best_parameter, best_model)

if features_extracted == True:

    loaded_data = np.load("B2/extracted_features.npz")
    tr_X = loaded_data["tr_X"]
    tr_Y = loaded_data["tr_Y"]
    te_X = loaded_data["te_X"]
    te_Y = loaded_data["te_Y"]

    RF_main(tr_X.reshape((np.shape(tr_X)[0], 3*2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 3*2)), list(zip(*te_Y))[0])
    model_train(tr_X.reshape((np.shape(tr_X)[0], 3 * 2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 3 * 2)), list(zip(*te_Y))[0])

else:

    tr_X, tr_Y, te_X, te_Y = get_data()
    np.savez("B2/extracted_features", tr_X=tr_X, tr_Y=tr_Y, te_X=te_X, te_Y=te_Y)
    SVM_main(tr_X.reshape((np.shape(tr_X)[0], 3 * 2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 3 * 2)), list(zip(*te_Y))[0])
