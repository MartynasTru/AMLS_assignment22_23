import matplotlib.pyplot as plt
from matplotlib import cm
import os
import numpy as np
import pandas as pd

    
def plot_grid_search(cv_results, param_grid):

    model_name = "forest"
    if(model_name == "svm"):
        scores_mean = cv_results['mean_test_score'] * 100
        param_C = [result['C'] for result in cv_results['params']]
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.scatter(param_C, scores_mean, label = 'Test Score vs C', s= 80, alpha=1)    
        plt.grid(which='both', linestyle=':', linewidth=1.5)
        plt.xlabel('Parameter C tunning', fontsize = 27)
        plt.ylabel('Accuracy Score, %', fontsize = 27)
        plt.tick_params(axis='both', which='major', labelsize=16.5)
        plt.title('SVM model. Linear kernel. C performance', fontsize = 32)
        plt.savefig('A1/graphs/SVC model C performance, linear.png')

        scores_mean = cv_results['mean_test_score'] * 100
        param_gamma = [result['gamma'] for result in cv_results['params']]
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.scatter(param_gamma, scores_mean, label = 'Test Score vs C', s= 80, alpha=1)    
        plt.grid(which='both', linestyle=':', linewidth=1.5)
        plt.xlabel('Parameter C tunning', fontsize = 27)
        plt.ylabel('Accuracy Score, %', fontsize = 27)
        plt.tick_params(axis='both', which='major', labelsize=16.5)
        plt.title('SVM model. Linear kernel. Gamma performance', fontsize = 32)
        plt.savefig('A1/graphs/SVC model gamma performance.png')
    elif(model_name == "knn"):
        scores_mean = cv_results['mean_test_score'] * 100
        param_n = [result['n_neighbors'] for result in cv_results['params']]
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.scatter(param_n, scores_mean, label = 'Test Score vs n', s= 80, alpha=1)    
        plt.grid(which='both', linestyle=':', linewidth=1.5)
        plt.xlabel('Number of neighbours, N', fontsize = 27)
        plt.ylabel('Accuracy Score, %', fontsize = 27)
        plt.tick_params(axis='both', which='major', labelsize=16.5)
        plt.title('KNN Regressiom model. N performance', fontsize = 32)
        plt.savefig('A1/graphs/KNN model N performance, linear.png')

    elif(model_name == "forest"):
        scores_mean = cv_results['mean_test_score'] * 100
        param_n = [result['n_estimators'] for result in cv_results['params']]
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.scatter(param_n, scores_mean, label = 'Test Score vs n', s= 80, alpha=1)    
        plt.grid(which='both', linestyle=':', linewidth=1.5)
        plt.xlabel('Number of estimators, N', fontsize = 27)
        plt.ylabel('Accuracy Score, %', fontsize = 27)
        plt.tick_params(axis='both', which='major', labelsize=16.5)
        plt.title('Random Forest model. N_estimators performance', fontsize = 32)
        plt.savefig('A1/graphs/RF model N_estimators performance.png')
    




    # scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # # Plot Grid search scores

    # _, ax = plt.subplots(1,1)

    # # Param1 is the X-axis, Param 2 is represented as a different curve (color line)

    # for idx, val in enumerate(grid_param_2):

    #     ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    # ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')

    # ax.set_xlabel(name_param_1, fontsize=16)

    # ax.set_ylabel('CV Average Score', fontsize=16)

    # ax.legend(loc="best", fontsize=15)

    # ax.grid('on')

#plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')
