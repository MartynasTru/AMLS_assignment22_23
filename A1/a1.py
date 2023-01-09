import random 
import numpy as np
import pandas as pd
import cv2
from cv2 import IMREAD_GRAYSCALE
import os

import sklearn 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
import dlib
from landmarks import extract_features_labels, get_data

from keras.preprocessing import image


 
features_extracted = False





def img_SVM(training_images, training_labels, test_images, test_labels):
    
    classifier = SVC(kernel='sigmoid')

    classifier.fit(training_images, training_labels)

    pred = classifier.predict(test_images)

    print(pred)

    print("Accuracy:", accuracy_score(test_labels, pred) * 100)
    print("Classification report:", classification_report(test_labels, pred))

if(features_extracted == False):
    tr_X, tr_Y, te_X, te_Y = get_data()
    np.savez('A1/extracted_features', tr_X=tr_X, tr_Y=tr_Y, te_X=te_X, te_Y=te_Y)

loaded_data = np.load('A1/extracted_features.npz')
tr_X = loaded_data['tr_X']
tr_Y = loaded_data['tr_Y']
te_X = loaded_data['te_X']
te_Y = loaded_data['te_Y']
print("Shape tr_X:" ,np.shape(tr_X))
print("Shape tr_X:" ,np.shape(te_X))

pred=img_SVM(tr_X.reshape((np.shape(tr_X)[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((np.shape(te_X)[0], 68*2)), list(zip(*te_Y))[0])









# #tunning model

# defining parameter range
# param_grid = {'C': [0.1, 1, 10], 
#               'gamma': [1, 0.1, 0.01],
#               'kernel': ['rbf', 'linear']} 
            
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 2)
# grid.fit(X_train, y_train)

# print best parameter after tuning
# print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)

# results = grid.cv_results_
# for mean_score, params in zip(results["mean_test_score"], results["params"]):
#     print(f"mean_score: {mean_score:.3f}, params: {params}")



# Reading labels csv file
labels = pd.read_csv('Datasets/celeba/labels.csv', sep='\t', header=None, skiprows=[0]) 
labels_test = pd.read_csv('Datasets/celeba_test/labels.csv', sep='\t', header=None, skiprows=[0]) 
# Selecting gender labels
gender_labels = labels.iloc[:, 2] 
gender_labels_test = labels_test.iloc[:, 2] 
# Creating path to all images
img_path = os.path.join('Datasets/celeba/img/') 
img_path_test = os.path.join('Datasets/celeba_test/img/') 
# Reading all image file names for dynamic path
image_names = labels.iloc[:, 1]
image_names_test = labels_test.iloc[:, 1]




# Reading all images into a list
images_ar = []
for img_name in image_names:
    image = cv2.imread((os.path.join(img_path, img_name)), IMREAD_GRAYSCALE)
    image = image.astype('uint8')
    images_ar.append(image)

# Converting images into an array. This is very important as model requires arrays not lists. 
images = np.array(images_ar)
print("Shape of images array before reshaping: ", np.shape(images))
images = images.reshape(np.shape(images)[0], np.shape(images)[1] * np.shape(images)[2])
print("Shape of images array after reshaping: ", np.shape(images))

images_ar_test = []
for img_name in image_names_test:
    image = cv2.imread((os.path.join(img_path_test, img_name)), IMREAD_GRAYSCALE)
    image = image.astype('uint8')
    images_ar_test.append(image)

images_test = np.array(images_ar_test)
print("Shape of images_test array before reshaping: ", np.shape(images_test))
images_test = images_test.reshape(np.shape(images_test)[0], np.shape(images_test)[1] * np.shape(images_test)[2])
print("Shape of images_test array after reshaping: ", np.shape(images_test))

landmark_features, genderr = extract_features_labels()


print ("Splitting data into training sets")
X_train, X_test, y_train, y_test = train_test_split(landmark_features, genderr, test_size = 0.5)



print("Shape of X_train array after split: ", np.shape(X_train))
print("Shape of X_train array after split: ", np.shape(y_train))

model = SVC(kernel = "rbf", C=1000, gamma = 'scale')
#model = 
model.fit(X_train, y_train)

print("Training...")

label_pred = model.predict(X_test)
print(classification_report(y_test, label_pred))
accuracy = accuracy_score(y_test, label_pred) * 100
print("Accuracy:" ,accuracy)




