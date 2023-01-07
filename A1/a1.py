
import numpy as np
import pandas as pd
import cv2
from cv2 import IMREAD_GRAYSCALE
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn as sk
from sklearn.svm import SVC
import random 

# Reading labels csv file
labels = pd.read_csv('Datasets/celeba_test/labels.csv', sep='\t', header=None, skiprows=[0]) 
# Selecting gender labels
gender_labels = labels.iloc[:, 2] 
# Creating path to all images
img_path = os.path.join('Datasets/celeba_test/img/') 
# Reading all image file names for dynamic path
image_names = labels.iloc[:, 1]

# Reading all images into a list
images_ar = []
for img_name in image_names:
    image = cv2.imread((os.path.join(img_path, img_name)), IMREAD_GRAYSCALE)
    images_ar.append(image)

# Converting images into an array. This is very important as model requires arrays not lists. 
images = np.array(images_ar)
print("Shape of images array before reshaping: ", np.shape(images))
images = images.reshape(np.shape(images)[0], np.shape(images)[1] * np.shape(images)[2])
print("Shape of images array after reshaping: ", np.shape(images))


print ("Splitting data into training sets")
X_train, X_test, y_train, y_test = train_test_split(images, gender_labels, test_size = 0.5, random_state=42)


print("Shape of X_train array after split: ", np.shape(X_train))
print("Shape of X_train array after split: ", np.shape(y_train))

model = SVC(kernel = "linear", C=10)
model.fit(X_train, y_train)

print("Training...")
random.shuffle(X_train)
label_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, label_pred) * 100
print("Accuracy:" ,accuracy)







#image read append to a list
#image np.array
#images reshape all list
#images_dataset = images.reshape()




#X = []
#y = []
#for features, label in training:
#    X.append(features)
#    y.append(label)
#X = np.array(X).reshape(-1, 218, 178, 3)
#X = X.astype('float32')
#X /= 255
#print(np.shape(X[1]))
#print(np.shape(y))
##Y = np_utils.to_categorical(y, 4)
#print(Y[100])
#print(np.shape(Y))

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
#MAYBE SHUFFLE
# random.shuffle(training)
#print(np.shape(X_test))

































# SPLITTING DATA
#X_train, X_test, y_train, y_test = train_test_split(dfs, gender_labels, test_size=0.1)
# Preprocess the data (scaling, imputation, etc.)
#model = RandomForestClassifier(n_estimators=10, max_depth=12)
#model.fit(X_train, y_train)
#ccuracy = model.score(X_test, y_test)
#print(accuracy)






#dfs = []

#for i, label in zip(image_names, gender_labels):
 #   path = os.path.join('Datasets/celeba_test/img/', i)
  #  image = cv2.imread(path)
    #image = image.flatten()
 #   image = image.reshape(image.shape[0], np.prod(image.shape[1:]))
 #   image = image.flatten()
 #   print(image.shape)
 #   df = pd.DataFrame({"image" : image, "label": label})
 #   dfs.append(df)
    #print(images.shape)
    
#dfs = dfs.drop(df.columns[0], axis=1)
#print (len(dfs))
#print(np.shape(dfs))
#print(dfs[0])
#print(dfs[23])
#print(gender_label)
#print(images) 
#df_images = pd.concat(images, axis =0)
#print(images.shape)