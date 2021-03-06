# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:52:19 2020

@author: Gaetan
"""
import time 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import os
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from resizeimage import resizeimage
import pandas as pd


start_time = time.time()

"""
GLCM
"""

df = pd.DataFrame (columns = ['osteoporosis','dissimilarity', 'correlation', 'homogeneity', 'contrast', 'energy']);

counter = 0;

#Loop for image with osteoporosis (Y)
for img in glob.glob(os.path.join("Y_JPG/*")) :
    
    #Load original image
    image = Image.open(img);  
    
    #Resize image to fit with GLCM
    img_res = resizeimage.resize_cover(image, [256, 256]);
    
    image_arr = np.array(img_res);

    #Compute GLCM matrix
    glcm_matrix = greycomatrix(image_arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True);
    
    counter = counter + 1;
    
    #Compute GLCM features
    dissimilarity = greycoprops(glcm_matrix, 'dissimilarity')[0, 0];
    correlation = greycoprops(glcm_matrix, 'correlation')[0, 0];
    homogeneity = greycoprops(glcm_matrix, 'homogeneity')[0, 0];
    contrast = greycoprops(glcm_matrix, 'contrast')[0, 0];
    energy = greycoprops(glcm_matrix, 'energy')[0, 0];
    
    
    #Add GLCM features to a row in the df
    df = df.append(pd.Series([int(0), dissimilarity, correlation, homogeneity, contrast, energy], index=df.columns), ignore_index=True);
    


#Loop for NO osteoporosis images
for img in glob.glob(os.path.join("N_JPG/*")) :
    
    #Load original image
    image = Image.open(img);  
    
    #Resize image to fit with GLCM
    img_res = resizeimage.resize_cover(image, [256, 256]);
    
    image_arr = np.array(img_res);

    #Compute GLCM matrix
    glcm_matrix = greycomatrix(image_arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True);
    
    counter = counter + 1;
    
    #Compute GLCM features
    dissimilarity = greycoprops(glcm_matrix, 'dissimilarity')[0, 0];
    correlation = greycoprops(glcm_matrix, 'correlation')[0, 0];
    homogeneity = greycoprops(glcm_matrix, 'homogeneity')[0, 0];
    contrast = greycoprops(glcm_matrix, 'contrast')[0, 0];
    energy = greycoprops(glcm_matrix, 'energy')[0, 0];
    
    
    #Add GLCM features to a row in the df
    df = df.append(pd.Series([int(1), dissimilarity, correlation, homogeneity, contrast, energy], index=df.columns), ignore_index=True);
    
    
#print(counter);
    
"""

KNN Tuning

"""
#Create arrays for features and target variable
y = df['osteoporosis']
X = df.drop('osteoporosis', axis = 1)

#train test split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 


param_grid = [{'n_neighbors': np.arange(1,51)}]

#GridSearchCV to find n_neighbors most efficient value
knn_model = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1) #KNN other parameters algorithm='brute', metric='euclidean'
knn_model.fit(X_train, y_train)






"""
Results
"""


print("Best parameter combinaison : {}".format(knn_model.best_params_))
print("Best score : %0.3f" % (knn_model.best_score_))

print("Execution time : %0.3f seconds" % (time.time() - start_time))


"""
Best parameter combinaison : {'n_neighbors': 3}
Best score : 0.503
Execution time : 2.393 seconds
"""