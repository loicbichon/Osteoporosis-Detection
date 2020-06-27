# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 01:52:15 2020

@author: loicb
"""

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from skimage import feature
from PIL import Image
from imutils import paths
from operator import itemgetter
import numpy as np
import os
import time

class LocalBinaryPatterns:

	def __init__(self, numPoints, radius):
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		lbp = feature.local_binary_pattern(image, self.numPoints, self.radius,
                                     method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
                           bins=np.arange(0, self.numPoints + 3),
                           range=(0, self.numPoints + 2))
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		return hist

start_time = time.time()
print("[INFO] extracting image features...")
imagePaths = paths.list_images("./ROIsMATAIM")
desc = LocalBinaryPatterns(16, 3)
data = []
labels = []

for imagePath in imagePaths:
    print("[INFO] processing {}".format(imagePath))
    image = Image.open(imagePath)
    hist = desc.describe(image)
    data.append(hist)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)

param_grid = [{'kernel': ['linear'],
                'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
              {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
param_length = len(param_grid[0]['C']) + len(param_grid[1]['C']) * len(param_grid[1]['gamma'])

scores_mean = np.zeros(param_length)
scores_std = np.zeros(param_length)
sample = 500

for i in range(0, sample):

    X_train, X_test, y_train, y_test = train_test_split(data, labels)
    svm = GridSearchCV(SVC(), param_grid, n_jobs=-1)
    svm.fit(X_train, y_train)

    means = svm.cv_results_['mean_test_score']
    stds = svm.cv_results_['std_test_score']

    scores_mean += means
    scores_std += stds

scores_mean /= sample
scores_std /= sample
average_scores_mean = list(zip(svm.cv_results_['params'], scores_mean))
average_scores_std = list(zip(svm.cv_results_['params'], scores_std))
average_scores_mean.sort(key = lambda x: x[1])
average_scores_std.sort(key = lambda x: x[1])

print("Execution time : %s seconds" % (time.time() - start_time))
print("Best parameter combinaison : {}".format(max(average_scores_mean, key=itemgetter(1))[0]))
print("Mean : {} %".format(
    max(average_scores_mean, key=itemgetter(1))[1] * 100))
print("Std : {} %".format(average_scores_std[list(
    map(itemgetter(0), average_scores_mean)).index(
        max(average_scores_mean, key=itemgetter(1))[0])][1] * 100))
print("Sample number : {}".format(sample))


# Ouput :
# Execution time : 254.86485195159912 seconds
# Best parameter combinaison : {'C': 100, 'kernel': 'linear'}
# Mean : 63.57999999999995 %
# Std : 7.9952010953923 %
# Sample number : 500