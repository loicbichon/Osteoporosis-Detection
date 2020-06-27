# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 21:06:01 2020

@author: loicb
"""

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from skimage import feature
from PIL import Image
from imutils import paths
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
desc = LocalBinaryPatterns(8, 1)
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
model = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
model.fit(data, labels)

print("Best parameter combinaison : {}".format(model.best_params_))
print("Best score : %0.3f" % (model.best_score_))


# Ouput :
# Best parameter combinaison : {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
# Best score : 0.644
