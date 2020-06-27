# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 22:20:13 2020

@author: Loïc Bichon
"""

"""
This code measure and find the best parameter for KNN
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
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

param_grid = [{'n_neighbors': np.arange(1,51)}]
model = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
model.fit(data, labels)

print("Best parameter combinaison : {}".format(model.best_params_))
print("Best score : %0.3f" % (model.best_score_))


# Output :
# Best parameter combinaison : {'n_neighbors': 18}
# Best score : 0.678
