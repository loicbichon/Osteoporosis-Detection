# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:35:20 2020

@author: loicb
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
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

models = {
	"knn": KNeighborsClassifier(n_neighbors=18),
	"svm": SVC(C=1000, kernel='rbf', gamma=0.01),
}

model = models['knn']
cv_results = cross_val_score(model, data, labels, cv=5)
print("Execution time: %0.2f" % (time.time() - start_time))
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))















