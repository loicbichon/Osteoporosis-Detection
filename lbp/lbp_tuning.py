# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:53:42 2020

@author: loicb
"""

"""
This code measure and find the best parameter for LBP
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

models = {
	"knn": KNeighborsClassifier(n_neighbors=20),
	"svm": SVC(kernel="linear"),
}

start_time = time.time()
param_grid = {'n_neighbors': np.arange(1,41)}
n_points = np.arange(8, 40, 4)
radius = np.arange(1, 5)

score = []

for i in n_points:
    for j in radius:

        print("[INFO] extracting image features for lbp({0}, {1})".format(i,j))
        imagePaths = paths.list_images("./ROIsMATAIM")
        desc = LocalBinaryPatterns(i, j)
        data = []
        labels = []

        for imagePath in imagePaths:
            image = Image.open(imagePath)
            hist = desc.describe(image)
            data.append(hist)

            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        le = LabelEncoder()
        labels = le.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    stratify=labels,
                                                    random_state=42)
        model = models["knn"]
        model_cv = GridSearchCV(model, param_grid, n_jobs=-1, cv=5)
        model_cv.fit(X_train, y_train)
        y_pred = model_cv.predict(X_test)

        score.append([(i, j),model_cv.best_params_ , model_cv.best_score_])

score.sort(key = lambda x: x[2], reverse=True)
print("Execution time : %0.3f seconds" % (time.time() - start_time))
print(score)


# Output :
# Execution time : 784.864 seconds
# score = [[(8, 1), {'n_neighbors': 13}, 0.6923076923076923],
        # [(20, 1), {'n_neighbors': 11}, 0.6846153846153846],
        # [(32, 2), {'n_neighbors': 21}, 0.6846153846153846],
        # [(12, 1), {'n_neighbors': 8}, 0.676923076923077],
        # [(24, 1), {'n_neighbors': 11}, 0.676923076923077],
        # [(24, 2), {'n_neighbors': 23}, 0.676923076923077],
        # [(16, 1), {'n_neighbors': 17}, 0.6692307692307693],
        # [(36, 2), {'n_neighbors': 15}, 0.6692307692307693],
        # [(36, 3), {'n_neighbors': 18}, 0.6692307692307693],
        # [(28, 2), {'n_neighbors': 20}, 0.6615384615384616],
        # [(16, 2), {'n_neighbors': 37}, 0.6615384615384615],
        # [(28, 3), {'n_neighbors': 14}, 0.6615384615384615],
        # [(20, 3), {'n_neighbors': 13}, 0.6538461538461539],
        # [(36, 1), {'n_neighbors': 15}, 0.6538461538461539],
        # [(12, 2), {'n_neighbors': 9}, 0.6461538461538461],
        # [(20, 2), {'n_neighbors': 37}, 0.6461538461538461],
        # [(24, 3), {'n_neighbors': 13}, 0.6384615384615385],
        # [(28, 4), {'n_neighbors': 19}, 0.6384615384615385],
        # [(16, 3), {'n_neighbors': 16}, 0.6384615384615384],
        # [(32, 1), {'n_neighbors': 3}, 0.6384615384615384],
        # [(36, 4), {'n_neighbors': 30}, 0.6384615384615384],
        # [(12, 3), {'n_neighbors': 35}, 0.6307692307692307],
        # [(24, 4), {'n_neighbors': 29}, 0.6307692307692307],
        # [(28, 1), {'n_neighbors': 9}, 0.6307692307692307],
        # [(32, 3), {'n_neighbors': 19}, 0.6307692307692307],
        # [(8, 2), {'n_neighbors': 7}, 0.6230769230769232],
        # [(12, 4), {'n_neighbors': 1}, 0.6230769230769231],
        # [(20, 4), {'n_neighbors': 15}, 0.6230769230769231],
        # [(32, 4), {'n_neighbors': 19}, 0.6230769230769231],
        # [(8, 3), {'n_neighbors': 27}, 0.6076923076923078],
        # [(8, 4), {'n_neighbors': 13}, 0.6076923076923076],
        # [(16, 4), {'n_neighbors': 23}, 0.6]]