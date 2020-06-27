# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 02:07:49 2020

@author: loicb
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog
from PIL import Image
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import os
import time

models = {
 	"knn": KNeighborsClassifier(n_neighbors=2),
 	"svm": SVC(C=10, kernel="rbf", gamma=0.01),
}

start_time = time.time()
param_grid = {'n_neighbors': np.arange(1,41)}
orientations = np.arange(1, 13)

score = []

imagePaths = paths.list_images("./ROIsMATAIM")
data = []
labels = []

for imagePath in imagePaths:
    print("[INFO] processing {}".format(imagePath))
    image = Image.open(imagePath)
    (H, hogImage) = hog(image, orientations=2, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), transform_sqrt=True,
                    block_norm="L1", visualize=True)
    data.append(H)

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

score.append([model_cv.best_params_, model_cv.best_score_])

print("Execution time : %0.3f seconds" % (time.time() - start_time))
print(score)


# Output :
# Execution time : 2064.663 seconds
# Best parameters : pixels=16, cells=1, orientations=2, block_norm="L1"
