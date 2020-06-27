# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:31:19 2020

@author: loicb
"""

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from PIL import Image
from imutils import paths
import os
import time

start_time = time.time()
print("[INFO] extracting image features...")
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

param_grid = [{'kernel': ['linear'],
                'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
              {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
model = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
model.fit(data, labels)

print("Best parameter combinaison : {}".format(model.best_params_))
print("Best score : %0.3f" % (model.best_score_))


# Output :
# Best parameter combinaison : {'C': 25, 'gamma': 0.001, 'kernel': 'rbf'}
# Best score : 0.500
