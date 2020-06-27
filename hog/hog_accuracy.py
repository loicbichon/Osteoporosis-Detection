# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:45:27 2020

@author: loicb
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
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

models = {
 	"knn": KNeighborsClassifier(n_neighbors=2),
 	"svm": SVC(C=25, kernel="rbf", gamma=0.001),
}

model = models['svm']
cv_results = cross_val_score(model, data, labels, cv=5)
print("Execution time: %0.2f" % (time.time() - start_time))
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))

















