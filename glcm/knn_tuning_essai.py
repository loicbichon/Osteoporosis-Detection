# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:13:38 2020

@author: loicb
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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

neighbors = np.arange(1, 41)
sample = 700
average_scores_mean = np.empty(len(neighbors))
average_scores_std = np.empty(len(neighbors))

for i, k in enumerate(neighbors):

    scores = np.empty(sample)
    for j in range(0, sample):
        X_train, X_test, y_train, y_test = train_test_split(data, labels)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        scores[j] = accuracy_score(model.predict(X_test), y_test)

    average_scores_mean[i] = np.mean(scores)
    average_scores_std[i] = np.std(scores)

plt.title('k-NN: Average accuracy per neighbors, Samples : {}'.format(sample))
plt.plot(neighbors, average_scores_mean, label = 'Testing Accuracy')
plt.fill_between(neighbors, average_scores_mean + average_scores_std,
                 average_scores_mean - average_scores_std, alpha=0.2)
plt.axhline(np.max(average_scores_mean), linestyle='--', color='r')
plt.axhline(np.max(average_scores_mean) + np.max(average_scores_std[np.argmax(average_scores_mean)]),
            linestyle='--', color='.5')
plt.axhline(np.max(average_scores_mean) - np.max(average_scores_std[np.argmax(average_scores_mean)]),
            linestyle='--', color='.5')
plt.xlabel('Number of Neighbors')
plt.ylabel('Average accuracy +/- Std')
plt.xlim([neighbors[0], neighbors[-1]])
plt.show()

print("Execution time : %s seconds" % (time.time() - start_time))
print("Best number of beighbors : {}".format(np.argmax(average_scores_mean) + 1))
print("Mean : {} %".format(np.max(average_scores_mean) * 100))
print("Std : {} %".format(average_scores_std[np.argmax(average_scores_mean)] * 100))
print("Sample number : {}".format(sample))


# Ouput :
# Execution time : 148.41669631004333 seconds
# Best number of beighbors : 14
# Mean : 66.26623376623377 %
# Std : 6.544684684677991 %
# Sample number : 700
