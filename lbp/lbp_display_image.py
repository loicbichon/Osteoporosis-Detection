# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:58:44 2020

@author: Loïc Bichon
"""

"""
This code display an image before and after LBP
"""

from skimage.feature import local_binary_pattern
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class LocalBinaryPatterns:

    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return (hist, lbp)

filename = "./ROIsMATAIM/1/1_0585 ROI.tif"
image = Image.open(filename)

desc = LocalBinaryPatterns(8, 1)
hist, lbp = desc.describe(image)

fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
ax[0].axis('off')
ax[0].imshow(image, cmap="gray")
ax[0].set_title('Input image')
ax[0].set_adjustable('box')

ax[1].axis('off')
ax[1].imshow(lbp, cmap="gray")
ax[1].set_title('Histogram of Oriented Gradients')
ax[1].set_adjustable('box')
plt.show()











