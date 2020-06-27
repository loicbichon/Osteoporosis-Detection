# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:25:53 2020

@author: loicb
"""

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from PIL import Image

filename = "./ROIsMATAIM/0/1_0597 ROI.tif"
image = Image.open(filename)

(H, hogImage) = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), transform_sqrt=True,
                    block_norm="L1", visualize=True)

fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
ax[0].axis('off')
ax[0].imshow(image, cmap="gray")
ax[0].set_title('Input image')
ax[0].set_adjustable('box')

# Rescale histogram for better display
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

ax[1].axis('off')
ax[1].imshow(hogImage, cmap="gray")
ax[1].set_title('Histogram of Oriented Gradients')
ax[1].set_adjustable('box')
plt.show()