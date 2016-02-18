# height 200, 1400
# width 0, -1

import numpy as np
from PIL import Image
from scipy.ndimage import imread
import os
from matplotlib import pyplot as plt
# from skimage import exposure, img_as_float

image_root = '/home/cbarnes/work/denoise/images/'
screenshot_root = os.path.join(image_root, 'screenshot')
filenames = ['noise.png', 'no_noise.png']

def imsave2(path, img):
    stacked = np.stack([img, img, img], axis=2)
    scaled = (((stacked - stacked.min()) / (stacked.max() - stacked.min())) * 255.9).astype(np.uint8)
    image = Image.fromarray(scaled)
    image.save(path)

for filename in filenames:
    whole = imread(os.path.join(screenshot_root, filename)).sum(axis=2)
    arr = whole[200:1400, :]
    imsave2(os.path.join(image_root, filename[:-4] + '.png'), arr)

    plt.figure()
    plt.imshow(arr, cmap=plt.cm.gray)
    plt.colorbar()
    plt.title(filename)

plt.show()