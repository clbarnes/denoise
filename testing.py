import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave
from scipy.ndimage import imread
from skimage import exposure, img_as_float, color, img_as_uint
import os
from PIL import Image


def imsave2(path, img):
    stacked = np.stack([img, img, img], axis=2)
    scaled = (((stacked - stacked.min()) / (stacked.max() - stacked.min())) * 255.9).astype(np.uint8)
    image = Image.fromarray(scaled)
    image.save(path)

imroot = 'images/testing/'

raw = imread(imroot + 'raw.png')[200:1400, :, :]
imsave2(imroot + 'raw2.png', raw)
print('raw: {}, {}'.format(raw.min(), raw.max()))
flattened = imread(imroot + 'raw.png', flatten=True)[200:1400, :]
imsave2(imroot + 'flattened.png', flattened)
unflattened = np.stack([flattened, flattened, flattened], axis=2)
imsave2(imroot + 'unflattened.png', unflattened)
flattened_re = imread(imroot + 'flattened.png')
print('flattened: {}, {}'.format(flattened.min(), flattened.max()))

greyed = color.rgb2gray(raw)

fig, ax_arr = plt.subplots(3, 1)

# axes = ax_arr.ravel()
#
# axes[0].imshow(raw, cmap=plt.cm.gray)
# axes[1].imshow(flattened, cmap=plt.cm.gray)
# axes[2].imshow(greyed, cmap=plt.cm.gray)
#
# plt.show()

# imsave(imroot + 'greyed_uint.png', greyed*256)

print('done')