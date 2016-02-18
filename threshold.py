import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import imread
from skimage import exposure, img_as_float, filter
from PIL import Image

nonoise_arr = imread('images/no_noise.png', flatten=True)
noise_arr = imread('images/noise.png', flatten=True)

def imsave2(path, img):
    stacked = np.stack([img, img, img], axis=2)
    scaled = (((stacked - stacked.min()) / (stacked.max() - stacked.min())) * 255.9).astype(np.uint8)
    image = Image.fromarray(scaled)
    image.save(path)

# val = filter.threshold_otsu(noise_arr)
val2 = 0.06*259.9

vals = np.percentile(noise_arr, (1, 2.5, 5, 7.5, 10, 15, 20))
print(vals)


for val in vals:
    thresholded = noise_arr.copy()
    thresholded[noise_arr < val] = np.mean(noise_arr)
    plt.figure()
    plt.title('Threshold value = {}'.format(val))
    plt.imshow(thresholded, cmap=plt.cm.gray)
    plt.tight_layout()

plt.show()