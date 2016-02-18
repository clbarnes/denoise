import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import imread
from skimage import exposure, img_as_float, filters
from PIL import Image

nonoise_arr = imread('images/no_noise.png', flatten=True).astype(np.uint16)
noise_arr = imread('images/noise.png', flatten=True).astype(np.uint16)

def imsave2(path, img):
    stacked = np.stack([img, img, img], axis=2)
    scaled = (((stacked - stacked.min()) / (stacked.max() - stacked.min())) * 255.9).astype(np.uint8)
    image = Image.fromarray(scaled)
    image.save(path)


def plot_img_and_hist(img, axes, title='', bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    # img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    # ax_img.set_adjustable('box-forced')
    ax_img.set_title(title)

    # Display histogram
    ax_hist.hist(img.ravel()/img.max(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_ylabel('Number of pixels')

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

p2, p10, p98 = np.percentile(noise_arr, (2, 10, 98))
img_rescale = exposure.rescale_intensity(noise_arr, in_range=(p2, p98))
imsave2('images/rescale.png', img_rescale)
img_rescale_thresh = exposure.rescale_intensity(noise_arr, in_range=(p10, noise_arr.max()))
imsave2('images/rescale_threshold.png', img_rescale_thresh)
img_eq = exposure.equalize_hist(noise_arr)
imsave2('images/equalised.png', img_eq)
img_adapteq = exposure.equalize_adapthist(noise_arr, clip_limit=0.03)
imsave2('images/adapt_equalised.png', img_adapteq)

to_plot = [noise_arr, nonoise_arr]
fig, ax_arr = plt.subplots(2, len(to_plot))
plt.tight_layout()

for i, arr in enumerate(to_plot):
    ax_img, ax_hist, ax_cdf = plot_img_and_hist(arr, ax_arr.T[i, :])

plt.show()

