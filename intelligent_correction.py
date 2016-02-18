import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import imread
from skimage import exposure, img_as_float
from PIL import Image
from scipy import stats, optimize


DEFAULT_NOISE_DIST = lambda b: stats.beta(1, b)
DEFAULT_REAL_DIST = stats.norm


def imsave2(path, img):
    """
    Save greyscale array without reducing intensity.

    :param path: Path at which to save the image
    :param img: Greyscale image array
    :return: None
    """
    stacked = np.stack([img, img, img], axis=2)
    scaled = (((stacked - stacked.min()) / (stacked.max() - stacked.min())) * 255.9).astype(np.uint8)
    image = Image.fromarray(scaled)
    image.save(path)


def noise_pdf(data, *params, noise_dist=DEFAULT_NOISE_DIST):
    """
    Probability density function of pixels being 'black noise' artifacts based on their intensity value.

    :param data: Array of pixel intensities
    :param params: noise_shape, real_loc, real_scale, noise_prop
    :param noise_dist: Underlying distribution constructor reference
    :return: Probabilities of pixels being black noise
    """
    noise_shape, real_loc, real_scale, noise_prop = params

    #bounds
    if any([noise_shape < 0, noise_prop < 0, noise_prop > 1]):
        return np.inf

    p_noise = noise_dist(noise_shape).pdf(data/256) * noise_prop  # todo: remove /256 if not using beta
    # p_noise = noise_dist(noise_shape).pdf(data) * noise_prop

    return p_noise


def real_pdf(data, *params, real_dist=DEFAULT_REAL_DIST):
    """
    Probability density function of pixels being actual data based on their intensity value.

    :param data: Array of pixel intensities
    :param params: noise_shape, real_loc, real_scale, noise_prop
    :param real_dist: Underlying distribution constructor reference
    :return: Probabilities of pixels being real data
    """
    noise_shape, real_loc, real_scale, noise_prop = params

    if any([real_loc < 0, real_scale < 0]):
        return np.inf

    if isinstance(real_dist, stats.rv_continuous):
        return real_dist(real_loc, real_scale).pdf(data) * (1 - noise_prop)
    else:
        return real_dist(real_loc, real_scale).pmf(data) * (1 - noise_prop)


def summed_pdf(data, *params, noise_dist=DEFAULT_NOISE_DIST, real_dist=DEFAULT_REAL_DIST):
    """
    Probability density function of pixels being from the data + the weighted sum of real and noise PDFs.

    :param data: Array of pixel intensities
    :param params: noise_shape, real_loc, real_scale, noise_prop
    :param noise_dist: Underlying distribution constructor reference
    :param real_dist: Underlying distribution constructor reference
    :return: Probabilities of pixels being data
    """
    return noise_pdf(data, *params, noise_dist=noise_dist) + real_pdf(data, *params, real_dist=real_dist)


def find_switch(params, noise_certainty=0.5, noise_dist=DEFAULT_NOISE_DIST, real_dist=DEFAULT_REAL_DIST):
    """
    Find the x value at which pixels are more likely to be black noise than real data

    :param params: noise_shape, real_loc, real_scale, noise_prop
    :param noise_certainty: Required confidence that a pixel is noise (e.g. 0.9 finds the x value where there is a 90%
    certainty that the pixel is black noise)
    :param noise_dist: Underlying distribution constructor reference
    :param real_dist: Underlying distribution constructor reference
    :return: Switch point
    """
    diff = noise_pdf(np.arange(256), *list(params), noise_dist=noise_dist) * (1-noise_certainty) \
           - real_pdf(np.arange(256), *list(params), real_dist=real_dist) * noise_certainty
    try:
        return np.argwhere(diff > 0).max()
    except ValueError:
        return 0


def cost(params, x, y):
    """
    Mean squared error of model given by params compared to real x, y data

    :param params: noise_shape, real_loc, real_scale, noise_prop
    :param x: Pixel intensity values
    :param y: Proportion of pixels at this value
    :return: MSE
    """
    model_y = summed_pdf(x, *list(params))
    return np.mean(np.square(model_y - y))


def collapse_data(x, bins=256):
    """
    Get normalised histogram of all values in array

    :param x: array of pixel intensity values
    :param bins: number of bins in histogram
    :return: x, y values of histogram
    """
    h, edges = np.histogram(x, bins=bins)
    mids = np.stack((edges[:-1], edges[1:]), axis=0).mean(axis=0)

    return mids, h/h.sum()


def plot_fit(img, ax, label='', c='r'):
    """


    :param img: 2D array of pixel intensities making up an image
    :param ax: axis object to which lines should be added
    :param label: label to apply to all lines associated with this image
    :param c: colour
    :return: optimal PDF fitting parameters in this image
    """
    x, y = collapse_data(img.ravel())
    # optim_params = optimize.minimize(cost, p0, args=(x, y), bounds=[(1, None), (0, None), (0, None), (0, 1)]).x
    optim_params, _ = optimize.curve_fit(summed_pdf, x, y, p0)

    ax.plot(x, y, c=c, label=label + ' real')
    ax.plot(x, summed_pdf(x, *optim_params), c=c, linestyle='--', label=label + ' fitted')
    # ax.plot(x, summed_cdf(x, *p0), c='k', label='starting_params')

    noise_certainty = 0.9
    switch_point = find_switch(optim_params, noise_certainty=noise_certainty)
    ax.axvline(switch_point, color=c, label=label + ' switch point ($p={}$)'.format(noise_certainty))

    ax.plot(x, real_pdf(x, *optim_params), c=c, linestyle=':', label=label + ' denoised')

    return optim_params


if __name__ == '__main__':
    nonoise_arr = imread('images/no_noise.png', flatten=True).astype(np.uint16)
    noise_arr = imread('images/noise.png', flatten=True).astype(np.uint16)

    p0 = (10, 127, 44, 0.1)

    fig, ax = plt.subplots()

    noise_params = plot_fit(noise_arr, ax, label='noise', c='r')
    nonoise_params = plot_fit(nonoise_arr, ax, label='no noise', c='b')

    plt.legend()

    plt.show()
    print('done')
