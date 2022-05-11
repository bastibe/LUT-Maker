import math
import pathlib
import numpy
import scipy.signal
import numba
from PIL import Image
from tqdm import tqdm


MIN_COLOR_SAMPLES = 5  # discard LUT pixels if they have less than this many color samples
LUT_CUBE_SIZE = 64  # LUT cube size, this many steps per color
LUT_IMAGE_SIZE = 512  # corresponding LUT image size
WEIGHT_FACTOR = 2  # factor to weigh sampled colors higher than neutral colors
BOUNDARY_WEIGHT_FACTOR = 5  # factor to weigh boundary colors higher than neutral colors
SMOOTHING_SIGMA = 4  # standard deviation of smoothing kernel
                     # kernel is a 6*sigma gaussian cube
SUBSAMPLING = 5  # downscale images by this factor

RGB2IDX = int(256 / LUT_CUBE_SIZE)  # conversion factor from RGB levels to cube coordinates
assert LUT_CUBE_SIZE**3 == LUT_IMAGE_SIZE**2, "LUT configuration invalid"


def main(source_path, target_path, lut_name):
    """Read all images in source/target path, smooth and extrapolate, then create LUT"""

    color_sum = numpy.zeros([LUT_CUBE_SIZE, LUT_CUBE_SIZE, LUT_CUBE_SIZE, 3], dtype='uint64')
    color_count = numpy.zeros([LUT_CUBE_SIZE, LUT_CUBE_SIZE, LUT_CUBE_SIZE], dtype='uint64')

    for source_img, target_img in tqdm(list(same_images(source_path, target_path)),
                                       desc='loading images'):
        try:
            load_and_map_images(source_img, target_img, color_sum, color_count)
        except TypeError as err:
            print(err)  # skip image if there was an error

    lut_matrix = generate_lut(color_sum, color_count)
    lut_matrix = smooth_and_extrapolate_lut(lut_matrix, color_count, SMOOTHING_SIGMA)
    lut_image = lut_matrix.swapaxes(0, 2).reshape([LUT_IMAGE_SIZE, LUT_IMAGE_SIZE, 3])
    Image.fromarray(lut_image).save(lut_name, 'PNG')


def same_images(source_path, target_path):
    """Iterator that walks source and target paths, looking for identical images"""

    for source_img in source_path.glob('*.jpg'):
        target_img = target_path / source_img.name
        if target_img.exists():
            yield source_img, target_img
        else:
            print(f"No matching image found for {source_img.name}")


def load_and_map_images(source_file, target_file, color_sum, color_count):
    """Load all pixels from source/target file into LUT matrices

    Target images are rotated as necessary.
    All images and downsampled to hide sharpening/artifacts.
    """

    source_image = Image.open(source_file)
    target_image = Image.open(target_file)

    # read orientation tag (EXIF 274):
    orientation = target_image.getexif()[274]
    if orientation == 1:
        pass  # right side up
    elif orientation == 8:
        target_image = target_image.transpose(2)  # rotate 90
    elif orientation == 3:
        target_image = target_image.transpose(3)  # rotate 180
    elif orientation == 6:
        target_image = target_image.transpose(4)  # rotate 270

    # crop away outer pixels if one of the images is larger:
    source_crop = [0, 0, source_image.width, source_image.height]
    target_crop = [0, 0, target_image.width, target_image.height]
    if source_image.width > target_image.width:
        source_crop[0] = (source_image.width - target_image.width)//2
        source_crop[2] = target_image.width
    elif source_image.width < target_image.width:
        target_crop[0] = (target_image.width - source_image.width)//2
        target_crop[2] = source_image.width
    if source_image.height > target_image.height:
        source_crop[1] = (source_image.height - target_image.height)//2
        source_crop[3] = target_image.height
    elif source_image.height < target_image.height:
        target_crop[1] = (target_image.height - source_image.height)//2
        target_crop[3] = source_image.height

    # resize to hide sharpening etc.
    source_image = source_image.resize([source_crop[2] // SUBSAMPLING, source_crop[3] // SUBSAMPLING],
                                       resample=Image.Resampling.LANCZOS, box=source_crop)
    target_image = target_image.resize([target_crop[2] // SUBSAMPLING, target_crop[3] // SUBSAMPLING],
                                       resample=Image.Resampling.LANCZOS, box=target_crop)

    source = numpy.asarray(source_image)
    target = numpy.asarray(target_image)

    if source.shape != target.shape:
        raise TypeError(f'Shape different in {source_file.name} ({source.shape}) and '
                        f'{target_file.name} ({target.shape}): skipping image')

    count_pixels(source, target, color_sum, color_count)


@numba.jit(nopython=True)
def count_pixels(source, target, color_sum, color_count):
    """Iterate through pixels in source/target, add their values to color_sum and color_count

    color_sum holds the sum of rgb values for each LUT bin in all images
    color_count holds the number of samples for each bin

    Does not count black pixels, as they are most likely artifacts.
    """
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            ridx, gidx, bidx = source[x, y] // RGB2IDX
            color_sum[ridx, gidx, bidx] += target[x, y]
            if not (ridx == gidx == bidx == 0):  # don't count black pixels; they are probably artifacts
                color_count[ridx, gidx, bidx] += 1


def generate_lut(color_sum, color_count):
    """Average out samples from color_sum, but reject obvious artifacts

    artifacts are replaced by identity colors
    """

    lut_matrix = numpy.zeros(color_sum.shape, dtype='uint8')
    for ridx in range(lut_matrix.shape[0]):
        for gidx in range(lut_matrix.shape[1]):
            for bidx in range(lut_matrix.shape[2]):
                sample_count = color_count[ridx, gidx, bidx]
                identity_color = [ridx * RGB2IDX, gidx * RGB2IDX, bidx * RGB2IDX]
                mean_color = (color_sum[ridx, gidx, bidx] / sample_count).clip(0, 255) \
                    if sample_count > 0 else [0, 0, 0]
                # artifact conditions:
                # - not enough samples
                # - color obviously wrong
                if sample_count < MIN_COLOR_SAMPLES or sum(mean_color) < sum(identity_color)/2:
                    lut_matrix[ridx, gidx, bidx] = identity_color
                    # and mark color as interpolated for later smoothing step:
                    color_count[ridx, gidx, bidx] = 0
                else:
                    lut_matrix[ridx, gidx, bidx] = mean_color
    return lut_matrix


def smooth_and_extrapolate_lut(lut_matrix, sample_count, sigma):
    """Smooth lut_matrix with a gaussian kernel of standard deviation sigma

    Smoothing is not just a convolution with the kernel, but weighs
    sampled LUT pixels higher than guessed remaining pixels.
    Also does not sample beyond lut boundaries.

    """
    out_matrix = lut_matrix.copy()
    kernel = gaussian_kernel(sigma)
    for ridx in tqdm(range(LUT_CUBE_SIZE), desc='processing LUT'):
        for gidx in range(LUT_CUBE_SIZE):
            for bidx in range(LUT_CUBE_SIZE):
                out_matrix[ridx, gidx, bidx] = \
                    smooth_extrapolate_color(lut_matrix, sample_count, sigma, (ridx, gidx, bidx), kernel)
    return out_matrix


def gaussian_kernel(sigma):
    """Create a gaussian kernel of size 3*sigma cubed, with standard deviation sigma"""
    radius = sigma * 3
    kernel = numpy.zeros((2*radius+1, 2*radius+1, 2*radius+1))
    for rdelta in range(-radius, radius+1):
        for gdelta in range(-radius, radius+1):
            for bdelta in range(-radius, radius+1):
                distance = ( rdelta**2 + gdelta**2 + bdelta**2 )**0.5
                kernel[rdelta+radius, gdelta+radius, bdelta+radius] = \
                    gaussian(distance, sigma_squared=sigma**2)
    return kernel


@numba.jit(nopython=True)
def gaussian(x, mu=0, sigma_squared=1):
    return 1.0 / (sigma_squared**0.5 * (2*math.pi)**0.5) * math.exp( -(x - mu)**2 / (2 * sigma_squared) )


@numba.jit(nopython=True)
def smooth_extrapolate_color(lut_matrix, count_matrix, sigma, coordinate, kernel):
    """Calculate the smoothed color of the pixel at coordinate in lut_matrix

    Weighs pixels with count > MIN_COLOR_SAMPLES more highly than
    remaining interpolated pixels.

    Near LUT boundaries, only samples a symmetric area around
    coordinate, so as to preserve boundary values.

    """
    rstart, gstart, bstart = coordinate
    kernel_radius = sigma * 3  # kernel is a 2*radius+1 cube

    sum_color = numpy.zeros(3)
    sum_weights = 0
    # calculate radii around coordinate that fit within the LUT. Radii
    # are kernel_radius by default, but get smaller near boundaries.
    # If this were not done, smoothing volumes would become asymmetric
    # at boundaries, and limit extreme values.
    rradius = min(kernel_radius, rstart, LUT_CUBE_SIZE-rstart-1)
    gradius = min(kernel_radius, gstart, LUT_CUBE_SIZE-gstart-1)
    bradius = min(kernel_radius, bstart, LUT_CUBE_SIZE-bstart-1)
    # iterate over all indices in a box around coordinate:
    for rdelta in range(-rradius, rradius+1):
        for gdelta in range(-gradius, gradius+1):
            for bdelta in range(-bradius, bradius+1):
                weight = kernel[kernel_radius+rdelta, kernel_radius+gdelta, kernel_radius+bdelta]
                sample_count = count_matrix[rstart+rdelta, gstart+gdelta, bstart+bdelta]
                if rstart+rdelta == 0 or gstart+gdelta == 0 or bstart+bdelta == 0 or \
                   rstart+rdelta == LUT_CUBE_SIZE-1 or gstart+gdelta == LUT_CUBE_SIZE-1 or \
                   bstart+bdelta == LUT_CUBE_SIZE-1:
                    weight *= BOUNDARY_WEIGHT_FACTOR
                elif sample_count > MIN_COLOR_SAMPLES:
                    weight *= WEIGHT_FACTOR
                lut_color = lut_matrix[rstart+rdelta, gstart+gdelta, bstart+bdelta]
                sum_color = sum_color + weight * lut_color
                sum_weights += weight
    r, g, b = sum_color / sum_weights
    return numpy.array([r, g, b], dtype='uint8')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Make a LUT from image pairs')
    parser.add_argument('source_dir', type=str, help='directory containing unprocessed images')
    parser.add_argument('target_dir', type=str, help='directory containing processed images')
    parser.add_argument('lut_name', type=str, help='name of the resulting LUT')
    args = parser.parse_args()
    main(pathlib.Path(args.source_dir), pathlib.Path(args.target_dir), args.lut_name)
