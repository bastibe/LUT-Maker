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

    lut_matrix = numpy.zeros(color_sum.shape, dtype='uint8')
    for ri in range(lut_matrix.shape[0]):
        for gi in range(lut_matrix.shape[1]):
            for bi in range(lut_matrix.shape[2]):
                num = color_count[ri, gi, bi]
                if num > MIN_COLOR_SAMPLES:
                    mean_color = (color_sum[ri, gi, bi] / num).clip(0, 255)
                    if not all(mean_color == 0):
                        # all-black is probably an artifact
                        lut_matrix[ri, gi, bi] = mean_color
                    else:
                        lut_matrix[ri, gi, bi] = [ri * RGB2IDX, gi * RGB2IDX, bi * RGB2IDX]
                else:
                    lut_matrix[ri, gi, bi] = [ri * RGB2IDX, gi * RGB2IDX, bi * RGB2IDX]

    lut_matrix = smooth_and_extrapolate(lut_matrix, color_count, SMOOTHING_SIGMA)
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
            ri, gi, bi = source[x, y] // RGB2IDX
            color_sum[ri, gi, bi] += target[x, y]
            if not (ri == gi == bi == 0):  # don't count black pixels; they are probably artifacts
                color_count[ri, gi, bi] += 1


def smooth_and_extrapolate(lut_matrix, sample_count, sigma):
    """Smooth lut_matrix with a gaussian kernel of standard deviation sigma

    Smoothing is not just a convolution with the kernel, but weighs
    sampled LUT pixels higher than guessed remaining pixels.
    Also does not sample beyond lut boundaries.

    """
    kernel = gaussian_kernel(sigma)
    for ri in tqdm(range(lut_matrix.shape[0]), desc='fixing LUT'):
        for gi in range(lut_matrix.shape[1]):
            for bi in range(lut_matrix.shape[2]):
                lut_matrix[ri, gi, bi] = smooth_extrapolate_color(lut_matrix, sample_count, sigma,
                                                                  (ri, gi, bi), kernel)
    return lut_matrix


def gaussian_kernel(sigma):
    """Create a gaussian kernel of size 3*sigma cubed, with standard deviation sigma"""
    radius = sigma * 3
    kernel = numpy.zeros((2*radius+1, 2*radius+1, 2*radius+1))
    for dr in range(-radius, radius+1):
        for dg in range(-radius, radius+1):
            for db in range(-radius, radius+1):
                distance = ( dr**2 + dg**2 + db**2 )**0.5
                kernel[dr+radius, dg+radius, db+radius] = gaussian(distance, sigma_squared=sigma**2)
    return kernel


@numba.jit(nopython=True)
def gaussian(x, mu=0, sigma_squared=1):
    return 1.0 / (sigma_squared**0.5 * (2*math.pi)**0.5) * math.exp( -(x - mu)**2 / (2 * sigma_squared) )


@numba.jit(nopython=True)
def smooth_extrapolate_color(lut_matrix, count_matrix, sigma, coordinate, kernel):
    """Calculate the smoothed color of the pixel at coordinate in lut_matrix

    weighs pixels with count > MIN_COLOR_SAMPLES more highly than
    remaining interpolated pixels.

    """
    ri, gi, bi = coordinate
    radius = sigma * 3

    sum_color = numpy.zeros(3)
    sum_weights = 0
    for dr in range(max(0, ri-radius), min(ri+radius+1, LUT_CUBE_SIZE)):
        for dg in range(max(0, gi-radius), min(gi+radius+1, LUT_CUBE_SIZE)):
            for db in range(max(0, bi-radius), min(bi+radius+1, LUT_CUBE_SIZE)):
                weight = kernel[dr-ri+radius, dg-gi+radius, db-bi+radius]
                if count_matrix[dr, dg, db] > MIN_COLOR_SAMPLES:
                    weight *= WEIGHT_FACTOR
                sum_color = sum_color + weight * lut_matrix[dr, dg, db]
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
