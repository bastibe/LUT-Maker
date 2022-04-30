import math
import pathlib
import numpy
import scipy.signal
import numba
from PIL import Image
from tqdm import tqdm


MIN_COLOR_SAMPLES = 5
LUT_CUBE_SIZE = 64
LUT_IMAGE_SIZE = 512
RGB2IDX = int(256 / LUT_CUBE_SIZE)
assert LUT_CUBE_SIZE**3 == LUT_IMAGE_SIZE**2, "LUT configuration invalid"
WEIGHT_FACTOR = 2  # factor to weigh sampled colors higher than neutral colors
SMOOTHING_SIGMA = 4  # standard deviation of smoothing kernel
                     # kernel is a 6*sigma gaussian cube


def main(source_path, target_path, lut_name):
    color_sum = numpy.zeros([LUT_CUBE_SIZE, LUT_CUBE_SIZE, LUT_CUBE_SIZE, 3], dtype='uint64')
    color_count = numpy.zeros([LUT_CUBE_SIZE, LUT_CUBE_SIZE, LUT_CUBE_SIZE], dtype='uint64')

    for source_img, target_img in tqdm(list(same_images(source_path, target_path)),
                                       desc='loading images'):
        try:
            load_and_map_images(source_img, target_img, color_sum, color_count)
        except TypeError as err:
            print(err)

    lut_matrix = numpy.zeros(color_sum.shape, dtype='uint8')
    for ri in range(lut_matrix.shape[0]):
        for gi in range(lut_matrix.shape[1]):
            for bi in range(lut_matrix.shape[2]):
                num = color_count[ri, gi, bi]
                if num > MIN_COLOR_SAMPLES:
                    lut_matrix[ri, gi, bi] = (color_sum[ri, gi, bi] / num).clip(0, 255)
                else:
                    lut_matrix[ri, gi, bi] = [ri * RGB2IDX, gi * RGB2IDX, bi * RGB2IDX]

    lut_matrix = smooth_and_extrapolate(lut_matrix, color_count, SMOOTHING_SIGMA)
    lut_image = lut_matrix.swapaxes(0, 2).reshape([LUT_IMAGE_SIZE, LUT_IMAGE_SIZE, 3])
    Image.fromarray(lut_image).save(lut_name, 'PNG')


def same_images(source_path, target_path):
    for source_img in source_path.glob('*.png'):
        target_img = target_path / source_img.name
        if target_img.exists():
            yield source_img, target_img


def load_and_map_images(source_file, target_file, color_sum, color_count):
    source = numpy.asarray(Image.open(source_file))
    target = numpy.asarray(Image.open(target_file))

    if source.shape != target.shape:
        raise TypeError(f'Shape different in {source_file.name} ({source.shape}) and '
                        f'{target_file.name} ({target.shape}): skipping image')

    count_pixels(source, target, color_sum, color_count)


@numba.jit(nopython=True)
def count_pixels(source, target, color_sum, color_count):
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            ri, gi, bi = source[x, y] // RGB2IDX
            color_sum[ri, gi, bi] += target[x, y]
            color_count[ri, gi, bi] += 1


def smooth_and_extrapolate(lut_matrix, sample_count, sigma):
    kernel = gaussian_kernel(sigma)
    for ri in tqdm(range(lut_matrix.shape[0]),
                   desc='fixing LUT'):
        for gi in range(lut_matrix.shape[1]):
            for bi in range(lut_matrix.shape[2]):
                lut_matrix[ri, gi, bi] = smooth_extrapolate_color(lut_matrix, sample_count, sigma,
                                                                  (ri, gi, bi), kernel)
    return lut_matrix


def gaussian_kernel(sigma):
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
    ri, gi, bi = coordinate
    radius = sigma * 3

    sum_color = numpy.zeros(3)
    sum_weights = 0
    for dr in range(max(0, ri-radius), min(ri+radius+1, LUT_CUBE_SIZE)):
        for dg in range(max(0, gi-radius), min(gi+radius+1, LUT_CUBE_SIZE)):
            for db in range(max(0, bi-radius), min(bi+radius+1, LUT_CUBE_SIZE)):
                weight = kernel[dr-ri+radius, dg-gi+radius, db-bi+radius]
                if count_matrix[dr, dg, db] > 5:
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
