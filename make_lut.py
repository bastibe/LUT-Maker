import pathlib
import numpy
import numba
from PIL import Image
from tqdm import tqdm


MIN_COLOR_SAMPLES = 5
LUT_CUBE_SIZE = 64
LUT_IMAGE_SIZE = 512
RGB2IDX = int(256 / LUT_CUBE_SIZE)
assert LUT_CUBE_SIZE**3 == LUT_IMAGE_SIZE**2, "LUT configuration invalid"


def main(source_path, target_path, lut_name):
    color_sum = numpy.zeros([LUT_CUBE_SIZE, LUT_CUBE_SIZE, LUT_CUBE_SIZE, 3], dtype='uint64')
    color_count = numpy.zeros([LUT_CUBE_SIZE, LUT_CUBE_SIZE, LUT_CUBE_SIZE], dtype='uint64')

    for source_img, target_img in tqdm(list(same_images(source_path, target_path))):
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
                    lut_matrix[ri, gi, bi] = color_sum[ri, gi, bi] / num
                else:
                    lut_matrix[ri, gi, bi] = [ri * RGB2IDX, gi * RGB2IDX, bi * RGB2IDX]

    lut_matrix = lut_matrix.clip(0, 255)
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
                        f'{target_file.name} ({target.shape})')

    count_pixels(source, target, color_sum, color_count)


@numba.jit(nopython=True)
def count_pixels(source, target, color_sum, color_count):
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            ri, gi, bi = source[x, y] // RGB2IDX
            color_sum[ri, gi, bi] += target[x, y]
            color_count[ri, gi, bi] += 1


def gaussian(x, mu=0, sigma_squared=1):
    sigma = sigma_squared ** 0.5
    pi = math.pi
    return 1.0 / (sigma * (2*pi)**0.5) * exp( -(x - mu)**2 / (2 * sigma_squared) )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Make a LUT from image pairs')
    parser.add_argument('source_dir', type=str, help='directory containing unprocessed images')
    parser.add_argument('target_dir', type=str, help='directory containing processed images')
    parser.add_argument('lut_name', type=str, help='name of the resulting LUT')
    args = parser.parse_args()
    main(pathlib.Path(args.source_dir), pathlib.Path(args.target_dir), args.lut_name)
