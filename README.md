# Create LUTs from image pairs

given two directories full of JPGs, one *source* directory containing neutral output of an image editing program, and one *target* directory containing the same images with a desired look, the script 

    python make_lut.py source_dir/ target_dir/ lut.png

calculates a LUT png that transforms source images to target images.

Make sure that source pixels map exactly to target pixels, so disable all lens corrections (or better, shoot with a manual lens). Images are subsampled by a factor of 5 to cover any sharpening artifacts, but disable any sharpening all the same.

This is currently very much work in progress, but I have used it to create LUTs of Fujifilm Film Simulations.
