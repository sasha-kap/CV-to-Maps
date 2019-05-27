"""
Usage:
    python3 save_splash_imgs.py --image=<URL or path to file directory>
"""

import skimage.io
import skimage.color
import os
import numpy as np
import pickle

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def load_and_color_splash(image_path=None):
    assert image_path
    output_path = os.path.join(os.path.abspath("."),'splash_imgs_fixed/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open("results.pkl", "rb") as rp:
        detect_lst = pickle.load(rp)
    i = 0
    for entry in os.scandir(image_path):
        fname = entry.path
        image = skimage.io.imread(fname)
        r_dict = detect_lst[i]
        if r_dict['masks'].size != 0:
            #apply color splash
            splash = color_splash(image, r_dict['masks'])
            #save output
            file_name = "{}_splash.jpg".format(fname.split('/')[-1].split('.')[0])
            skimage.io.imsave('{}{}'.format(output_path,file_name),splash)
        i += 1

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Save images with positive detections with splash masks added')
    parser.add_argument('--image', required=True,
                        metavar="path to images",
                        help='Path to images that went through detection pipeline')
    args = parser.parse_args()

    assert args.image, "Provide --image to apply color splash"

    load_and_color_splash(image_path=args.image)
