"""
Usage:
    python3 get_pixels_of_detections.py --image_list=<path to file containing image file names>
"""

import os
import numpy as np
import pickle
import json

def get_mask_pixels(image_file=None):
    #load list of detections
    with open("results.pkl", "rb") as rp:
        detect_lst = pickle.load(rp)
    #load JSON of all file names that went through detection
    with open('fnames.json', 'r') as read_file:
        fname_lst = json.load(read_file)
    #load list of file names that need to be processed
    #save it as a list
    imgs_to_process = []
    with open(image_file, 'r') as sl:
        for line in sl:
            imgs_to_process.append(line.split()[0])
    #save mask info for relevant images to new dictionary with frame identifier as key
    final_mask_dict = {}
    for img in imgs_to_process:
        #find index of image filename in fname_lst:
        img_loc = fname_lst.index(img.replace('_splash',''))
        final_mask_dict[img] = detect_lst[img_loc]['masks']
    #save final dictionary to pickle file
    with open("masks_dict.pkl", "wb") as md:
        pickle.dump(final_mask_dict,md)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Retrieve pixel locations of true positive splash masks')
    parser.add_argument('--image_list', required=True,
                        metavar="path to file containing list of image file names",
                        help='Path to list of images that have correct (TP) detections')
    args = parser.parse_args()

    assert args.image_list, "Provide --image_list with list of image file names"

    get_mask_pixels(image_file=args.image_list)
