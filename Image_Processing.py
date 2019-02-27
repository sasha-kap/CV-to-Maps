
# coding: utf-8

from PIL import Image
import os

#FUNCTION TO CROP BOTTOM LEFT AND UPPER RIGHT PORTIONS OF EACH FRAME
#Assumes 1920x1080 frame size
def txt_crops(input_image):
    bl_crop_width = 290
    bl_crop_height = 220
    width, height = input_image.size
    #box coordinates: x, y of upper left corner, then x, y of lower right corner
    #0, 0 coordinates are upper left corner
    bl_box = 0, height - bl_crop_height, bl_crop_width, height
    ur_box = 1710, 15, 1910, 82
    return input_image.crop(bl_box), input_image.crop(ur_box)

#CROP AND SAVE BOTTOM-LEFT AND UPPER-RIGHT PARTS OF EACH IMAGE
try:
    if not os.path.exists('./s3data/loc_box'):
        os.makedirs('./s3data/loc_box')
except OSError:
    print('Error: Creating directory of location boxes')

try:
    if not os.path.exists('./s3data/gps_box'):
        os.makedirs('./s3data/gps_box')
except OSError:
    print('Error: Creating directory of GPS boxes')

indir = './s3data/frames_1ps/'
outdir_loc = './s3data/loc_box'
outdir_gps = './s3data/gps_box'
for root, dirs, filenames in os.walk(indir):
    for filename in filenames:
        im = Image.open(indir + filename)
        place_time_region, lat_lon_region = txt_crops(im)
        place_time_region.save(f'{outdir_loc}/{filename}','JPEG')
        lat_lon_region.save(f'{outdir_gps}/{filename}','JPEG')
