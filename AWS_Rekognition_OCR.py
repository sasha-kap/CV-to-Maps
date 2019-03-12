
# coding: utf-8

import boto3
from datetime import datetime
import string
import json

bucket_name = 'txt-boxes'

#produce count of files in the folder containing frame images
bucket = boto3.resource('s3').Bucket(bucket_name)
n_images = 0
for object in bucket.objects.all():
    if object.key.endswith('jpg'):
        n_images += 1
n_images //= 2

def key_gen(n_frames):
    i = 1
    while i <= n_frames:
        yield f'loc_box/frame{i}.jpg', f'gps_box/frame{i}.jpg'
        i += 1

client = boto3.client('rekognition')

image_dict = {}

for image_key_loc, image_key_gps in key_gen(n_images):
    response = client.detect_text(Image={'S3Object':{'Bucket':bucket_name,'Name':image_key_loc}})
    textDetections = response['TextDetections']

    loc_time_dict = {}

    for text in textDetections:
        if text['Type'] == 'LINE':
            if text['Id'] == 0:
                loc_time_dict['State'] = text['DetectedText']
            elif text['Id'] == 1:
                if text['DetectedText'].startswith('ELEVATION'):
                    loc_time_dict['Elev'] = int(text['DetectedText'].split(' ')[1])
                else:
                    loc_time_dict['County'] = text['DetectedText'].split(' ')[0]
            elif text['Id'] == 2:
                if text['DetectedText'].startswith('ELEVATION'):
                    loc_time_dict['Elev'] = int(text['DetectedText'].split(' ')[1])
                else:
                    loc_time_dict['Heading'] = int(text['DetectedText'].split(' ')[1])
            elif text['Id'] == 3:
                if text['DetectedText'].startswith('HEADING'):
                    loc_time_dict['Heading'] = int(text['DetectedText'].split(' ')[1])
                else:
                    loc_time_dict['DateTime'] = datetime.strptime(text['DetectedText'].split(' ')[1],'%m/%d/%y')
            elif text['Id'] == 4:
                if text['DetectedText'].startswith('DATE'):
                    loc_time_dict['DateTime'] = datetime.strptime(text['DetectedText'].split(' ')[1],'%m/%d/%y')
                else:
                    loc_time_dict['DateTime'] = datetime.combine(loc_time_dict['DateTime'],
                                                                 datetime.strptime(text['DetectedText'][6:],'%I:%M:%S %p').time())
            elif text['Id'] == 5:
                loc_time_dict['DateTime'] = datetime.combine(loc_time_dict['DateTime'],
                                                             datetime.strptime(text['DetectedText'][6:],'%I:%M:%S %p').time())

    response = client.detect_text(Image={'S3Object':{'Bucket':bucket_name,'Name':image_key_gps}})
    textDetections = response['TextDetections']

    GPS_dict = {}
    for text in textDetections:
        if text['Type'] == 'LINE':
            remove = string.punctuation + string.ascii_uppercase
            remove = remove.replace(".", "") # don't remove periods
            if text['Id'] == 0:
                GPS_dict['Lat'] = text['DetectedText'].translate({ord(char): None for char in remove})
                GPS_dict['Lat'] = (float(GPS_dict['Lat'].split(' ')[0]) + float(GPS_dict['Lat'].split(' ')[1])/60 +
                float(GPS_dict['Lat'].split(' ')[2])/3600)
            elif text['Id'] == 1:
                GPS_dict['Lon'] = text['DetectedText'].translate({ord(char): None for char in remove})
                GPS_dict['Lon'] = (float(GPS_dict['Lon'].split(' ')[0]) + float(GPS_dict['Lon'].split(' ')[1])/60 +
                float(GPS_dict['Lon'].split(' ')[2])/3600)

    full_frame_dict = {**loc_time_dict,**GPS_dict}

    dict_key = image_key_loc.split('/')[-1]
    image_dict[dict_key] = full_frame_dict

json = json.dumps(image_dict)
f = open("image_txt","w")
f.write(json)
f.close()
