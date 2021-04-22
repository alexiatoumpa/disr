import numpy as np
import pandas as pd
import os, cv2

from params import *
from proc.extract_disr import *


def get_objects_coord(lines, ul, lr):
  for line in lines:
    prsl = line.split(',')
    fr = int(prsl[0])-1 # index 0
    obj = prsl[1]
    ulx, uly = float(prsl[2]),float(prsl[3])
    lrx, lry = float(prsl[4]),float(prsl[5])

    ul[fr] = ulx, uly
    lr[fr] = lrx, lry


def get_object_coordinates(ul, lr):
  ulx, uly = float(ul[0]), float(ul[1])
  lrx, lry = float(lr[0]), float(lr[1])
  return [ulx, uly, lrx, lry]


def calculate_boxcenter(object_coord):
  boxcenter_x = object_coord[0]+(object_coord[2]-object_coord[0])/2
  boxcenter_y = object_coord[3]+(object_coord[1]-object_coord[3])/2
  return boxcenter_x, boxcenter_y


def calculate_boxsize(object_coord):
  xs = abs(object_coord[2]-object_coord[0])
  ys = abs(object_coord[1]-object_coord[3])
  return xs, ys


def parse_videos(VIDEOS, ANNOTATIONS):
  for v in range(len(VIDEOS)):
    annotations = ANNOTATIONS[v]
    video = VIDEOS[v]
    depth_video = VIDEOS[v]
    subject = video_info[v]['subject']
    activity = video_info[v]['activity']
    task = video_info[v]['task']
    video_length = len(os.listdir(video))/2
    number_of_objects = len(annotations)       

    UL = np.zeros(((number_of_objects,video_length,2)))
    LR = np.zeros(((number_of_objects,video_length,2)))

    for o in range(number_of_objects):
      pathobj = DATASET_DIR +  "enhanced_annotations/" + subject[0:8] + \
         '_annotations/' + activity + '/' + task + '_obj' + str(o+1) + '.txt'
      file = open(pathobj, 'r')
      frame_lines = file.readlines()
      if len(frame_lines)>video_length:
        frame_lines = frame_lines[:-1]
      get_objects_coord(frame_lines, UL[o], LR[o])


    for frame in range(100,video_length):
      imgname = video + 'RGB_' + str(frame+1) + '.png'
      img = cv2.imread(imgname)
      dimgname = depth_video + 'Depth_' + str(frame+1) + '.png'
      dimg = cv2.imread(dimgname,0)

      object_data = []
      for objid in range(number_of_objects): 
        object_ = {}
        object_['ID'] = objid
        object_['Name'] = str(object_['ID'])
        # Object's coordinates: [x_ul,y_ul,x_lr,y_lr]
        object_coord = get_object_coordinates(UL[object_['ID']][frame], 
                                              LR[object_['ID']][frame])
        boxcenter_x, boxcenter_y = calculate_boxcenter(object_coord)
        boxsize_x, boxsize_y = calculate_boxsize(object_coord)
        object_['Box'] =  object_coord
        object_['Centers'] = [boxcenter_x, boxcenter_y]
        object_['Size'] = [boxsize_x, boxsize_y]
        object_data.append(object_)


      extractDiSR(object_data, img, dimg, frame)


      cv2.imshow("RGB Image", img)
      k = cv2.waitKey(30) & 0xff # press ESC to go to the next video
      if k == 27: break
        

if __name__ == '__main__':

  DATASET_DIR = dataset_path
  csvfile = fold_data_file
  FOLD_VIDEOS = pd.read_csv(csvfile, dtype = {'task': str})

  video_info = [ {'subject': row['subject'],
                  'activity': row['activity'],
                  'task': row['task']} for index, row in FOLD_VIDEOS.iterrows()]


  VIDEOS = [ DATASET_DIR +  "images/" + data['subject'] + "/" + data['activity'] + \
             "/" + data['task'] + "/" for data in video_info]
  ANNOTATIONS = [[ DATASET_DIR +  "enhanced_annotations/" + data['subject'][:9] + \
                   "annotations/" + data['activity'] + "/" + \
                 object_file for object_file in os.listdir(DATASET_DIR + \
                 "enhanced_annotations/" + data['subject'][:9] + "annotations/" + \
                 data['activity'] + "/") if object_file.startswith(data['task'] + \
                 "_obj")] for data in video_info]
  parse_videos(VIDEOS, ANNOTATIONS)

