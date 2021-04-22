import numpy as np
import cv2, copy

from utils import *
from params import *

def get_depth(dimg, object_coord):
  """
  Get the depth information of the detected object from the depth image in a 
  list.
  """
  depth_info = []
  for x in range(int(object_coord[0]), int(object_coord[2])):
    for y in range(int(object_coord[1]), int(object_coord[3])):
      if dimg[y][x]!= 0 and dimg[y][x]!=255: # NOISE: if 0 or 255
        depth_info.append(dimg[y][x])
  return depth_info


def process_depth(depth_info):
  """
  Process the depth information of an image by computing its depth distribution
  over all pixels which define the object in the image. From there we compute
  the derivative of the depth to filter out noise depth datapoints.

  Parameters:
  - depth_info: (type: list(int)) ascended ordered list of depth information
  """


  """
  1) Finding the group where the object is most likely to exist considering the
     derivative of the depth distribution.

     Slope of the secant line: Q(h) = (f[i] - f[i-h]) / h
     Values used (defined in __init__.py):
      - sensitivity: 10 (10 is good value)
      - h = 1 (1: secant line)
  """   
  
  derivative_checkpoints = [i for i in range(1, len(depth_info)) if (((depth_info[i] - depth_info[i-h])/h) > sensitivity)]
  derivative_checkpoints.insert(0,0)
  derivative_checkpoints.append(len(depth_info))
  obj_group_start, obj_group_end = 0,0

  """
  2) Find maximum amount of depth pixels corresponding to the detected object;
     find the longest sequence of sorted depth information, according to the
     derivative peaks.
  """
  max_difference = 0 
  for i in range(1,len(derivative_checkpoints)):
    difference_groups = derivative_checkpoints[i] - derivative_checkpoints[i-1]
    if max_difference < difference_groups:
      max_difference = difference_groups
      obj_group_start, obj_group_end = derivative_checkpoints[i-1], derivative_checkpoints[i]

  """
  3) Coarse-graining the depth list according to the derivative and the biggest
     group of pixels; keeping only the object's information
  """
  depth_list = copy.deepcopy(depth_info[obj_group_start:obj_group_end])
  pixel_list = range(obj_group_end - obj_group_start)

  dmean = np.mean(depth_list)
  sigma = np.std(depth_list)
  thres_min = dmean - sigma
  thres_max = dmean + sigma

  return depth_list, pixel_list, dmean, sigma, thres_min, thres_max, derivative_checkpoints


def find_concavity(dimg, img, thres_max, thres_min, thres_area, depth_list, object_data):

  if thres_area > maximum_threshold: # CONCAVE OBJECT
    object_data['Type'] = 'concave'
    section = thres_area / divider
  
    minDeepdatapoint = thres_max - section * section_number
    # Concaveness: min , max
    object_data['Threshold_values'] = [minDeepdatapoint, thres_max]

    crop_obj = img[int(object_data['Box'][1]):int(object_data['Box'][3]), 
                   int(object_data['Box'][0]):int(object_data['Box'][2])]


    # Color object bounding box with regions of concaveness.        
    #if init.vis_concave_regions: img = vis_concaveness(img, dimg, minDeepdatapoint, section, thres_min, thres_max, numo) #TODO

    regions = concave_object_regions(dimg, object_data['Box'], minDeepdatapoint, 
                                     section, thres_min, thres_max)
    is_this_convex = check_contour_hierarchy(regions, object_data['Box'])


    if is_this_convex:
      # this is a convex object
      object_data['Type'] = 'convex_surface'
      object_data['Threshold_values'] = [thres_min, thres_max]
      img = vis_convex_surface_groundtruth_obj(object_data['Box'], img, dimg, 
                                               thres_min, thres_max)
    #elif init.vis_concave_regions: img = vis_concaveness(img, dimg, minDeepdatapoint, section, thres_min, thres_max, numo) #TODO
       
  else: # CONVEX OBJECT
    object_data['Type'] = 'convex'
    object_data['Threshold_values'] = [thres_min, thres_max]
    img = vis_convex_groundtruth_obj(object_data['Box'], img, dimg, depth_list)


def concave_object_regions(dimg, object_coord, min_depth, section, thres_min, thres_max):

  depth_cropped = dimg[int(object_coord[1]):int(object_coord[3]), int(object_coord[0]):int(object_coord[2])]

  regions_ = np.zeros_like(depth_cropped)
  regions_ = np.where(depth_cropped >= (min_depth-section), \
                     np.where(depth_cropped <= thres_max, 100, regions_), \
                     regions_)
  regions_ = np.where(regions_!=100, \
                      np.where(depth_cropped >= (min_depth-section), \
                               np.where(depth_cropped <= min_depth, 100, regions_), \
                               regions_), \
                      regions_)
  regions_ = np.where(regions_!=100, \
                      np.where(depth_cropped >= thres_min, \
                               np.where(depth_cropped <= (min_depth-section), 100, regions_), \
                               regions_), \
                      regions_)
  regions_ = np.where(regions_!=100, \
                      np.where(depth_cropped > thres_max, 50, regions_), \
                      regions_)

  regions_ = np.where(regions_<50, \
                      np.where(depth_cropped < thres_min, \
                               np.where(depth_cropped != 0, 100, regions_), \
                               regions_), \
                      regions_)
  return regions_


def check_contour_hierarchy(box, object_coord):
  """
  Find which of the concave objects are actual concave and which of them are 
  convex long surfaces.
  To differentiate between these two we search for the concave curve by:
  - finding the contour of the BLUE area
  - define the hierarchy of the contours and pick the inner child if it 
    significant enough in terms of area occupying.
  """
  # compute area of the bounding box of the object
  object_area = abs(object_coord[3] - object_coord[1]) * \
                abs(object_coord[2] - object_coord[0])
  hierarchy, areas = contour_hierarchy(box)

  multiplier_value = 0.03
  # Find if the object in concave or a convex surface by looking if it has any 
  # parent, no children, and exceeds the area threshold.
  convexity_flag = True
  for i in range(len(areas)):
    hierarchy_info = hierarchy[0][i]
    parent = hierarchy_info[3]
    child = hierarchy_info[2]
    area = areas[i]
    # this is a concave object
    if parent!=-1 and child==-1 and area>=(multiplier_value*object_area): convexity_flag = False

  return convexity_flag


def contour_hierarchy(cropped_object):
  # convert to gay scale (if input param is img)
  crop_gray = cropped_object.astype(np.uint8)
  # binarize the information
  threshold_value = crop_gray.min() 

  ret, thresh = cv2.threshold(crop_gray, threshold_value, 255, 0)
  # find hierarchy of contours 
  contour_obj, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 
                                                      cv2.CHAIN_APPROX_SIMPLE)
                            
  # compute area of the concave area
  areas = [cv2.contourArea(c) for c in contours]
  return hierarchy, areas


