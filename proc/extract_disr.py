from params import *
import sys
sys.path.append(disrrepo_path + 'qsrlib/src/')

from qsrlib_io.world_trace import Object_State
from proc.depth_process import *
import copy


def findConcavityType(img, dimg, object_data):

  depth_info = get_depth(dimg, object_data['Box'])
  if len(depth_info) != 0:
    depth_info = sorted(depth_info)
    depth_list, pixel_list, dmean, sigma, thres_min, thres_max, derivative_checkpoints = \
     process_depth(depth_info)

    thres_area = thres_max - thres_min

    find_concavity(dimg, img, thres_max, thres_min, thres_area, depth_list, object_data)
  else: # object out of range of sensor; undefined object type
    object_data['Type'] = 'notype'
    object_data['Threshold_values'] = []



def extractDiSR(object_data, img, dimg, frame):
  processed_img = copy.deepcopy(img)

  for obj in range(len(object_data)):
    findConcavityType(processed_img, dimg, object_data[obj])
 

  if len(object_data) > 1: ### find all pairs of interacting objects in the scene
    for pair in itertools.combinations(range(len(object_data)), 2):
      # Get object id
      obj0_id = object_data[pair[0]]['ID']
      obj1_id = object_data[pair[1]]['ID']

      # Get object name
      obj0_name = object_data[pair[0]]['Name']
      obj1_name = object_data[pair[1]]['Name']

      # Get object type
      obj0_type = object_data[pair[0]]['Type']
      obj1_type = object_data[pair[1]]['Type']

      # Get object location
      obj0_loc = object_data[pair[0]]['Box']
      obj1_loc = object_data[pair[1]]['Box']

      # Get obj threshold values
      obj0_thr = object_data[pair[0]]['Threshold_values']
      obj1_thr = object_data[pair[1]]['Threshold_values']

      # object states
      x, y = object_data[pair[0]]['Centers'][0], object_data[pair[0]]['Centers'][1]
      xs, ys = object_data[pair[0]]['Size'][0], object_data[pair[0]]['Size'][1]
      obj0_loc = [x, y, xs, ys]
      obj0_state_series = Object_State(name=obj0_name, timestamp=frame, 
                                       x=x, y=y, xsize=xs, ysize=ys)
      x, y = object_data[pair[1]]['Centers'][0], object_data[pair[1]]['Centers'][1]
      xs, ys = object_data[pair[1]]['Size'][0], object_data[pair[1]]['Size'][1]
      obj1_loc = [x, y, xs, ys]
      obj1_state_series = Object_State(name=obj1_name, timestamp=frame, 
                                       x=x, y=y, xsize=xs, ysize=ys)

      obj0_state_series, obj1_state_series = interaction(obj0_state_series,
                                                         obj1_state_series,
                                                         obj0_id, obj1_id,
                                                         obj0_name, obj1_name,
                                                         obj0_type, obj1_type,
                                                         obj0_loc, obj1_loc,
                                                         obj0_thr, obj1_thr)

      # 'On' relation
      supporter0, supporter1 = check_position_for_support_relation(obj0_loc,
                                                                   obj1_loc,
                                                                   obj0_name,
                                                                   obj1_name)

      if supporter0 and check_if_possible_to_support(obj0_state_series,
                                                     obj1_state_series):
        obj0_state_series = negate_centers(obj0_state_series)
        type0 = 'convex_surface'

      if supporter1 and check_if_possible_to_support(obj1_state_series,
                                                     obj0_state_series):
        obj1_state_series = negate_centers(obj1_state_series)
        type1 = 'convex_surface'


      qsr_message(obj0_state_series, obj1_state_series, obj0_name, obj1_name)

