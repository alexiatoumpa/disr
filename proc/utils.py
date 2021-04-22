import itertools, cv2

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace
import qsrlib_qstag.utils as utils
import qsrlib_qstag.qstag

from params import *

# -------------------------- VISUALIZATION FUNCTIONS --------------------------
def vis_convex_groundtruth_obj(object_coord, img, dimg, depth_list):
  for x in range(int(object_coord[0]), int(object_coord[2])):
    for y in range(int(object_coord[1]), int(object_coord[3])):
      px, py = (x,y)
      if (py<=dimg.shape[0]) and (px<=dimg.shape[1]):
        datapoint = dimg[int(py)][int(px)]
      else: break
      if datapoint >= depth_list[0] and datapoint <= depth_list[len(depth_list)-1]:
        cv2.circle(img,(x,y),1,[255,0,0],-1)

  return img


def vis_convex_surface_groundtruth_obj(object_coord, img, dimg, thres_min, thres_max):
    for x in range(int(object_coord[0]), int(object_coord[2])):
        for y in range(int(object_coord[1]), int(object_coord[3])):
            px, py = (x,y)
            if (py<=dimg.shape[0]) and (px<=dimg.shape[1]):
              datapoint = dimg[int(py)][int(px)]
            else: break

            if datapoint >= thres_min and datapoint <= thres_max:
                cv2.circle(img,(x,y),1,[0,0,0],-1)
            else:
                cv2.circle(img,(x,y),1,[255,255,255],-1)
    return img
# -----------------------------------------------------------------------------


# ----------------------------- QSR INTERACTIONS -----------------------------

def interaction(obj0_state_series, obj1_state_series, obj0_id, obj1_id,
                obj0_name, obj1_name, obj0_type, obj1_type, obj0_loc, obj1_loc,
                obj0_thr, obj1_thr):

  qsr_value = qsr

  ##############################################################################
  # Interaction between CONCAVE and ANY OTHER object.
  ##############################################################################
  if ((obj0_type == 'concave' and ((obj1_type == 'convex') or (obj1_type == 'convex_surface'))) or \
      (obj1_type == 'concave' and ((obj0_type == 'convex') or (obj0_type == 'convex_surface')))):
        
    if (obj0_type == 'concave' and ((obj1_type == 'convex') or (obj1_type == 'convex_surface'))):
      hole_min = obj0_thr[0]
      data_max = obj1_thr[1]
    else:
      hole_min = obj1_thr[0]
      data_max = obj0_thr[1]

    # Change the color of the bounding boxes if a RCC interaction occurs
    qsr_value = WhichInteraction(obj0_loc, obj1_loc, obj0_name, obj1_name) 
    area0 = obj0_loc[2] * obj0_loc[3]
    area1 = obj1_loc[2] * obj1_loc[3]
    if ((obj0_type == 'concave' and obj1_type == 'convex_surface' and area0 < area1) or \
        (obj1_type == 'concave' and obj0_type == 'convex_surface' and area1 < area0)) and \
       (qsr_value == 'cont' or qsr_value == 'conti'):
      obj0_state_series, obj1_state_series = change_Cont_to_Adj(obj0_state_series,
                                                                obj1_state_series)

    # If it is supposed to be 'dr'
    if not (qsr_value !='ni' and hole_min <= data_max and hole_min > 0): 
      obj0_state_series, obj1_state_series = change_any_to_NI(obj0_state_series,
                                                              obj1_state_series)
           
    ############################################################################
    # Interaction between (BOTH) CONCAVE objects.
    ############################################################################
  elif (obj0_type == 'concave' and obj1_type == 'concave'):
    # Find a reference blue-values and compare the other one against that one.
    # If both objects have a hole then it is most likely that the big one 
    # will contain the small one, so the big objects (big bounding box) is
    # most likely to have correct blue values.
    # For now we will color the small object red to distringuish it from the big one.
    area0 = obj0_loc[2] * obj0_loc[3]
    area1 = obj1_loc[2] * obj1_loc[3]
    if area0 > area1:
      ar_xul = obj1_loc[0] - obj1_loc[2]/2
      ar_xlr = obj1_loc[0] + obj1_loc[2]/2
      ar_yul = obj1_loc[1] - obj1_loc[3]/2
      ar_ylr = obj1_loc[1] + obj1_loc[3]/2
      obj1_type == 'convex'

      hole_min = obj0_thr[0] # Concave
      hole_max = obj0_thr[1] # Concave
                                
      sec = (obj1_thr[1] - obj1_thr[0])/section_number
      data_min = obj1_thr[1] - divider * sec # Convex 
      data_max = obj1_thr[1]

    elif area0 < area1:
      ar_xul = obj0_loc[0] - obj0_loc[2]/2
      ar_xlr = obj0_loc[0] + obj0_loc[2]/2
      ar_yul = obj0_loc[1] - obj0_loc[3]/2
      ar_ylr = obj0_loc[1] + obj0_loc[3]/2
      obj0_type == 'convex'
      hole_min = obj1_thr[0] # Concave
      hole_max = obj1_thr[1] # Concave

      sec = (obj0_thr[1] - obj0_thr[0])/section_number
      data_min = obj0_thr[1]  - divider * sec # Convex
      data_max = obj1_thr[1]

    #else:
    #  print('Object are the same size. No interaction.')
                                
    try:
      del ar_xul, ar_xlr, ar_yul, ar_ylr # Delete so the values are not stored.
    except NameError:
      pass

    qsr_value = WhichInteraction(obj0_loc, obj1_loc, obj0_name, obj1_name)
    if not (qsr_value !='ni' and hole_min <= data_max and hole_min > 0 ):
      # If it is supposed to be 'dr'
      obj0_state_series, obj1_state_series = change_any_to_NI(obj0_state_series,
                                                              obj1_state_series)


  elif (obj0_type == 'convex_surface' and obj1_type == 'convex_surface'):
    qsr_value = WhichInteraction(obj0_loc, obj1_loc, obj0_name, obj1_name)
        
    if qsr_value == 'cont' or qsr_value == 'conti': # supporting relation ocurring
      if qsr_value == 'cont':
        obj1_state_series = change_to_supporter(obj1_state_series)
      if qsr_value == 'conti':
        obj0_state_series = change_to_supporter(obj0_state_series)

    if qsr_value !='ni':   
      if obj1_thr[1] >= obj0_thr[1]: union_thres = obj0_thr[1] - obj1_thr[0]
      else: union_thres = obj1_thr[1] - obj0_thr[0]

      connected = False
      if obj0_thr[0] < obj1_thr[0]:
        if obj1_thr[1] < obj0_thr[1]: connected = True
      elif obj1_thr[0] < obj0_thr[0]:
        if obj0_thr[1] < obj1_thr[1]: connected = True
                               

      if not connected: # If it is supposed to be 'ni'
        obj0_state_series, obj1_state_series = change_any_to_NI(obj0_state_series,
                                                                obj1_state_series)

    ############################################################################
    # Interaction between CONVEX and CONVEX_SURFACE objects.
    ############################################################################     
  else: # If both objects are convex of one is convex_surface
    qsr_value = WhichInteraction(obj0_loc, obj1_loc, obj0_name, obj1_name)
    if (obj0_type == 'convex_surface' or obj1_type == 'convex_surface'):
      if qsr_value == 'cont' or qsr_value == 'conti':
        if obj0_type == 'convex_surface':
          obj0_state_series = change_to_supporter(obj0_state_series)
        if obj0_type == 'convex_surface':
          obj1_state_series = change_to_supporter(obj1_state_series)

        
    if qsr_value !='ni':
      if obj1_thr[1] >= obj0_thr[1]: union_thres = obj0_thr[1] - obj1_thr[0]
      else: union_thres = obj1_thr[1] - obj0_thr[0]
                

      if not (union_thres >0): # If it is supposed to be 'ni'
        obj0_state_series, obj1_state_series = change_any_to_NI(obj0_state_series,
                                                                obj1_state_series) 

  return obj0_state_series, obj1_state_series


def WhichInteraction(obj0_loc, obj1_loc, obj0_name, obj1_name):
    x0, y0, xs0, ys0 = obj0_loc[0], obj0_loc[1], obj0_loc[2], obj0_loc[3]
    x1, y1, xs1, ys1 = obj1_loc[0], obj1_loc[1], obj1_loc[2], obj1_loc[3]
    frame = 0
    qsrlib = QSRlib()
    options = sorted(qsrlib.qsrs_registry.keys())
    qsr_choise = qsr
    if qsr_choise in options:
        which_qsr = qsr_choise
    else:
        raise ValueError("qsr not found, keywords: %s" % options)


    world = World_Trace()

    object_types = {obj0_name: obj0_name,
                    obj1_name: obj1_name}

    dynamic_args = {"qstag": {"object_types" : object_types,
                              "params" : {"min_rows" : 1, "max_rows" : 1, "max_eps" : 3}},

                    "rcc2": {"qsrs_for": [(obj0_name, obj1_name)]},

                    "disr": {"qsrs_for": [(obj0_name, obj1_name)]},

                    "rcc5": {"qsrs_for": [(obj0_name, obj1_name)]},

                    "rcc8": {"qsrs_for": [(obj0_name, obj1_name)]}
                    }

    o0 = [Object_State(name=obj0_name, timestamp=frame, x=x0, y=y0, xsize=xs0, ysize=ys0)]
    o1 = [Object_State(name=obj1_name, timestamp=frame, x=x1, y=y1, xsize=xs1, ysize=ys1)]
    world.add_object_state_series(o0)
    world.add_object_state_series(o1)
    qsrlib_request_message = QSRlib_Request_Message(which_qsr=which_qsr,
                                                    input_data=world,
                                                    dynamic_args=dynamic_args)
    qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)
    qsr_value = ReturnInteraction(which_qsr, qsrlib_response_message)
    return qsr_value


def ReturnInteraction(which_qsr, qsrlib_response_message):
  for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
    for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
                    qsrlib_response_message.qsrs.trace[t].qsrs.values()):
      return str(v.qsr[which_qsr])


def change_Cont_to_Adj(obj0_state_series, obj1_state_series):
    obj0_state_series = obj0_state_series.negate_center_x()
    obj1_state_series = obj1_state_series.negate_center_x()
    return obj0_state_series, obj1_state_series


def change_any_to_NI(obj0_state_series, obj1_state_series):
    obj0_state_series = obj0_state_series.negate_center_y()
    obj1_state_series = obj1_state_series.negate_center_y()
    return obj0_state_series, obj1_state_series


def change_to_supporter(obj_state_series):
    obj_state_series = negate_centers(obj_state_series)
    return obj_state_series


def negate_centers(obj_state_series):
    obj_state_series = obj_state_series.negate_bounding_box_centers()
    return obj_state_series


def check_position_for_support_relation(obj0_loc, obj1_loc, obj0_name, obj1_name):
  x1, y1, xs1, ys1 = obj0_loc[0], obj0_loc[1], obj0_loc[2], obj0_loc[3]
  x2, y2, xs2, ys2 = obj1_loc[0], obj1_loc[1], obj1_loc[2], obj1_loc[3]

  qsr_value = WhichInteraction(obj0_loc, obj1_loc, obj0_name, obj1_name)
  if qsr_value == 'adj':
    # obj1 is ontop of obj2
    if (x1-xs1/2)>=(x2-xs2/2) and (x1+xs1/2)<=(x2+xs2/2) and \
       (y1-ys1/2) < (y2-ys2/2) and (y1+ys1/2) <= (y2+ys2/2):
      supporter1, supporter2 = False, True
    # obj2 is ontop of obj1
    elif (x2-xs2/2)>=(x1-xs1/2) and (x2+xs2/2)<=(x1+xs1/2) and \
         (y2-ys2/2) < (y1-ys1/2) and (y2+ys2/2) <= (y1+ys1/2):
      supporter1, supporter2 = True, False
    else:
      supporter1, supporter2 = False, False

  else:
    supporter1, supporter2 = False, False

  return supporter1, supporter2


def check_if_possible_to_support(this_object_states, other_object_states):
  other_supporter = other_object_states.check_negative_center_x() and \
                    other_object_states.check_negative_center_y()
  not_interacting = this_object_states.check_positive_center_x() and \
                    this_object_states.check_negative_center_y()
  this_supporter = this_object_states.check_negative_center_x() and \
                   this_object_states.check_negative_center_y()
  if (not other_supporter) and (not not_interacting) and (not this_supporter):
    return True
  return False


def get_centers(obj_state_series):
    cx, cy = obj_state_series.get_bounding_box_centers()
    return cx, cy



def qsr_message(obj0_state_series, obj1_state_series, obj0_name, obj1_name):
    qsrlib = QSRlib()
    options = sorted(qsrlib.qsrs_registry.keys())
    which_qsr = qsr

    world = World_Trace()

    # Create dynamic arguments for every qsr type
    object_types = {obj0_name: obj0_name,
                    obj1_name: obj1_name}

    dynamic_args = {"qstag": {"object_types" : object_types,
                              "params" : {"min_rows" : 1, "max_rows" : 1, "max_eps" : 3}},

                    "rcc2": {"qsrs_for": [(obj0_name, obj1_name)]},

                    "disr": {"qsrs_for": [(obj0_name, obj1_name)]},

                    "rcc5": {"qsrs_for": [(obj0_name, obj1_name)]},

                    "rcc8": {"qsrs_for": [(obj0_name, obj1_name)]}
                    }
    
    # Add all object states in the World.
    #for o in state_series:
    #    world.add_object_state_series(o)
    world.add_object_state_series([obj0_state_series])
    world.add_object_state_series([obj1_state_series])
    # Create request message
    qsrlib_request_message = QSRlib_Request_Message(which_qsr=which_qsr,
                                                    input_data=world,
                                                    dynamic_args=dynamic_args)
    qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)
    pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message)


def pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message):
    print(which_qsr, "request was made at ", str(qsrlib_response_message.req_made_at)
    + " and received at " + str(qsrlib_response_message.req_received_at)
    + " and finished at " + str(qsrlib_response_message.req_finished_at))
    print("---")
    print("timestamps:", qsrlib_response_message.qsrs.get_sorted_timestamps())
    print("Response is:")
    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        foo = str(t) + ": "
        for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
        qsrlib_response_message.qsrs.trace[t].qsrs.values()):
                foo += str(k) + ":" + str(v.qsr) + "; "
        print(foo)

