# -*- coding: utf-8 -*-
from abc import abstractmethod, ABCMeta
from qsrlib_qsrs.qsr_tetradic_abstractclass import QSR_Tetradic_Abstractclass


class QSR_DiSR_Abstractclass(QSR_Tetradic_Abstractclass):
    """Abstract class of DiSR relations.

    Values of the abstract properties
        * **_unique_id** = defined by the DiSR variant.
        * **_all_possible_relations** = defined by the DiSR variant.
        * **_dtype** = "bounding_boxes_2d"

    QSR specific `dynamic_args`
        * **quantisation_factor** (*float*) = 0.0: Threshold that determines whether two rectangle regions are disconnected.
    """

    __metaclass__ = ABCMeta

    _dtype = "bounding_boxes_2d"
    """str: On what kind of data the QSR works with."""

    def __init__(self):
        """Constructor."""
        super(QSR_DiSR_Abstractclass, self).__init__()

        self.__qsr_params_defaults = {"quantisation_factor": 0.0}
        """float: ?"""

    def _process_qsr_parameters_from_request_parameters(self, req_params, **kwargs):
        qsr_params = self.__qsr_params_defaults.copy()
        try:
            qsr_params["quantisation_factor"] = float(req_params["dynamic_args"][self._unique_id]["quantisation_factor"])
        except (KeyError, TypeError):
            try:
                qsr_params["quantisation_factor"] = float(req_params["dynamic_args"]["for_all_qsrs"]["quantisation_factor"])
            except (TypeError, KeyError):
                pass
        return qsr_params

    def _compute_qsr(self, bb1, bb2, qsr_params, **kwargs):
        """Return symmetrical RCC8 relation
            :param bb1: diagonal points coordinates of first bounding box (x1, y1, x2, y2)
            :param bb2: diagonal points coordinates of second bounding box (x1, y1, x2, y2)
            :param q: quantisation factor for all objects
            :return: an RCC8 relation from the following:
                'dc'     bb1 is disconnected from bb2
                'ec'     bb1 is externally connected with bb2
                'po'     bb1 partially overlaps bb2
                'eq'     bb1 equals bb2
                'tpp'    bb1 is a tangential proper part of bb2
                'ntpp'   bb1 is a non-tangential proper part of bb2
                'tppi'   bb2 is a tangential proper part of bb1
                'ntppi'  bb2 is a non-tangential proper part of bb1
                 +-------------+         +-------------+
                 |a            |         |c            |
                 |             |         |             |
                 |     bb1     |         |     bb2     |
                 |             |         |             |
                 |            b|         |            d|
                 +-------------+         +-------------+
        """

        """ Specific to DiSR relations
            +-------------------------+
            | cx | cy | Action | Flag |
            |----|----|--------|------|
            |  P |  P | normal |  1   |
            |  P |  N | _ : NI |  2   |
            |  N |  P |Cont:Abj|  3   |
            |  N |  N | support| 4,-4 |
            +-------------------------+

        """

        q = qsr_params["quantisation_factor"]

        ax, ay, bx, by = bb1
        cx, cy, dx, dy = bb2


        # Calculate centers of bounding boxes.
        center1_x = ax + abs(bx - ax)/2
        center1_y = ay + abs(by - ay)/2

        center2_x = cx + abs(dx - cx)/2
        center2_y = cy + abs(dy - cy)/2

        # Calculate the size of the bounding boxes.
        size1_x = bx - ax
        size1_y = by - ay
        size2_x = dx - cx
        size2_y = dy - cy

        Action1 = 1 # normal
        Action2 = 1 # normal

        if center1_x<0 and center1_y<0:
            Action1 = 4 # support
        elif center1_y<0 and center2_y<0:
            Action1 = 2 # _ : NI
            Action2 = 2 # _ : NI
        elif center1_x<0 and center2_x<0:
            Action1 = 3 # Cont : Adj
            Action2 = 3 # Cont : Adj

        if center2_x<0 and center2_y<0:
            Action2 = 4 # support




        # If one of the objects is marked for an operation(action), we keep the
        # bounding boxes information unchanged to check for their actual RCC5
        # relations.
        if center1_x <0 or center1_y <0 :
            new_center1_x = abs(center1_x)
            new_center1_y = abs(center1_y)
            ax = new_center1_x - size1_x/2
            ay = new_center1_y - size1_y/2
            bx = new_center1_x + size1_x/2
            by = new_center1_y + size1_y/2
        if center2_x <0 or center2_y <0 :
            new_center2_x = abs(center2_x)
            new_center2_y = abs(center2_y)
            cx = new_center2_x - size2_x/2
            cy = new_center2_y - size2_y/2
            dx = new_center2_x + size2_x/2
            dy = new_center2_y + size2_y/2
        
        # CALCULATE EQ
        # Is object1 equal to object2
        if (bb1 == bb2):
            return self._convert_to_requested_rcc_type("equal")


        # Are objects disconnected?
        # Cond1. If A's left edge is to the right of the B's right edge, - then A is Totally to right Of B
        # Cond2. If A's right edge is to the left of the B's left edge, - then A is Totally to left Of B
        # Cond3. If A's top edge is below B's bottom edge, - then A is Totally below B
        # Cond4. If A's bottom edge is above B's top edge, - then A is Totally above B

        #    Cond1           Cond2          Cond3         Cond4
        if ((ax-q > dx+q) or (bx+q < cx-q) or (ay-q > dy+q) or (by+q < cy-q)) or \
          ((Action1 == 2) and (Action2 == 2)):
            return self._convert_to_requested_rcc_type("not_interacting")

        # Is one object inside the other ()
        BinsideA = (ax <= cx) and (ay <= cy) and (bx >= dx) and (by >= dy)
        AinsideB = (ax >= cx) and (ay >= cy) and (bx <= dx) and (by <= dy)

        # Do objects share an X or Y (but are not necessarily touching)
        sameX = (abs(ax - cx)<=q) or (abs(ax - dx)<=q) or (abs(bx - cx)<=q) or (abs(bx - dx)<=q)
        sameY = (abs(ay - cy)<=q) or (abs(ay - dy)<=q) or (abs(by - cy)<=q) or (abs(by - dy)<=q)

        if AinsideB and (sameX or sameY) and (Action1 == 1) and (Action2 == 1):
            return self._convert_to_requested_rcc_type("containee")

        if BinsideA and (sameX or sameY) and (Action1 == 1) and (Action2 == 1):
            return self._convert_to_requested_rcc_type("container")

        if AinsideB and (Action1 == 1) and (Action2 == 1):
            return self._convert_to_requested_rcc_type("containee")

        if BinsideA and (Action1 == 1) and (Action2 == 1):
            return self._convert_to_requested_rcc_type("container")

        # Are objects touching?
        # Cond1. If A's left edge is equal to B's right edge, - then A is to the right of B and touching
        # Cond2. If A's right edge is qual to B's left edge, - then A is to the left of B and touching
        # Cond3. If A's top edge equal to B's bottom edge, - then A is below B and touching
        # Cond4. If A's bottom edge equal to B's top edge, - then A is above B and touching

        # If quantisation overlaps, but bounding boxes do not then edge connected,
        # include the objects edges, but do not include the quantisation edge

        # If a EC relation occures and one of the objects is marked as a supporter
        # then we are referring to a Sup or a Supi relation in DiSR.
        if (((cx-q) <= (bx+q)) and ((cx-q) >= (bx)) or \
                        ((dx+q) >= (ax-q)) and ((dx+q) <= (ax)) or \
                        ((cy-q) <= (by+q)) and ((cy-q) >= (by)) or \
                        ((dy+q) >= (ay-q)) and ((dy+q) <= (ay))) and \
            (Action1 == 4):
            return self._convert_to_requested_rcc_type("supportee")
        if (((cx-q) <= (bx+q)) and ((cx-q) >= (bx)) or \
                        ((dx+q) >= (ax-q)) and ((dx+q) <= (ax)) or \
                        ((cy-q) <= (by+q)) and ((cy-q) >= (by)) or \
                        ((dy+q) >= (ay-q)) and ((dy+q) <= (ay))) and \
            (Action2 == 4):
            return self._convert_to_requested_rcc_type("supporter")

        # If a PO relation occures and one of the objects is marked as a supporter
        # then we are referring to a Sup or a Supi relation in DiSR.
        if (Action1 == 4):
            return self._convert_to_requested_rcc_type("supportee")
        if (Action2 == 4):
            return self._convert_to_requested_rcc_type("supporter")

        # If none of the other conditions are met, the objects must be parially overlapping
        return self._convert_to_requested_rcc_type("adjacent")


    @abstractmethod
    def _convert_to_requested_rcc_type(self, qsr):
        """Overwrite this function to filter and return only the relations corresponding to the particular DiSR version.

        Example for RCC2: return qsr if qsr =="dc" else "c"

        :param qsr: The RCC8 relation between two objects
        :type qsr: str
        :return: The part of the tuple you would to have as a result
        :rtype: str
        """
        return
