# -*- coding: utf-8 -*-
from __future__ import print_function, division
from qsrlib_qsrs.qsr_disr_abstractclass import QSR_DiSR_Abstractclass


class QSR_DiSR(QSR_DiSR_Abstractclass):
    """Symmetrical DiSR relations.

    Values of the abstract properties
        * **_unique_id** = "disr"
        * **_all_possible_relations** = ("ni", "adj", "sup", "supi", "cont", "conti", "eq")
        * **_dtype** = "bounding_boxes_2d"

    QSR specific `dynamic_args`
        * **'quantisation_factor'** (*float*) = 0.0: Threshold that determines whether two rectangle regions are disconnected.

    .. seealso:: For further details about DiSR, refer to its :doc:`description. <../handwritten/qsrs/disr>`
    """

    _unique_id = "disr"

    _all_possible_relations = ("ni", "adj", "sup", "supi", "cont", "conti", "eq")

    __mapping = {"not_interacting": "ni",
                 "adjacent": "adj",
                 "supportee": "sup",
                 "supporter": "supi",
                 "containee": "cont",
                 "container": "conti",
                 "equal": "eq"}

    def __init__(self):
        super(QSR_DiSR, self).__init__()

    def _convert_to_requested_rcc_type(self, qsr):
        return self.__mapping[qsr]
