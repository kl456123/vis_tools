#!/usr/bin/env python
# encoding: utf-8


class BBox(object):
    def __init__(self, coords, code_type='xxyy'):
        self._bbox_xxyy = coords
        self._code_type = code_type
        self._bbox_xywh = None
        self.convert_bbox_format()

    def convert_bbox_format(self):
        if self._code_type == 'xywh':
            self._bbox_xywh = BBoxTools.convert_bbox_format(self._xxyy,self.code_type)

    def get_bbox(self,code_type):
        if code_type=='xxyy':
            return self._bbox_xxyy
        elif code_type=='xywh':
            return self._bbox_xywh
        else:
            raise TypeError('code type is unknown')



class BBoxTools(object):
    def __init__(self):
        pass

    def bbox_union(self, bbgt, bb, inters):
        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (bbgt[2] - bbgt[0] + 1.) *
               (bbgt[3] - bbgt[1] + 1.) - inters)
        return uni


# note that it is no matter with the order of arguments
    def bbox_intersection(self, bbgt, bb):
        # intersection
        ixmin = np.maximum(bbgt[0], bb[0])
        iymin = np.maximum(bbgt[1], bb[1])
        ixmax = np.minimum(bbgt[2], bb[2])
        iymax = np.minimum(bbgt[3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        return inters

    def bbox_iou(self, bbgt, bb):
        inters = bbox_intersection(bbgt, bb)
        uni = bbox_union(bbgt, bb, inters)
        return inters / uni

    def convert_bbox_format(self, bbox, code_type,inplace='False'):
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        if code_type == 'xywh':
            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin
        if inplace:
            bbox[0] = center_x
            bbox[1] = center_y
            bbox[2] = width
            bbox[3] = height
        return center_x,center_y,width,height
