#!/usr/bin/env python
# encoding: utf-8

import os
os.sys.path.append('lib')
from eval import *

datapath_prefix = '/home/breakpoint/data{}'
imageset_file = datapath_prefix.format(
    '/VOCdevkit/VOC2007/ImageSets/Main/test.txt')
annocache = 'output/cache/annocache.pkl'
annopath = datapath_prefix.format('/VOCdevkit/VOC2007/Annotations/{}.xml')

# PASS


#  def test_load_pascal_annotation():

gts = load_pascal_annotation(imageset_file, annopath, annocache)


# test show_boxes_by_class
classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
           'cow','diningtable','dog','horse','motorbike','person','pottedplant',
           'sheep','sofa','train','tvmonitor']
class_name = classes[1]
detpath = 'output/{}'.format('results/VOC2007/Main/comp4_det_test_{}.txt')

dets = load_pascal_detections(detpath, class_name)
dets = reorganize_detections(dets)
# test the first one of dets
#  image_ids = dets.keys()
#  image_filename_index = image_ids[0]
#  image_filename = image_filename_index + '.txt'
#  image_filename = datapath_prefix.format('/VOCdevkit/VOC2007/JPEGImages/{}.jpg')\
#  .format(image_filename_index)

#  show_boxes_by_class(
#  image_filename, dets[image_filename_index], class_name)
show_boxes_by_class_all(dets, class_name, datapath_prefix,
                        gts, year='2007', threshold=0.2)
