#!/usr/bin/env python
# encoding: utf-8

import os
import pprint

os.sys.path.append('lib')

from drawer import Drawer
from dataset import PASCAL_VOC_GTs_Dataset, Dets_Dataset


gts_name = 'gts'
gts_root_path = '/home/breakpoint/data'
cache_path = './output/cache'
year = '2007'


def test_gts_dataset():

    gts_dataset = PASCAL_VOC_GTs_Dataset(name=gts_name, root_path=gts_root_path,
                                         cache_path=cache_path, year=year)
    pprint.pprint(gts_dataset.get_image_gts(1))
    return gts_dataset


dets_name = 'dets'
dets_root_path = 'output/'


def test_dets_dataset():
    dets_dataset = Dets_Dataset(name=dets_name, root_path=dets_root_path,
                                cache_path=cache_path, year=year)
    return dets_dataset


gts_dataset = test_gts_dataset()
dets_dataset = test_dets_dataset()

drawer = Drawer(root_path=gts_root_path, path_template='VOCdevkit/VOC{}/JPEGImages/{}.jpg',
                year=year, gts_dataset=gts_dataset, dets_dataset=dets_dataset,
                save_path_template='output/images/{}.jpg')

drawer.draw_in_all_image(threshold=0.3, class_name='person')
