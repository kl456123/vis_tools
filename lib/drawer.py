#!/usr/bin/env python
# encoding: utf-8

import cv2
import matplotlib.pyplot as plt
from random import random as rand
import os


class Drawer(object):
    def __init__(self, root_path, path_template, year, gts_dataset, dets_dataset, save_path_template):
        self._root_path = root_path
        # self._path = root_path
        self._path_template = path_template
        self._gts_dataset = gts_dataset
        self._dets_dataset = dets_dataset
        self._save_path_template = save_path_template
        self._year = year

    def generate_saved_image_path(self, image_id):
        return self._save_path_template.format(image_id)

    def draw_in_one_image(self, image_id, scale=1.0, threshold=0.03, save=True, class_name=''):
        path_to_img = self.get_abspath_from_id(image_id)
        im = cv2.imread(path_to_img)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)
        one_imgid_dets = self._dets_dataset.get_data(image_id)
        one_imgid_gts = self._gts_dataset.get_data(image_id)

        count_gt_per_image = 0
        for one_imgid_gt in one_imgid_gts:
            if one_imgid_gt['class'] == class_name or\
                    class_name == '':
                count_gt_per_image += 1
            else:
                continue
            bbox = []
            for i in range(len(one_imgid_gt['bbox'])):
                bbox.append(one_imgid_gt['bbox'][i] * scale)
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1],
                           '{:s} '.format('gt'),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')

        if count_gt_per_image == 0:
            # all bbox is false positive
            print 'no detections are proposaled'
            return

        for one_imgid_det in one_imgid_dets:
            score = one_imgid_det['confidence']
            if score < threshold:
                continue
            bbox = []
            for i in range(len(one_imgid_det['bbox'])):
                bbox.append(one_imgid_det['bbox'][i] * scale)
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            plt.gca().text(bbox[0], bbox[1],
                           '{:s} {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=7, color='white')

        saved_image_path = self.generate_saved_image_path(image_id)
        plt.savefig(saved_image_path)
        plt.show()

    def draw_in_all_image(self, scale=1.0, threshold=0.03, save=True, class_name=''):
        all_imgid_dets = self._dets_dataset.get_data()
        for image_id in all_imgid_dets:
            self.draw_in_one_image(
                image_id, scale, threshold, save, class_name)

    def get_abspath_from_id(self, image_id):
        return os.path.join(self._root_path,
                            self._path_template.format(self._year, image_id))
