#!/usr/bin/env python
# encoding: utf-8
import os
import cPickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
from random import random as rand
import cv2

# note that annotation here means records mapping that maps image_filename to
# objects array which in the image named image_filename.
# As for objects array,each object in it has 'name','difficult' and 'bbox' properties
# For be more quick to get annotation again,it can be saved to cache using cPickle


def parse_voc_one_rec(filename):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                            int(float(bbox.find('ymin').text)),
                            int(float(bbox.find('xmax').text)),
                            int(float(bbox.find('ymax').text))]
        objects.append(obj_dict)
    return objects


# load annotations
# imageset file contains ids of images for testing
def load_pascal_all_recs(imageset_file, annopath, annocache, display=True):
    assert os.path.isfile(imageset_file),\
        print('imageset file is not exist')

    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    if not os.path.isfile(annocache):
        gts = {}
        for ind, image_filename in enumerate(image_filenames):
            gts[image_filename] = parse_voc_one_rec(
                annopath.format(image_filename))
            if display and ind % 100 == 0:
                print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames))
        print 'saving annotations cache to {:s}'.format(annocache)
        with open(annocache, 'wb') as f:
            cPickle.dump(gts, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        # load annotations from cache
        with open(annocache, 'rb') as f:
            gts = cPickle.load(f)
    return gts


# load detections

# note that detections and collections of objects detected
# They are collected into one file by classname.
# In the file one line is filled by the information of one object
# The information here refers to image_id,object's confidence and
# its bbox

def load_one_class_detections(detpath, classname):
    # read detections
    detfile = detpath.format(classname)

    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    dets = []
    for line in splitlines:
        det = {}
        det['image_id'] = line[0]
        det['confidence'] = float(line[1])
        det['bbox'] = [float(z) for z in line[2:]]
        dets.append(det)
    return dets

#return all class names used by pascal voc
def get_pascal_voc_class_names():
    class_names = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
           'cow','diningtable','dog','horse','motorbike','person','pottedplant',
           'sheep','sofa','train','tvmonitor']
    return class_names
def get_coco_class_names():
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    return class_names

def get_class_names_by_dataset(dataset_name):
    x_class_names = eval('get_'+dataset_name+'pascal_voc_class_names')()
    return x_class_names

def load_all_classes_detections(dataset,detpath):
    x_class_names = get_class_names_by_dataset(dataset)
    all_cls_dets={}
    #for each class
    for class_name in x_class_names:
        one_cls_dets = load_one_class_detections(detpath,class_name)
        all_cls_dets[class_name] = one_cls_dets

    return all_cls_dets

# Due to inconvience to view all objects detected already,so collect
# them by image_id.
# In other word, get all detections in one image together.
def reorganize_detections(dets):
    dets_reorg_by_imgids = {}
    # imgids = []
    for det in dets:
        image_id = str(det['image_id'])
        del det['image_id']
        if image_id in dets_reorg_by_imgids:
            dets_reorg_by_imgids[image_id].append(det)
        else:
            dets_reorg_by_imgids[image_id] = []
            dets_reorg_by_imgids[image_id].append(det)
    return dets_reorg_by_imgids


# do evaluation in certain image
def evaluation_detections(gts, preds, image_id, vis, threshold=0.5):
    assert isinstance(gts, dict),\
        'ground truth should be a dict'
    assert isinstance(preds, dict),\
        'prediction should be a dict,please \
        use function to convert it'
    gt = gts[image_id]
    pred = preds[image_id]
    visited = []
    for _ in range(len(gt)):
        visited.append(False)
    pred_match_gt = []
    for _ in range(len(pred)):
        pred_match_gt.append(-1)

    pred_gt_overlaps = []

    for index_pred, object_pred in enumerate(pred):
        max_overlaps = -1
        pred_gt_overlaps[index_pred] = []
        for index_gt, object_gt in enumerate(gt['bbox']):
            overlap = bbox_iou(object_pred['bbox'], object_gt['bbox'])
            pred_gt_overlaps[index_pred].append(overlap)
            if overlap > max_overlaps:
                max_overlaps = overlap

        if max_overlaps > threshold and\
                not object_gt['difficult'] and\
                not visited[index_gt]:
            visited[index_gt] = True
            pred_match_gt[index_pred] = index_gt
        # else:
            # do something for false positive

            # display information of results

    for index_pred, object_pred in enumerate(pred):
        # exist ground truth is matched with pred
        if not pred_match_gt[index_pred] == -1:
            # true positive
            print('true positive')
            pprint.pprint(object_pred['bbox'])
            pprint.pprint(gt[pred_match_gt[index_pred]]['bbox'])
        else:
            # false positive
            print('false positive')
            pprint.pprint(object_pred['bbox'])

    for index_gt, object_gt in enumerate(gt):
        # not exist pred to match with gt
        if not visited[index_gt]:
            print('false negative')
            pprint.pprint(object_gt['bbox'])
            # try to find the nearest pred
            # TODO


def bbox_union(bbgt, bb, inters):
    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (bbgt[2] - bbgt[0] + 1.) *
           (bbgt[3] - bbgt[1] + 1.) - inters)
    return uni


# note that it is no matter with the order of arguments
def bbox_intersection(bbgt, bb):
    # intersection
    ixmin = np.maximum(bbgt[0], bb[0])
    iymin = np.maximum(bbgt[1], bb[1])
    ixmax = np.minimum(bbgt[2], bb[2])
    iymax = np.minimum(bbgt[3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    return inters


def bbox_iou(bbgt, bb):
    inters = bbox_intersection(bbgt, bb)
    uni = bbox_union(bbgt, bb, inters)
    return inters / uni


def analysis_all_detections():
    pass


def vis_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-3):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(),
                 random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def draw_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-1):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256),
                 random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def show_boxes_by_class(image_filename, pred, class_name, scale=1.0, threshold=0.03, gt=[]):
    # visualize
    im = cv2.imread(image_filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    plt.cla()
    plt.axis("off")

    plt.imshow(im)
    count_gt_per_image = 0
    for object_gt in gt:
        if object_gt['name'] == class_name:
            count_gt_per_image += 1
        else:
            continue
        det = object_gt['bbox']
        bbox = []
        for i in range(len(det)):
            bbox.append(det[i] * scale)
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
        return
    for object_pred in pred:
        score = object_pred['confidence']
        if score < threshold:
            continue
        det = object_pred['bbox']
        bbox = []
        for i in range(len(det)):
            bbox.append(det[i] * scale)
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=2.5)
        plt.gca().add_patch(rect)

        # if cls_dets.shape[1] == 5:
        plt.gca().text(bbox[0], bbox[1],
                       '{:s} {:.3f}'.format(class_name, score),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=7, color='white')

    plt.show()
    # image_filename_saved = 'det.jpg'
    # plt.savefig(image_filename_saved)
    return im


def get_abspath_from_id(image_id, datapath_prefix, year='2007'):
    image_filename = datapath_prefix.format(
        '/VOCdevkit/VOC{}/JPEGImages/{}.jpg')\
        .format(year, image_id)
    return image_filename


def show_boxes_by_class_all(preds, class_name, datapath_prefix, gts={}, scale=1.0, threshold=0.03, year='2007'):
    for image_id in preds:
        pred = preds[image_id]
        image_filename = get_abspath_from_id(image_id, datapath_prefix, year)
        gt = []
        if image_id in gts:
            gt = gts[image_id]
        show_boxes_by_class(image_filename, pred,
                            class_name, scale, threshold, gt)
