#!/usr/bin/env python
# encoding: utf-8
import os
import cPickle


class Dataset(object):
    _class_names = []

    def __init__(self, name, data_type, cache_path, cache_name):
        self._name = name
        self._data_type = data_type
        self.deter_class_names()
        self._data = None
        self._cache_path = cache_path
        self._cache_name = cache_name
        self._saved = False

    def get_class_names(self):
        return self._class_names

    def deter_class_names(self):
        if self._data_type == 'voc':
            self._class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor']
        else:
            self._class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                                 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                                 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                                 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                                 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def get_data(self, image_id=None):
        if image_id is None:

            return self._data
        else:
            return self._data[image_id]

    def get_image_ids(self, num=-1):
        if len(self._data.keys()) >= num:
            return self._data.keys()[:num]

    def get_image_gts(self, num=-1):
        image_ids = self.get_image_ids(num)
        image_gts = {}
        for image_id in image_ids:
            image_gts[image_id] = self.get_data(image_id)
        return image_gts

    def save_data(self):
        if self._saved:
            print 'already saved'
            return
        cache_path_to_file = os.path.join(self._cache_path, self._cache_name)
        with open(cache_path_to_file, 'wb') as f:
            cPickle.dump(self._data, f, cPickle.HIGHEST_PROTOCOL)
            # print 'save data to ',cache_path_to_file
        self._saved = True


class GTs_Dataset(Dataset):

    def __init__(self, name, root_path, cache_path, data_type='voc',
                 path_template='VOCdevkit/VOC{}/Annotations/{}.xml', year='',
                 save_format='image_first'):
        super(GTs_Dataset, self).__init__(
            name, data_type, cache_path, cache_name='recs')
        # self._name = name
        self._root_path = root_path
        self._path_template = path_template
        self._year = year
        # self._cache_path = cache_path
        self._save_format = save_format
        self.parse_all_recs()

    def get_abspath_from_id(self, image_id):
        image_filename = os.path.join(self._root_path, self._path_template
                                      .format(self._year, image_id))

        return image_filename

    def parse_one_rec(self):
        pass

    def parse_all_recs(self):
        pass


class Dets_Dataset(Dataset):
    def __init__(self, name, root_path, cache_path, data_type='voc', path_template='results/VOC{}/Main/comp4_det_test_{}.txt',
                 year='', save_format='image_first'):
        Dataset.__init__(self, name, data_type, cache_path, cache_name='dets')
        self._root_path = root_path
        self._path_template = path_template
        self._year = year
        self._data = None
        self._save_format = save_format
        self.load_data()
        # self.save_data()

    def load_one_class_detections(self, class_name, display=False):
        # read detections
        one_class_dets_file = os.path.join(self._root_path, self._path_template.format(
            self._year, class_name))

        with open(one_class_dets_file, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        length = len(splitlines)
        one_class_dets = []
        for index, line in enumerate(splitlines):
            det = {}
            det['image_id'] = line[0]
            det['confidence'] = float(line[1])
            det['bbox'] = [float(z) for z in line[2:]]
            det['class'] = class_name
            one_class_dets.append(det)
            if index % 10000 == 0 and display:
                print 'load {:d}/{:d} in {:s} class'\
                    .format(index + 1, length, class_name)
        return one_class_dets

    def load_all_classes_detections(self, display=True):
        x_class_names = self.get_class_names()
        all_cls_dets = {}
        # for each class
        length = len(x_class_names)
        for class_index, class_name in enumerate(x_class_names):
            one_cls_dets = self.load_one_class_detections(class_name)
            all_cls_dets[class_name] = one_cls_dets
            if display:
                print 'load {:s} class dets done({:d}/{:d})'.format(class_name, class_index + 1, length)

        self._data = all_cls_dets

    def convert_data_format_to_image_first(self):
        all_imgid_dets = {}
        all_cls_dets = self._data
        for class_name in all_cls_dets:
            one_cls_dets = all_cls_dets[class_name]
            for det in one_cls_dets:
                image_id = str(det['image_id'])
                del det['image_id']
                if image_id in all_imgid_dets:
                    all_imgid_dets[image_id].append(det)
                else:
                    all_imgid_dets[image_id] = [det]
        self._data = all_imgid_dets

    def convert_data_format(self):
        if self._save_format == 'image_first':
            self.convert_data_format_to_image_first()
        elif self._save_format == 'class_first':
            print 'already done'

    def load_data(self):
        self.load_all_classes_detections()
        self.convert_data_format()


class PASCAL_VOC_GTs_Dataset(GTs_Dataset):

    def parse_one_rec(self, image_id):
        """
        parse pascal voc record into a dictionary
        :param filename: xml file path
        :return: list of dict
        """
        filename = self.get_abspath_from_id(image_id)
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_dict = dict()
            obj_dict['class'] = obj.find('name').text
            obj_dict['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                                int(float(bbox.find('ymin').text)),
                                int(float(bbox.find('xmax').text)),
                                int(float(bbox.find('ymax').text))]
            objects.append(obj_dict)
        return objects

    def parse_all_recs(self, display=True):
        path_to_imageset_file = os.path.join(self._root_path, 'VOCdevkit/VOC{}/ImageSets/Main/test.txt'
                                             .format(self._year))
        assert os.path.isfile(path_to_imageset_file),\
            'imageset file is not exist'
        cache_name = self._cache_name
        cache_path_to_file = os.path.join(self._cache_path, cache_name)
        with open(path_to_imageset_file, 'r') as f:
            lines = f.readlines()
        image_ids = [x.strip() for x in lines]

        if not os.path.isfile(cache_path_to_file):
            gts = {}
            for ind, image_id in enumerate(image_ids):
                gts[image_id] = self.parse_one_rec(image_id)
                if display and ind % 100 == 0:
                    print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(image_ids))
            print 'saving annotations cache to {:s}'.format(cache_path_to_file)
            with open(cache_path_to_file, 'wb') as f:
                cPickle.dump(gts, f, protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            # load annotations from cache
            self._saved = True
            with open(cache_path_to_file, 'rb') as f:
                gts = cPickle.load(f)
        self._data = gts
        # return gts


# class PASCAL_VOC_Dets_Dataset(Dets_Dataset):
    # # _class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        # # 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        # # 'sheep', 'sofa', 'train', 'tvmonitor']
    # pass
