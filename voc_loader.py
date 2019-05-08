import logging
import os

import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image

from paths import DATASETS_ROOT

log = logging.getLogger()

VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']


class VOCLoader():
    
    def __init__(self, year, split, segmentation=False, augmented_seg=False):
        assert year in ['07', '12']
        self.dataset = 'voc'
        self.year = year
        self.root = os.path.join(DATASETS_ROOT, 'VOCdevkit/VOC20%s/' % year)
        self.split = split
        assert split in ['train', 'val', 'trainval', 'test']

        cats = VOC_CATS
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]

        self.segmentation = segmentation
        self.augmented_seg = augmented_seg

        assert not self.segmentation or self.segmentation and self.year == '12'

        if self.augmented_seg:
            filelist = 'ImageSets/SegmentationAug/%s.txt'
        elif self.segmentation:
            filelist = 'ImageSets/Segmentation/%s.txt'
        else:
            filelist = 'ImageSets/Main/%s.txt'
        with open(os.path.join(self.root, filelist % self.split), 'r') as f:
            self.filenames = f.read().split('\n')[:-1]
        log.info("Created a loader VOC%s %s with %i images" % (year, split, len(self.filenames)))

    def load_image(self, name):
        im = Image.open('%sJPEGImages/%s.jpg' % (self.root, name)).convert('RGB')
        im = np.array(im) / 255.0
        im = im.astype(np.float32)
        return im

    def get_filenames(self):
        return self.filenames

    def read_annotations(self, name):
        bboxes = []
        cats = []

        tree = ET.parse('%sAnnotations/%s.xml' % (self.root, name))
        root = tree.getroot()
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        difficulty = []
        for obj in root.findall('object'):
            cat = self.cats_to_ids[obj.find('name').text]
            difficult = (int(obj.find('difficult').text) != 0)
            difficulty.append(difficult)
            cats.append(cat)
            bbox_tag = obj.find('bndbox')
            x = int(bbox_tag.find('xmin').text)
            y = int(bbox_tag.find('ymin').text)
            w = int(bbox_tag.find('xmax').text)-x
            h = int(bbox_tag.find('ymax').text)-y
            bboxes.append((x, y, w, h))

        gt_cats = np.array(cats)
        gt_bboxes = np.array(bboxes).reshape((len(bboxes), 4))
        difficulty = np.array(difficulty)

        seg_gt = self.read_segmentations(name, height, width)

        output = gt_bboxes, seg_gt, gt_cats, width, height, difficulty
        return output
    
    def read_segmentations(self, name, height, width):
        if self.segmentation:
            try:
                seg_folder = self.root + 'SegmentationClass/'
                seg_file = seg_folder + name + '.png'
                seg_map = Image.open(seg_file)
            except:
                assert self.augmented_seg
                seg_folder = self.root + 'SegmentationClassAug/'
                seg_file = seg_folder + name + '.png'
                seg_map = Image.open(seg_file)
            segmentation = np.array(seg_map, dtype=np.uint8)
        else:
            # if there is no segmentation for a particular image we fill the mask
            # with zeros to keep the same amount of tensors but don't learn from it
            segmentation = np.zeros([height, width], dtype=np.uint8) + 255
        return segmentation
    

class VOCLoaderTutor():
    
    def __init__(self, year, split, task_set):
        assert year in ['07', '12']
        self.dataset = 'voc'
        self.year = year
        self.root = os.path.join(DATASETS_ROOT, 'VOCdevkit/VOC20%s/' % year)
        self.split = split
        self.taskset = task_set
        assert split in ['train', 'val', 'trainval', 'test']
        assert task_set in ['Main', 'Segmentation', 'SegmentationAug']

        cats = VOC_CATS
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]
        
        filelist = 'ImageSets/%s/%s.txt' % (self.taskset,self.split)
        with open(os.path.join(self.root, filelist ), 'r') as f:
            self.filenames = f.read().split('\n')[:-1]
        log.info("Created a loader VOC%s %s%s with %i images" % (year, task_set,split, len(self.filenames)))

    def load_image(self, name):
        im = Image.open('%sJPEGImages/%s.jpg' % (self.root, name)).convert('RGB')
        im = np.array(im) / 255.0
        im = im.astype(np.float32)
        return im

    def get_filenames(self):
        return self.filenames

    def read_annotations(self, name):
        bboxes = []
        cats = []

        tree = ET.parse('%sAnnotations/%s.xml' % (self.root, name))
        root = tree.getroot()
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        difficulty = []
        for obj in root.findall('object'):
            cat = self.cats_to_ids[obj.find('name').text]
            difficult = (int(obj.find('difficult').text) != 0)
            difficulty.append(difficult)
            cats.append(cat)
            bbox_tag = obj.find('bndbox')
            x = int(bbox_tag.find('xmin').text)
            y = int(bbox_tag.find('ymin').text)
            w = int(bbox_tag.find('xmax').text)-x
            h = int(bbox_tag.find('ymax').text)-y
            bboxes.append((x, y, w, h))
        
        gt_cats = np.array(cats)
        gt_bboxes = np.array(bboxes).reshape((len(bboxes), 4))
        difficulty = np.array(difficulty)
        
        bboxes_tutor = []
        cats_tutor = []
        treeTutor = ET.parse('/mnt/raid/data/ni/dnn/blitzNet/models/detection/coco_2014_train_res152' + 
                             '/results/VOC2012/trainval_Seg_Det_xml/%s.xml' % name  )
        rootTutor = treeTutor.getroot()
        confidence = []
        for obj in rootTutor.findall('object'):
            cat = self.cats_to_ids[obj.find('name').text]
            cats_tutor.append(cat)
            if cat<0 or cat>20:
                print('FUCKED UP CLASS NAME')
            conf = float(obj.find('confidence').text)
            confidence.append(conf)
            bbox_tag = obj.find('bndbox')
            x = float(bbox_tag.find('xmin').text)
            y = float(bbox_tag.find('ymin').text)
            w = float(bbox_tag.find('xmax').text)-x
            h = float(bbox_tag.find('ymax').text)-y
            bboxes_tutor.append((x, y, w, h))

        tt_cats = np.array(cats_tutor)
        tt_bboxes = np.array(bboxes_tutor).reshape((len(bboxes_tutor), 4))
        confidence = np.array(confidence)

        seg_gt, has_seg_gt, seg_aug, has_seg_aug, seg_tt = self.read_segmentations2(name, height, width)

        output = gt_bboxes, gt_cats, difficulty, tt_bboxes, tt_cats, confidence, seg_gt, has_seg_gt, seg_aug, has_seg_aug, seg_tt, width, height
        return output

    def read_segmentations(self, name, height, width):
        has_seg_gt = False
        gt_file = self.root + 'SegmentationClass/' + name + '.png'
        if os.path.isfile(gt_file):
            has_seg_gt = True
            seg_gt = np.array( Image.open( gt_file ), dtype = np.uint8 )
        else:
            seg_gt = np.zeros([height, width], dtype=np.uint8) + 255
        
        has_seg_aug = False
        aug_file = self.root + 'SegmentationClassAug/' + name + '.png'
        if os.path.isfile(aug_file):
            has_seg_aug = True
            seg_aug = np.array( Image.open( aug_file ), dtype = np.uint8 )
        else:
            seg_aug = np.zeros([height, width], dtype=np.uint8) + 255
        
        tt_file = '/mnt/raid/data/ni/dnn/blitzNet/models/segmentation/mask_rcnn/results/VOC2012/trainval_Seg_Det_vol/' + name + '.npy'
        seg_tt = np.load( tt_file )
        return seg_gt, has_seg_gt, seg_aug, has_seg_aug, seg_tt
