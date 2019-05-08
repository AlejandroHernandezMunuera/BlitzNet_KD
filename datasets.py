import os

import numpy as np
import tensorflow as tf

from paths import DATASETS_ROOT
from voc_loader import VOC_CATS, VOCLoader, VOCLoaderTutor
#from coco_loader import COCOLoader, COCO_CATS

from math import ceil

slim = tf.contrib.slim


def normalize_bboxes(bboxes, w, h):
    """rescales bboxes to [0, 1]"""
    new_bboxes = np.array(bboxes, dtype=np.float32)
    new_bboxes[:, [0, 2]] /= w
    new_bboxes[:, [1, 3]] /= h
    return new_bboxes


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    """Wrapper for inserting float64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, bboxes, cats, difficulty, segmentation, height, width):
    xmin = bboxes[:, 0].tolist()
    ymin = bboxes[:, 1].tolist()
    xmax = (bboxes[:, 2] + bboxes[:, 0]).tolist()
    ymax = (bboxes[:, 3] + bboxes[:, 1]).tolist()
    labels = cats.tolist()
    image_format = 'JPEG'
    segmentation_format = 'PNG'
    difficulty = difficulty.tolist()

    # FIXME empty segmentation if passed as argument?
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/object/bbox/ymin': _float64_feature(ymin),
        'image/object/bbox/xmin': _float64_feature(xmin),
        'image/object/bbox/ymax': _float64_feature(ymax),
        'image/object/bbox/xmax': _float64_feature(xmax),
        'image/object/class/label': _int64_feature(labels),
        'image/object/difficulty': _int64_feature(difficulty),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer)),
        'image/segmentation/format': _bytes_feature(tf.compat.as_bytes(segmentation_format)),
        'image/segmentation/encoded': _bytes_feature(tf.compat.as_bytes(segmentation)),
    }))
    return example

def _convert_to_example_tutor(filename, image_buffer, gt_bboxes, gt_cats, difficulty, tt_bboxes, tt_cats, confidence, segmentation_gt, has_seg_gt, segmentation_aug, has_seg_aug, segmentation_tt, height, width):
                
    xmin_gt = gt_bboxes[:, 0].tolist()
    ymin_gt = gt_bboxes[:, 1].tolist()
    xmax_gt = (gt_bboxes[:, 2] + gt_bboxes[:, 0]).tolist()
    ymax_gt = (gt_bboxes[:, 3] + gt_bboxes[:, 1]).tolist()
    labels_gt = gt_cats.tolist()
    difficulty = difficulty.tolist()
    
    xmin_tt = tt_bboxes[:, 0].tolist()
    ymin_tt = tt_bboxes[:, 1].tolist()
    xmax_tt = (tt_bboxes[:, 2] + tt_bboxes[:, 0]).tolist()
    ymax_tt = (tt_bboxes[:, 3] + tt_bboxes[:, 1]).tolist()
    labels_tt = tt_cats.tolist()
    confidence = confidence.tolist()
    
    has_seg_gt = int(has_seg_gt)
    has_seg_aug = int(has_seg_aug)
    segmentation_tt = segmentation_tt.reshape(-1).tolist()
    
    image_format = 'JPEG'
    segmentation_format = 'PNG'

    # FIXME empty segmentation if passed as argument?
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        
        'image/object_gt/bbox/ymin': _float64_feature(ymin_gt),
        'image/object_gt/bbox/xmin': _float64_feature(xmin_gt),
        'image/object_gt/bbox/ymax': _float64_feature(ymax_gt),
        'image/object_gt/bbox/xmax': _float64_feature(xmax_gt),
        'image/object_gt/class/label': _int64_feature(labels_gt),
        'image/object_gt/difficulty': _int64_feature(difficulty),
        
        'image/object_tt/bbox/ymin': _float64_feature(ymin_tt),
        'image/object_tt/bbox/xmin': _float64_feature(xmin_tt),
        'image/object_tt/bbox/ymax': _float64_feature(ymax_tt),
        'image/object_tt/bbox/xmax': _float64_feature(xmax_tt),
        'image/object_tt/class/label': _int64_feature(labels_tt),
        'image/object_tt/confidence': _float64_feature(confidence),
        
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer)),
        
        'image/segmentation_gt/format': _bytes_feature(tf.compat.as_bytes(segmentation_format)),
        'image/segmentation_gt/encoded': _bytes_feature(tf.compat.as_bytes(segmentation_gt)),
        'image/segmentation_gt/has_gt': _int64_feature(has_seg_gt),
        
        'image/segmentation_aug/format': _bytes_feature(tf.compat.as_bytes(segmentation_format)),
        'image/segmentation_aug/encoded': _bytes_feature(tf.compat.as_bytes(segmentation_aug)),
        'image/segmentation_aug/has_gt': _int64_feature(has_seg_aug),
        
        'image/segmentation_tt/encoded': _float64_feature(segmentation_tt),
    }))
    return example

splits_to_sizes = {
    'voc07_test': 4952,
    'voc07_trainval': 5011,
    'voc07-trainval-segmentation': 5011,
    'voc12_train': 5717,
    'voc12_val': 5823,
    'voc12-train-segmentation': 10582,
    'voc12-train-segmentation-original': 1464,
    'voc12-val-segmentation': 1449,
    'voc12-val': 5823,
    'coco-train2014-*': 82783,
    'coco-valminusminival2014-*': 35504,
    'coco-minival2014': 5000,
    'coco-seg-train2014-*': 82783,
    'coco-seg-valminusminival2014-*': 35504,
    'coco-seg-minival2014': 5000,
}

splits_to_sizes_tutor = {
    'voc12-Main-train': 5717,
    'voc12-Main-val': 5823,
    'voc12-Main-trainval': 11540,
    'voc12-Segmentation-train': 1464,
    'voc12-Segmentation-val': 1449,
    'voc12-Segmentation-trainval': 2913,
    'voc12-SegmentationAug-train': 10582,
}


def get_dataset(*files):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='JPEG'),
        'image/segmentation/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/segmentation/format': tf.FixedLenFeature(
            (), tf.string, default_value='RAW'),
        'image/object/bbox/xmin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(
            dtype=tf.int64),
        'image/object/difficulty': tf.VarLenFeature(
            dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
        'image/segmentation': slim.tfexample_decoder.Image('image/segmentation/encoded', 'image/segmentation/format', channels=1),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
        'object/difficulty': slim.tfexample_decoder.Tensor('image/object/difficulty'),
    }

    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'image/segmentation': 'A semantic segmentation.',
        'object/bbox': 'A list of bounding boxes.',
        'object/label': 'A list of labels, one per each object.',
        'object/difficulty': 'A list of binary difficulty flags, one per each object.',
    }

    is_coco = all('coco' in f for f in files)
    is_voc = all('voc' in f for f in files)
    if not (is_coco ^ is_voc):
        raise ValueError("It is a bad idea to mix in one dataset VOC and COCO files")
    if is_coco:
        categories = COCO_CATS
    if is_voc:
        categories = VOC_CATS

    return slim.dataset.Dataset(
        data_sources=[os.path.join(DATASETS_ROOT, f) for f in files],
        reader=tf.TFRecordReader,
        decoder=slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers),
        num_samples=sum(splits_to_sizes[s] for s in files),
        items_to_descriptions=items_to_descriptions,
        num_classes=len(categories),
        labels_to_names={i: cat for i, cat in enumerate(categories)})


def get_dataset_tutor(*files):
    keys_to_features = {
        'image/height': tf.VarLenFeature(
            dtype=tf.int64),
        'image/width': tf.VarLenFeature(
            dtype=tf.int64),
        
        'image/object_gt/bbox/xmin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object_gt/bbox/ymin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object_gt/bbox/xmax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object_gt/bbox/ymax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object_gt/class/label': tf.VarLenFeature(
            dtype=tf.int64),
        'image/object_gt/difficulty': tf.VarLenFeature(
            dtype=tf.int64),
        
        'image/object_tt/bbox/xmin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object_tt/bbox/ymin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object_tt/bbox/xmax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object_tt/bbox/ymax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object_tt/class/label': tf.VarLenFeature(
            dtype=tf.int64),
        'image/object_tt/confidence': tf.VarLenFeature(
            dtype=tf.float32),
        
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='JPEG'),
        
        'image/segmentation_gt/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/segmentation_gt/format': tf.FixedLenFeature(
            (), tf.string, default_value='RAW'),
        'image/segmentation_gt/has_gt': tf.VarLenFeature(
            dtype=tf.int64),
        
        'image/segmentation_aug/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/segmentation_aug/format': tf.FixedLenFeature(
            (), tf.string, default_value='RAW'),
        'image/segmentation_aug/has_gt': tf.VarLenFeature(
            dtype=tf.int64),
        
        'image/segmentation_tt/encoded': tf.VarLenFeature(
            dtype=tf.float32),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3, dct_method="INTEGER_ACCURATE"),
        
        'image/filename': slim.tfexample_decoder.Tensor('image/filename'),
        'image/height': slim.tfexample_decoder.Tensor('image/height'),
        'image/width': slim.tfexample_decoder.Tensor('image/width'),
        
        'image/segmentation_gt': slim.tfexample_decoder.Image('image/segmentation_gt/encoded', 'image/segmentation_gt/format',
                                                              channels=1, dct_method="INTEGER_ACCURATE"),
        'image/has_gt': slim.tfexample_decoder.Tensor('image/segmentation_gt/has_gt'),
        
        'image/segmentation_aug': slim.tfexample_decoder.Image('image/segmentation_aug/encoded', 'image/segmentation_aug/format',
                                                               channels=1, dct_method="INTEGER_ACCURATE"),
        'image/has_aug': slim.tfexample_decoder.Tensor('image/segmentation_aug/has_gt'),
        
        'image/segmentation_tt': slim.tfexample_decoder.Tensor('image/segmentation_tt/encoded'),
        
        'object_gt/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object_gt/bbox/'),
        'object_gt/label': slim.tfexample_decoder.Tensor('image/object_gt/class/label'),
        'object_gt/difficulty': slim.tfexample_decoder.Tensor('image/object_gt/difficulty'),
        
        'object_tt/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object_tt/bbox/'), 
        'object_tt/label': slim.tfexample_decoder.Tensor('image/object_tt/class/label'),
        'object_tt/confidence': slim.tfexample_decoder.Tensor('image/object_tt/confidence'),
    }

    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        
        'image/filename': 'image file number',
        'image/height': 'image height',
        'image/width': 'image width',
        
        'image/segmentation_gt': 'A semantic segmentation(ground truth).',
        'image/has_gt': 'if it has segmentation ground truth',
        
        'image/segmentation_aug': 'A semantic segmentation(augmented dataset).',
        'image/has_aug': 'if it has segmentation augmented ground truth',
        
        'image/segmentation_tt': 'A semantic segmentation(np tutor output).',
        
        'object_gt/bbox': 'A list of bounding boxes(ground truth).',
        'object_gt/label': 'A list of labels, one per each object(ground truth).',
        'object_gt/difficulty': 'A list of difficulties, one per each object.',
        
        'object_tt/bbox': 'A list of bounding boxes(tutor).',
        'object_tt/label': 'A list of labels, one per each object(tutor).',
        'object_tt/confidence': 'A list of confidences, one per each object.',
    }

    is_coco = all('coco' in f for f in files)
    is_voc = all('voc' in f for f in files)
    if not (is_coco ^ is_voc):
        raise ValueError("It is a bad idea to mix in one dataset VOC and COCO files")
    if is_coco:
        categories = COCO_CATS
    if is_voc:
        categories = VOC_CATS

    return slim.dataset.Dataset(
        data_sources=[os.path.join(DATASETS_ROOT, f) for f in files],
        reader=tf.TFRecordReader,
        decoder=slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers),
        num_samples=sum(splits_to_sizes_tutor[s] for s in files),
        items_to_descriptions=items_to_descriptions,
        num_classes=len(categories),
        labels_to_names={i: cat for i, cat in enumerate(categories)})


def create_coco_dataset(split):
    loader = COCOLoader(split)
    sz = len(loader.get_filenames())
    print("Contains %i files" % sz)

    if split in ['minival2014', 'valminusminival2014']:
        realsplit = 'val2014'
    else:
        realsplit = split

    shard_size = 5000
    num_shards = ceil(sz/shard_size)
    print("So we decided to split it in %i shards" % num_shards)
    image_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_png(tf.expand_dims(image_placeholder, 2))
    with tf.Session('') as sess:
        for shard in range(num_shards):
            print("Shard %i/%i is starting" % (shard, num_shards))
            output_file = os.path.join(DATASETS_ROOT, 'coco-seg-%s-%.5d-of-%.5d' % (split, shard, num_shards))
            writer = tf.python_io.TFRecordWriter(output_file)

            for i in range(shard*shard_size, min(sz, (shard+1)*shard_size)):
                f = loader.get_filenames()[i]
                img = loader.coco.loadImgs(f)[0]
                path = '%simages/%s/%s' % (loader.root, realsplit, img['file_name'])
                with tf.gfile.FastGFile(path, 'rb') as ff:
                    image_data = ff.read()
                gt_bb, gt_cats, w, h, diff = loader.read_annotations(f)
                gt_bb = normalize_bboxes(gt_bb, w, h)

                segmentation = np.zeros((h, w), dtype=np.uint8)
                coco_anns = loader._get_coco_annotations(f, only_instances=False)
                for ann in coco_anns:
                    mask = loader._read_segmentation(ann, h, w)
                    cid = loader.coco_ids_to_internal[ann['category_id']]
                    assert mask.shape == segmentation.shape
                    segmentation[mask > 0] = cid

                png_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: segmentation})
                example = _convert_to_example(path, image_data, gt_bb, gt_cats, diff, png_string, h, w)
                if i % 100 == 0:
                    print("%i files are processed" % i)
                writer.write(example.SerializeToString())

            writer.close()
    print("Done")


def create_voc_dataset(year, split, segmentation=False, augmented_seg=False):
    """packs a dataset to a protobuf file
    Args:
        year: the year of voc dataset. choice=['07', '12']
        split: split of data, choice=['train', 'val', 'trainval', 'test']
        segmentation: if True, segmentation annotations are encoded
        augmented_seg: if True, encodes extra annotations
        """
    assert not ((year=='07' or split=='val') and augmented_seg==True), \
        'There is no extra segmentation masks for VOC07 or VOC12-val'

    loader = VOCLoader(year, split, segmentation=segmentation, augmented_seg=augmented_seg)
    print("Contains %i files" % len(loader.get_filenames()))
    output_file = os.path.join(DATASETS_ROOT, 'voc%s-%s%s' %
                               (year, split, '-segmentation' * segmentation))
    image_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_png(tf.expand_dims(image_placeholder, 2))
    writer = tf.python_io.TFRecordWriter(output_file)
    with tf.Session('') as sess:
        for i, f in enumerate(loader.get_filenames()):
            path = '%sJPEGImages/%s.jpg' % (loader.root, f)
            with tf.gfile.FastGFile(path, 'rb') as ff:
                image_data = ff.read()
            gt_bb, segmentation, gt_cats, w, h, diff = loader.read_annotations(f)
            gt_bb = normalize_bboxes(gt_bb, w, h)
            png_string = sess.run(encoded_image,
                                  feed_dict={image_placeholder: segmentation})
            example = _convert_to_example(path, image_data, gt_bb, gt_cats,
                                          diff, png_string, h, w)
            if i % 100 == 0:
                print("%i files are processed" % i)
            writer.write(example.SerializeToString())

        writer.close()
    print("Done")
    
def create_voc_dataset_tutor(year, split, task_set):
    """packs a dataset to a protobuf file
    Args:
        year: the year of voc dataset. choice=['07', '12']
        split: split of data, choice=['train', 'val', 'trainval', 'test']
        task_set: task to read its datasets, choice=['Main', 'Segmentation', 'SegmentationAug']
        """
    loader = VOCLoaderTutor(year, split, task_set)
    print("Contains %i files" % len(loader.get_filenames()))
    output_file = os.path.join(DATASETS_ROOT, 'voc%s-%s-%s' %
                               (year, task_set, split ))
    
    image_placeholder1 = tf.placeholder(dtype=tf.uint8)
    encoded_image1 = tf.image.encode_png(tf.expand_dims(image_placeholder1, 2))
    
    image_placeholder2 = tf.placeholder(dtype=tf.uint8)
    encoded_image2 = tf.image.encode_png(tf.expand_dims(image_placeholder2, 2))
    
    writer = tf.python_io.TFRecordWriter(output_file)
    with tf.Session('') as sess:
        for i, f in enumerate(loader.get_filenames()):
            path = '%sJPEGImages/%s.jpg' % (loader.root, f)
            with tf.gfile.FastGFile(path, 'rb') as ff:
                image_data = ff.read()
            
            gt_bb, gt_cats, diff,tt_bb, tt_cats, conf, seg_gt, has_seg_gt, seg_aug, has_seg_aug, seg_tt, width, height = loader.read_annotations(f)
            gt_bb = normalize_bboxes(gt_bb, width, height)
            tt_bb = normalize_bboxes(tt_bb, width, height)
            
            png_string_gt, png_string_aug = sess.run( [encoded_image1, encoded_image2],
                                  feed_dict={image_placeholder1: seg_gt, image_placeholder2: seg_aug})
            
            
            example = _convert_to_example_tutor(path, image_data, gt_bb, gt_cats, diff, tt_bb, tt_cats, conf, 
                                          png_string_gt, has_seg_gt, png_string_aug, has_seg_aug, seg_tt, height, width)
            
            if i % 100 == 0:
                print("%i files are processed" % i)
                
            writer.write(example.SerializeToString())

        writer.close()
    print("Done")


if __name__ == '__main__':
    
    create_voc_dataset_tutor('12', 'train', 'Main' )
    create_voc_dataset_tutor('12', 'val', 'Main' )
    create_voc_dataset_tutor('12', 'trainval', 'Main' )
    #TODO: allow GT missing
    #create_voc_dataset_tutor('12', 'test', 'Main' )
    
    create_voc_dataset_tutor('12', 'train', 'Segmentation' )
    create_voc_dataset_tutor('12', 'val', 'Segmentation' )
    create_voc_dataset_tutor('12', 'trainval', 'Segmentation' )
    #TODO: allow GT missing
    #create_voc_dataset_tutor('12', 'test', 'Segmentation' )
    
    create_voc_dataset_tutor('12', 'train', 'SegmentationAug' )

    '''
    create_voc_dataset('07', 'test')
    create_voc_dataset('07', 'trainval')
    create_voc_dataset('12', 'train', True, True)
    create_voc_dataset('12', 'trainval', True)
    create_voc_dataset('12', 'val', True)
    
    create_coco_dataset('val2014')
    create_coco_dataset('valminusminival2014')
    create_coco_dataset('minival2014')
    create_coco_dataset('train2014')
    '''
