#!/usr/bin/env python3

from config import get_logging_config, args, train_dir
from config import config as net_config

import time
import os
import sys
import socket
import logging
import logging.config
import subprocess

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')

from vgg import VGG
from resnet import ResNet
from utils import print_variables
from utils_tf import yxyx_to_xywh, data_augmentation, data_augmentation_tutor
from datasets import get_dataset, get_dataset_tutor
from boxer import PriorBoxGrid

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou

logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger()

TUTOR_DISTRIBUTION = 1.

DET_TUTOR = True
FILTER_LOW_CONF_BOXES = True
CONF_THRESHOLD = 0.001

SEG_TUTOR = False 
TEMPERATURE = 0.4   

def smooth_l1(x, y):
    abs_diff = tf.abs(x-y)
    return tf.reduce_sum(tf.where(abs_diff < 1,
                                  0.5*abs_diff*abs_diff,
                                  abs_diff - 0.5),
                         1)

def segmentation_loss(seg_logits, seg_gt, config, dataset):
    mask = seg_gt <= dataset.num_classes
    seg_logits = tf.boolean_mask(seg_logits, mask)
    seg_gt = tf.boolean_mask(seg_gt, mask)
    seg_predictions = tf.argmax(seg_logits, axis=1)
    
    seg_loss_local = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_logits,
                                                                    labels=seg_gt)
    seg_loss = tf.reduce_mean(seg_loss_local)
    tf.summary.scalar('loss/segmentation', seg_loss)
    
    mean_iou, update_mean_iou = streaming_mean_iou(seg_predictions, seg_gt,
                                                   dataset.num_classes)
    tf.summary.scalar('accuracy/mean_iou', mean_iou)
    return seg_loss, mean_iou, update_mean_iou

def segmentation_loss_tutor(seg_logits, seg_gt, config, dataset):
    softmax = tf.exp(tf.div(seg_gt, tf.constant(TEMPERATURE,dtype=tf.float32)) ) / tf.expand_dims(tf.reduce_sum(tf.exp(tf.div(seg_gt, tf.constant(TEMPERATURE,dtype=tf.float32))), axis=-1), -1)
    seg_gt_pred = tf.argmax(seg_gt, axis=-1)
    seg_predictions = tf.argmax(seg_logits, axis=-1)
    
    seg_loss_local = tf.nn.softmax_cross_entropy_with_logits_v2( labels = softmax, logits= seg_logits)
    
    seg_loss = tf.reduce_mean(seg_loss_local)
    tf.summary.scalar('loss_tutor/segmentation', seg_loss)
    
    mean_iou, update_mean_iou = streaming_mean_iou(seg_predictions, seg_gt_pred,
                                                       dataset.num_classes)
    tf.summary.scalar('accuracy_tutor/mean_iou', mean_iou)
    return seg_loss, mean_iou, update_mean_iou

def detection_loss(location, confidence, refine_ph, classes_ph, pos_mask):
    neg_mask = tf.logical_not(pos_mask)
    number_of_positives = tf.reduce_sum(tf.to_int32(pos_mask))
    true_number_of_negatives = tf.minimum(3 * number_of_positives,
                                          tf.shape(pos_mask)[1] - number_of_positives)
    # max is to avoid the case where no positive boxes were sampled
    number_of_negatives = tf.maximum(1, true_number_of_negatives)
    num_pos_float = tf.to_float(tf.maximum(1, number_of_positives))
    normalizer = tf.to_float(tf.add(number_of_positives, number_of_negatives))
    #tf.summary.scalar('batch/size', normalizer)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=confidence,
                                                                   labels=classes_ph)
    pos_class_loss = tf.reduce_sum(tf.boolean_mask(cross_entropy, pos_mask))
    #tf.summary.scalar('loss/class_pos', pos_class_loss / num_pos_float)

    top_k_worst, top_k_inds = tf.nn.top_k(tf.boolean_mask(cross_entropy, neg_mask),
                                          number_of_negatives)
    # multiplication is to avoid the case where no positive boxes were sampled
    neg_class_loss = tf.reduce_sum(top_k_worst) * \
                     tf.cast(tf.greater(true_number_of_negatives, 0), tf.float32)
    class_loss = (neg_class_loss + pos_class_loss) / num_pos_float
    #tf.summary.scalar('loss/class_neg', neg_class_loss / tf.to_float(number_of_negatives))
    #tf.summary.scalar('loss/class', class_loss)

    # cond is to avoid the case where no positive boxes were sampled
    bbox_loss = tf.cond(tf.equal(tf.reduce_sum(tf.cast(pos_mask, tf.int32)), 0),
                        lambda: 0.0,
                        lambda: tf.reduce_mean(smooth_l1(tf.boolean_mask(location, pos_mask),
                                                         tf.boolean_mask(refine_ph, pos_mask))))
    #tf.summary.scalar('loss/bbox', bbox_loss)

    inferred_class = tf.cast(tf.argmax(confidence, 2), tf.int32)
    positive_matches = tf.equal(tf.boolean_mask(inferred_class, pos_mask),
                                tf.boolean_mask(classes_ph, pos_mask))
    hard_matches = tf.equal(tf.boolean_mask(inferred_class, neg_mask),
                            tf.boolean_mask(classes_ph, neg_mask))
    hard_matches = tf.gather(hard_matches, top_k_inds)
    train_acc = ((tf.reduce_sum(tf.to_float(positive_matches)) +
                tf.reduce_sum(tf.to_float(hard_matches))) / normalizer)
    #tf.summary.scalar('accuracy/train', train_acc)

    recognized_class = tf.argmax(confidence, 2)
    tp = tf.reduce_sum(tf.to_float(tf.logical_and(recognized_class > 0, pos_mask)))
    fp = tf.reduce_sum(tf.to_float(tf.logical_and(recognized_class > 0, neg_mask)))
    fn = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(recognized_class, 0), pos_mask)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*(precision * recall)/(precision + recall)
    #tf.summary.scalar('metrics/train/precision', precision)
    #tf.summary.scalar('metrics/train/recall', recall)
    #tf.summary.scalar('metrics/train/f1', f1)
    return class_loss, bbox_loss, train_acc, number_of_positives

def detection_loss_tutor(location, confidence, refine_ph, classes_ph, pos_mask, conf_ph):
    neg_mask = tf.logical_not(pos_mask)
    number_of_positives = tf.reduce_sum(tf.to_int32(pos_mask))
    true_number_of_negatives = tf.minimum(3 * number_of_positives,
                                          tf.shape(pos_mask)[1] - number_of_positives)
    # max is to avoid the case where no positive boxes were sampled
    number_of_negatives = tf.maximum(1, true_number_of_negatives)
    num_pos_float = tf.to_float(tf.maximum(1, number_of_positives))
    normalizer = tf.to_float(tf.add(number_of_positives, number_of_negatives))
    #tf.summary.scalar('batch/size', normalizer)

    cross_entropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=confidence,
                                                                   labels=classes_ph), conf_ph )
    pos_class_loss = tf.reduce_sum(tf.boolean_mask(cross_entropy, pos_mask))
    #tf.summary.scalar('loss_tutor/class_pos', pos_class_loss / num_pos_float)

    top_k_worst, top_k_inds = tf.nn.top_k(tf.boolean_mask(cross_entropy, neg_mask),
                                          number_of_negatives)
    # multiplication is to avoid the case where no positive boxes were sampled
    neg_class_loss = tf.reduce_sum(top_k_worst) * \
                     tf.cast(tf.greater(true_number_of_negatives, 0), tf.float32)
    class_loss = (neg_class_loss + pos_class_loss) / num_pos_float
    #tf.summary.scalar('loss_tutor/class_neg', neg_class_loss / tf.to_float(number_of_negatives))
    #tf.summary.scalar('loss_tutor/class', class_loss)

    # cond is to avoid the case where no positive boxes were sampled
    bbox_loss = tf.cond(tf.equal(tf.reduce_sum(tf.cast(pos_mask, tf.int32)), 0),
                        lambda: 0.0,
                        lambda: tf.reduce_mean( tf.multiply( smooth_l1( tf.boolean_mask( location, pos_mask),
                                                         tf.boolean_mask(refine_ph, pos_mask)),
                                                            tf.boolean_mask(conf_ph, pos_mask) )))
    #tf.summary.scalar('loss_tutor/bbox', bbox_loss)

    inferred_class = tf.cast(tf.argmax(confidence, 2), tf.int32)
    positive_matches = tf.equal(tf.boolean_mask(inferred_class, pos_mask),
                                tf.boolean_mask(classes_ph, pos_mask))
    hard_matches = tf.equal(tf.boolean_mask(inferred_class, neg_mask),
                            tf.boolean_mask(classes_ph, neg_mask))
    hard_matches = tf.gather(hard_matches, top_k_inds)
    train_acc = ((tf.reduce_sum(tf.to_float(positive_matches)) +
                tf.reduce_sum(tf.to_float(hard_matches))) / normalizer)
    #tf.summary.scalar('accuracy_tutor/train', train_acc)

    recognized_class = tf.argmax(confidence, 2)
    tp = tf.reduce_sum(tf.to_float(tf.logical_and(recognized_class > 0, pos_mask)))
    fp = tf.reduce_sum(tf.to_float(tf.logical_and(recognized_class > 0, neg_mask)))
    fn = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(recognized_class, 0), pos_mask)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*(precision * recall)/(precision + recall)
    #tf.summary.scalar('metrics_tutor/train/precision', precision)
    #tf.summary.scalar('metrics_tutor/train/recall', recall)
    #tf.summary.scalar('metrics_tutor/train/f1', f1)
    return class_loss, bbox_loss, train_acc, number_of_positives

def objective_tutor_distributed(location, confidence, refine_ph, classes_ph,
              pos_mask, tt_refine_ph, tt_classes_ph, tt_pos_mask, tt_conf_ph,
              seg_logits, seg_gt, seg_tt, tt_positive, dataset, config):
                              
    the_loss = 0
    train_acc = tf.constant(1)
    mean_iou = tf.constant(1)
    update_mean_iou = tf.constant(1)
    tt_rate = tf.cast(tf.reduce_sum(tt_positive)/float(args.batch_size), tf.float32)
                                 
    if args.segment and SEG_TUTOR:
        tt_positive = tf.cast(tt_positive,tf.bool)
        gt_positive = tf.logical_not(tt_positive)
        
        tt_logits = tf.boolean_mask(seg_logits, tt_positive)
        gt_logits = tf.boolean_mask(seg_logits, gt_positive)
        
        tt_seg = tf.boolean_mask(seg_tt, tt_positive)
        gt_seg = tf.boolean_mask(seg_gt, gt_positive)
        
        seg_loss_gt, mean_iou_gt, update_mean_iou_gt = segmentation_loss(gt_logits, gt_seg, config, dataset)
        seg_loss_gt = tf.where(tf.is_nan(seg_loss_gt), tf.zeros_like(seg_loss_gt), seg_loss_gt)
        mean_iou_gt = tf.where(tf.is_nan(mean_iou_gt), tf.zeros_like(mean_iou_gt), mean_iou_gt)
        update_mean_iou_gt = tf.where(tf.is_nan(update_mean_iou_gt), tf.zeros_like(update_mean_iou_gt), update_mean_iou_gt)
        
        seg_loss_tt, mean_iou_tt, update_mean_iou_tt = segmentation_loss_tutor(tt_logits, tt_seg, config, dataset)
        seg_loss_tt = tf.where(tf.is_nan(seg_loss_tt), tf.zeros_like(seg_loss_tt), seg_loss_tt)
        mean_iou_tt = tf.where(tf.is_nan(mean_iou_tt), tf.zeros_like(mean_iou_tt), mean_iou_tt)
        update_mean_iou_tt = tf.where(tf.is_nan(update_mean_iou_tt), tf.zeros_like(update_mean_iou_tt), update_mean_iou_tt)
        
        seg_loss = ( ( 1. - tt_rate ) * tf.cast(seg_loss_gt, tf.float32) ) + ( ( tt_rate ) * tf.cast(seg_loss_tt, tf.float32) )
        mean_iou = ( ( 1. - tt_rate ) * tf.cast(mean_iou_gt, tf.float32) ) + ( ( tt_rate ) * tf.cast(mean_iou_tt, tf.float32) )
        update_mean_iou = ( ( 1. - tt_rate ) * tf.cast(update_mean_iou_gt, tf.float32) ) + ( ( tt_rate ) * tf.cast(update_mean_iou_tt,tf.float32) )
        
        the_loss += seg_loss
        
    elif args.segment and not SEG_TUTOR:
        seg_loss, mean_iou, update_mean_iou = segmentation_loss(seg_logits, seg_gt, config, dataset)
        the_loss += seg_loss

    if args.detect and DET_TUTOR:
        tt_positive = tf.cast(tt_positive,tf.bool)
        gt_positive = tf.logical_not(tt_positive)
        
        tt_location = tf.boolean_mask(location, tt_positive)
        gt_location = tf.boolean_mask(location, gt_positive)
        
        tt_confidence = tf.boolean_mask(confidence, tt_positive)
        gt_confidence = tf.boolean_mask(confidence, gt_positive)
        
        refine_ph_tt = tf.boolean_mask(tt_refine_ph, tt_positive)
        refine_ph_gt = tf.boolean_mask(refine_ph, gt_positive)
        
        classes_ph_tt = tf.boolean_mask(tt_classes_ph, tt_positive)
        classes_ph_gt = tf.boolean_mask(classes_ph, gt_positive)
        
        pos_mask_tt = tf.boolean_mask(tt_pos_mask, tt_positive)
        pos_mask_gt = tf.boolean_mask(pos_mask, gt_positive)
        
        conf_ph_tt = tf.boolean_mask(tt_conf_ph, tt_positive)
        
        default_resutls = (tf.cast(0, tf.float32),tf.cast(0, tf.float32),tf.cast(0, tf.float32),tf.cast(0, tf.int32))
        
        gt_class_loss, gt_bbox_loss, gt_train_acc, gt_number_of_positives = default_resutls
        gt_class_loss, gt_bbox_loss, gt_train_acc, gt_number_of_positives =\
            tf.cond(tf.equal(tt_rate, tf.cast(1, tf.float32)),
                    lambda:default_resutls,
                    lambda:detection_loss(gt_location, gt_confidence, refine_ph_gt, classes_ph_gt, pos_mask_gt))
        gt_det_loss = gt_class_loss + gt_bbox_loss
        
        tt_class_loss, tt_bbox_loss, tt_train_acc, tt_number_of_positives = default_resutls
        tt_class_loss, tt_bbox_loss, tt_train_acc, tt_number_of_positives =\
            tf.cond(tf.equal(tt_rate, tf.cast(0, tf.float32)),
                    lambda:default_resutls,
                    lambda:detection_loss_tutor(tt_location, tt_confidence, refine_ph_tt, classes_ph_tt,pos_mask_tt, conf_ph_tt))
        tt_det_loss = tt_class_loss + tt_bbox_loss
        
        det_loss = ( ( 1. - tt_rate ) * gt_det_loss ) + ( ( tt_rate ) * tt_det_loss )
        train_acc = ( ( 1. - tt_rate ) * gt_train_acc ) + ( ( tt_rate ) * tt_train_acc )
        
        the_loss += det_loss
        
    elif args.detect and not DET_TUTOR:
        class_loss, bbox_loss, train_acc, number_of_positives =\
            detection_loss(location, confidence, refine_ph, classes_ph, pos_mask)
        det_loss = class_loss + bbox_loss
        the_loss += det_loss

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    wd_loss = tf.add_n(regularization_losses)
    tf.summary.scalar('loss/weight_decay', wd_loss)
    the_loss += wd_loss

    tf.summary.scalar('loss/full', the_loss)
    return the_loss, train_acc, mean_iou, update_mean_iou

def objective_bothtutor_sum(location, confidence, refine_ph, classes_ph,
              pos_mask, tt_refine_ph, tt_classes_ph, tt_pos_mask, tt_conf_ph,
              seg_logits, seg_gt, seg_tt, dataset, config):
    
    the_loss = 0
    train_acc = tf.constant(1)
    mean_iou = tf.constant(1)
    update_mean_iou = tf.constant(1)

    if args.segment:
        seg_loss_gt, mean_iou_gt, update_mean_iou_gt = segmentation_loss(seg_logits, seg_gt, config, dataset)
        seg_loss_tt, mean_iou_tt, update_mean_iou_tt = segmentation_loss_tutor(seg_logits, seg_tt, config, dataset)
        
        seg_loss = ( 0.5 * seg_loss_gt ) + ( 0.5 * seg_loss_tt )
        mean_iou = ( 0.5 * mean_iou_gt ) + ( 0.5 * mean_iou_tt )
        update_mean_iou = ( 0.5 * update_mean_iou_gt ) + ( 0.5 * update_mean_iou_tt )
        
        the_loss += seg_loss

    if args.detect:
        gt_class_loss, gt_bbox_loss, gt_train_acc, gt_number_of_positives =\
            detection_loss(location, confidence, refine_ph, classes_ph, pos_mask)
        gt_det_loss = gt_class_loss + gt_bbox_loss
        
        tt_class_loss, tt_bbox_loss, tt_train_acc, tt_number_of_positives =\
            detection_loss_tutor(location, confidence, tt_refine_ph, tt_classes_ph, tt_pos_mask, tt_conf_ph)
        tt_det_loss = tt_class_loss + tt_bbox_loss
        
        det_loss = ( 0.5 * gt_det_loss ) + ( 0.5 * tt_det_loss )
        train_acc = ( 0.5 * gt_train_acc ) + ( 0.5 * tt_train_acc )
        
        the_loss += det_loss

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    wd_loss = tf.add_n(regularization_losses)
    tf.summary.scalar('loss_tutor/weight_decay', wd_loss)
    the_loss += wd_loss

    tf.summary.scalar('loss_tutor/full', the_loss)
    return the_loss, train_acc, mean_iou, update_mean_iou

def objective(location, confidence, refine_ph, classes_ph,
              pos_mask, seg_logits, seg_gt, dataset, config):
    
    the_loss = 0
    train_acc = tf.constant(1)
    mean_iou = tf.constant(1)
    update_mean_iou = tf.constant(1)

    if args.segment:
        seg_loss, mean_iou, update_mean_iou = segmentation_loss(seg_logits, seg_gt, config, dataset)
        the_loss += seg_loss

    if args.detect:
        class_loss, bbox_loss, train_acc, number_of_positives =\
            detection_loss(location, confidence, refine_ph, classes_ph, pos_mask)
        det_loss = class_loss + bbox_loss
        the_loss += det_loss

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    wd_loss = tf.add_n(regularization_losses)
    tf.summary.scalar('loss/weight_decay', wd_loss)
    the_loss += wd_loss

    tf.summary.scalar('loss/full', the_loss)
    return the_loss, train_acc, mean_iou, update_mean_iou

def extract_batch(dataset, config):
    with tf.device("/cpu:0"):
        bboxer = PriorBoxGrid(config)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, num_readers=2,
            common_queue_capacity=512, common_queue_min=32)
        if args.segment:
            im, bbox, gt, seg = data_provider.get(['image', 'object/bbox', 'object/label',
                                                   'image/segmentation'])
        else:
            im, bbox, gt = data_provider.get(['image', 'object/bbox', 'object/label'])
            seg = tf.expand_dims(tf.zeros(tf.shape(im)[:2]), 2)
        im = tf.to_float(im)/255
        bbox = yxyx_to_xywh(tf.clip_by_value(bbox, 0.0, 1.0))

        im, bbox, gt, seg = data_augmentation(im, bbox, gt, seg, config)
        inds, cats, refine = bboxer.encode_gt_tf(bbox, gt)

        return tf.train.shuffle_batch([im, inds, refine, cats, seg],
                                      args.batch_size, 2048, 64, num_threads=4)

def extract_conf_1(all_boxes, all_cats, all_conf):
    mask = tf.equal(all_conf, 1.)
    return tf.boolean_mask(all_boxes, mask), tf.boolean_mask(all_cats, mask), tf.boolean_mask(all_conf, mask)

def extract_conf_tutor(all_boxes, all_cats, all_conf):
    if FILTER_LOW_CONF_BOXES:
        mask = tf.logical_and( all_conf < 1., all_conf >= CONF_THRESHOLD )
    else:
        mask = tf.math.less(all_conf,1.)
    return tf.boolean_mask(all_boxes, mask), tf.boolean_mask(all_cats, mask), tf.boolean_mask(all_conf, mask)

def extract_batch_tutor(dataset, config):
    with tf.device("/cpu:0"):
        bboxer = PriorBoxGrid(config)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, num_readers=2,
            common_queue_capacity=512, common_queue_min=32)
        
        im, name, H, W, seg_gt, has_gt, seg_aug, has_aug, seg_tt, bbox_gt, labels_gt, diff_gt, bbox_tt, labels_tt, conf_tt = data_provider.get( 
            ['image','image/filename','image/height','image/width',
             'image/segmentation_gt','image/has_gt', 'image/segmentation_aug', 
             'image/has_aug', 'image/segmentation_tt', 'object_gt/bbox', 
             'object_gt/label','object_gt/difficulty','object_tt/bbox', 
             'object_tt/label', 'object_tt/confidence'])
        
        im = tf.to_float(im)/255
        bbox_gt = yxyx_to_xywh(tf.clip_by_value(bbox_gt, 0.0, 1.0))
        bbox_tt = yxyx_to_xywh(tf.clip_by_value(bbox_tt, 0.0, 1.0))
        seg_tt = tf.reshape(seg_tt, [ tf.shape(im)[0], tf.shape(im)[1], dataset.num_classes])
        has_gt = tf.reshape(has_gt,[1])
        has_aug = tf.reshape(has_aug,[1])
        
        img, all_boxes, all_cats, all_conf, gt_seg, aug_seg, tt_seg = data_augmentation_tutor(im, bbox_gt, labels_gt, bbox_tt, labels_tt, conf_tt, seg_gt, seg_aug, seg_tt, config)
        
        gt_boxes, gt_cats, gt_conf = extract_conf_1(all_boxes, all_cats, all_conf)
        tt_boxes, tt_cats, tt_conf = extract_conf_tutor(all_boxes, all_cats, all_conf)
        
        gt_inds, gt_cats, gt_refine, gt_conf = bboxer.encode_gt_tf_tutor(gt_boxes, gt_cats, gt_conf)
        tt_inds, tt_cats, tt_refine, tt_conf = bboxer.encode_gt_tf_tutor(tt_boxes, tt_cats, tt_conf)
        
        return tf.train.shuffle_batch([img, gt_inds, gt_refine, gt_cats, gt_conf, tt_inds, tt_refine, tt_cats, tt_conf, has_gt, gt_seg, has_aug, aug_seg, tt_seg], args.batch_size, 2048, 64, num_threads=4)

def train(dataset, net, config):
    image_ph, gt_inds, gt_refine, gt_cats, gt_conf, tt_inds, tt_refine, tt_cats, tt_conf, has_gt, gt_seg, has_aug, aug_seg, tt_seg = extract_batch_tutor(dataset, config)
    #image_ph, inds_ph, refine_ph, classes_ph, seg_gt = extract_batch(dataset, config)
    
    n_using_tutor = int(args.batch_size*TUTOR_DISTRIBUTION)
    using_tutor = tf.ones([n_using_tutor], dtype=tf.float32)
    not_using_tutor = tf.zeros([ args.batch_size - n_using_tutor ], dtype=tf.float32)
    tutor_use = tf.random_shuffle( tf.concat([using_tutor, not_using_tutor], 0) )
    
    not_has_gt = tf.squeeze( tf.cast( tf.logical_not(tf.cast(has_gt,tf.bool)), tf.float32) )
    
    net.create_trunk(image_ph)

    if args.detect:
        net.create_multibox_head(dataset.num_classes)
        confidence = net.outputs['confidence']
        location = net.outputs['location']
        tf.summary.histogram('location', location)
        tf.summary.histogram('confidence', confidence)
    else:
        location, confidence = None, None

    if args.segment:
        net.create_segmentation_head(dataset.num_classes)
        seg_logits = net.outputs['segmentation']
        tf.summary.histogram('segmentation', seg_logits)
    else:
        seg_logits = None

    #tutor_use: to use the tutors only when the image belongs to the subset defined by TUTOR_DISTRIBUTION
    #not_has_gt: to use the tutors only when the images miss GT 
    ###det tutor errors
    
    loss, train_acc, mean_iou, update_mean_iou = objective_tutor_distributed(location, confidence, gt_refine,
                                                           gt_cats, gt_inds, tt_refine, tt_cats,tt_inds, 
                                                           tt_conf, seg_logits, gt_seg, tt_seg,
                                                           tutor_use, dataset, config)
     
    #loss, train_acc, mean_iou, update_mean_iou = objective_bothtutor_sum(location, confidence, gt_refine,
    #                                                       gt_cats,gt_inds, tt_refine, tt_cats,tt_inds, 
    #                                                       tt_conf, seg_logits, gt_seg, tt_seg, dataset, config)
    
    #loss, train_acc, mean_iou, update_mean_iou = objective(location, confidence, gt_refine,
    #                                                       gt_cats,gt_inds, seg_logits,
    #                                                       gt_seg, dataset, config)


    ### setting up the learning rate ###
    global_step = slim.get_or_create_global_step()
    learning_rate = args.learning_rate

    learning_rates = [args.warmup_lr, learning_rate]
    steps = [args.warmup_step]

    if len(args.lr_decay) > 0:
        for i, step in enumerate(args.lr_decay):
            steps.append(step)
            learning_rates.append(learning_rate*10**(-i-1))

    learning_rate = tf.train.piecewise_constant(tf.to_int32(global_step),
                                                steps, learning_rates)

    tf.summary.scalar('learning_rate', learning_rate)
    #######

    if args.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif args.optimizer == 'nesterov':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError

    train_vars = tf.trainable_variables()
    print_variables('train', train_vars)

    train_op = slim.learning.create_train_op(
        loss, opt,
        global_step=global_step,
        variables_to_train=train_vars,
        summarize_gradients=True)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, keep_checkpoint_every_n_hours=1)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.random_trunk_init:
            print("Training from scratch")
        else:
            init_assign_op, init_feed_dict, init_vars = net.get_imagenet_init(opt)
            print_variables('init from ImageNet', init_vars)
            sess.run(init_assign_op, feed_dict=init_feed_dict)

        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if args.ckpt == 0:
                ckpt_to_restore = ckpt.model_checkpoint_path
            else:
                ckpt_to_restore = train_dir+'/model.ckpt-%i' % args.ckpt
            log.info("Restoring model %s..." % ckpt_to_restore)
            saver.restore(sess, ckpt_to_restore)

        starting_step = sess.run(global_step)
        tf.get_default_graph().finalize()
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        log.info("Launching prefetch threads")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        log.info("Starting training...")
        for step in range(starting_step, args.max_iterations+1):
            start_time = time.time()
            try:
                train_loss, acc, iou, _, lr = sess.run([train_op, train_acc, mean_iou,
                                                        update_mean_iou, learning_rate])
                
            except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
                break
            duration = time.time() - start_time

            num_examples_per_step = args.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('step %d, loss = %.2f, acc = %.2f, iou=%f, lr=%.3f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            log.info(format_str % (step, train_loss, acc, iou, -np.log10(lr),
                                examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 and step > 0:
                summary_writer.flush()
                log.debug("Saving checkpoint...")
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        
        summary_writer.close()
        
        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    assert args.detect or args.segment, "Either detect or segment should be True"
    if args.trunk == 'resnet50':
        net = ResNet
        depth = 50
    if args.trunk == 'vgg16':
        net = VGG
        depth = 16

    net = net(config=net_config, depth=depth, training=True, weight_decay=args.weight_decay)

    '''
    if args.dataset == 'voc12-trainval-tutor':
        dataset = get_dataset2('voc12-trainval-segmentation-DetectTutorOnly')
    if args.dataset == 'voc12-train-tutor':
        dataset = get_dataset2('voc12-train-segmentation-DetectTutorOnly')
    '''
    
    if args.dataset == 'voc12-Main-train' or args.dataset == 'voc12-Main-val' or args.dataset == 'voc12-Main-trainval':
        dataset = get_dataset_tutor(args.dataset)
    if (args.dataset == 'voc12-Segmentation-train' or args.dataset == 'voc12-Segmentation-val' 
        or args.dataset == 'voc12-Segmentation-trainval'):
        dataset = get_dataset_tutor(args.dataset)
    if args.dataset == 'voc12-SegmentationAug-train':
        dataset = get_dataset_tutor(args.dataset)
        
    if args.dataset == 'voc07':
        dataset = get_dataset('voc07_trainval')
    if args.dataset == 'voc12-trainval':
        dataset = get_dataset('voc12-train-segmentation', 'voc12-val')
    if args.dataset == 'voc12-train':
        dataset = get_dataset('voc12-train-segmentation')
    if args.dataset == 'voc12-val':
        dataset = get_dataset('voc12-val-segmentation')
    if args.dataset == 'voc07+12':
        dataset = get_dataset('voc07_trainval', 'voc12_train', 'voc12_val')
    if args.dataset == 'voc07+12-segfull':
        dataset = get_dataset('voc07-trainval-segmentation', 'voc12-train-segmentation', 'voc12-val')
    if args.dataset == 'voc07+12-segmentation':
        dataset = get_dataset('voc07_trainval', 'voc12-train-segmentation')
        #dataset = get_dataset('voc07-trainval-segmentation', 'voc12-train-segmentation')
    if args.dataset == 'coco':
        # support by default for coco trainval35k split
        dataset = get_dataset('coco-train2014-*', 'coco-valminusminival2014-*')
    if args.dataset == 'coco-seg':
        # support by default for coco trainval35k split
        dataset = get_dataset('coco-seg-train2014-*', 'coco-seg-valminusminival2014-*')

    train(dataset, net, net_config)

if __name__ == '__main__':
    exec_string = ' '.join(sys.argv)
    log.debug("Executing a command: %s", exec_string)
    cur_commit = subprocess.check_output("git log -n 1 --pretty=format:\"%H\"".split())
    cur_branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
    git_diff = subprocess.check_output('git diff --no-color'.split()).decode('ascii')
    log.debug("on branch %s with the following diff from HEAD (%s):" % (cur_branch, cur_commit))
    log.debug(git_diff)
    hostname = socket.gethostname()
    if 'gpuhost' in hostname:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        nvidiasmi = subprocess.check_output('nvidia-smi').decode('ascii')
        log.debug("Currently we are on %s and use gpu%s:" % (hostname, gpu_id))
        log.debug(nvidiasmi)
    tf.app.run()
