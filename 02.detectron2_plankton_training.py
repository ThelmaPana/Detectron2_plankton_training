#!/usr/bin/env python
# coding: utf-8

#--------------------------------------------------------------------------#
# Project: Detectron2_plankton_training
# Script purpose: Train a Detectron2 model on ISIIS plankton dataset
# Date: 02/04/2021
# Author: Thelma Panaiotis
#--------------------------------------------------------------------------#

"""
Code adapted from Detectron2 balloons training https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
Monitoring validation loss with code from https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
"""

## Import libraries
import torch
assert torch.__version__.startswith('1.7')

# Some basic setup:
# Setup detectron2 logger
import detectron2
#from detectron2.utils.logger import setup_logger
#setup_logger()

# import some common libraries
import numpy as np
import os
import json
import cv2
import random
import matplotlib.pyplot as plt
import copy
import glob
import pickle
import datetime
import logging
import yaml
#from importlib import reload

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, inference_context
from detectron2.engine import DefaultPredictor, DefaultTrainer

# import custom functions
import lib.training_functions as training_functions

###################################### Settings ######################################
# Directory to dataset
data_dir = 'data/detectron2_dataset'

# Set output_dir
output_dir = os.path.join('output', '_'.join(['output', datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")]))
os.mkdir(output_dir) # 

show_training = True # Whether to show a few frames from training set

base_model = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

ims_per_batch = 32 # batch size
max_iter = 30 
batch_size_per_image = 512 # default is 512
num_classes = 1  # only has one class (plankton)

base_lr = 0.00005 #  LR 0.000025 for BS=2
solver_gamma = 0.1 
solver_steps = (max_iter//3, 2*max_iter//3) # the iteration number to decrease learning rate by GAMMA
solver_warmup_factor = 1.0 / 1000
solver_warmup_iters = 1000
solver_warmup_method = 'linear'
solver_checkpoint_period = 100 # save a checkpoint after every this number of iterations
eval_period = 1000 # run evaluation on validation set after this number of iterations

test_threshold = 0.6 # score above which to consider objects for testing. Low for better recall, high for better accuracy.

######################################################################################

# Write settings (redundant with config writting but still)
settings = {
    'dataset': data_dir,
    'base_model': base_model,
    'batch_size': ims_per_batch,
    'max_iter': max_iter,
    'batch_size_per_image': batch_size_per_image,
    'num_classes': num_classes,
    'base_lr': base_lr,
    'solver_gamma': solver_gamma,
    'solver_steps': solver_steps,
    'solver_warmup_factor': solver_warmup_factor,
    'solver_warmup_iters': solver_warmup_iters,
    'solver_warmup_method': solver_warmup_method,
    'solver_checkpoint_period': solver_checkpoint_period,
    'eval_period': eval_period, 
    'test_threshold': test_threshold,
}
with open(os.path.join(output_dir, 'settings.pickle'),'wb') as set_file:
    pickle.dump(settings, set_file)

# Register the dataset to detectron2
for d in ['train', 'valid', 'test']:
    DatasetCatalog.register('plankton_' + d, lambda d=d: training_functions.format_bbox(os.path.join(data_dir, d)))
    MetadataCatalog.get('plankton_' + d).set(thing_classes=['plankton'])
plankton_metadata = MetadataCatalog.get('plankton_train') 
 
# Have a look at a few training frames
if show_training:
    dataset_dicts = training_functions.format_bbox(os.path.join(data_dir, 'train')) 
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:, :, ::-1], metadata=plankton_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize = (10,10))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()


## Model training
# Update config
cfg = get_cfg() # Get default config
cfg.merge_from_file(model_zoo.get_config_file(base_model)) 
cfg.DATASETS.TRAIN = ('plankton_train',) # Training set
cfg.DATASETS.TEST = ('plankton_valid',) # Validation set

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = test_threshold   

cfg.SOLVER.IMS_PER_BATCH = ims_per_batch 
cfg.SOLVER.BASE_LR = base_lr  
cfg.SOLVER.MAX_ITER = max_iter
cfg.SOLVER.GAMMA = solver_gamma
cfg.SOLVER.STEPS = solver_steps
cfg.SOLVER.WARMUP_FACTOR = solver_warmup_factor
cfg.SOLVER.WARMUP_ITERS = solver_warmup_iters
cfg.SOLVER.WARMUP_METHOD = solver_warmup_method
cfg.SOLVER.CHECKPOINT_PERIOD = solver_checkpoint_period

cfg.TEST.EVAL_PERIOD = eval_period

cfg.OUTPUT_DIR = output_dir

# Save config file
with open(os.path.join(output_dir, 'my_cfg.yaml'), 'w') as outfile:
    yaml.dump(cfg, outfile)

# Train
trainer = training_functions.MyTrainer(cfg=cfg) # Use custom trainer to monitor validation loss
trainer.resume_or_load(resume=False)
trainer.train()
print('Done training')


## Model evaluation
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth') # Load final model
cfg.DATASETS.TEST = ('plankton_test', ) # test set for model evaluation
predictor = DefaultPredictor(cfg) # Load default predictor

# We evaluate its performance using AP metric implemented in COCO API.
evaluator = COCOEvaluator('plankton_test', cfg, False, output_dir=output_dir)
test_loader = build_detection_test_loader(cfg, 'plankton_test')
test_results = inference_on_dataset(trainer.model, test_loader, evaluator)

# Write test results
with open(os.path.join(output_dir, 'test_results.pickle'),'wb') as test_file:
    pickle.dump(test_results, test_file)


