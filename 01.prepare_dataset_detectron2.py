#!/usr/bin/env python
# coding: utf-8

"""
Script to generate training, validation and testing datasets for detectron2 with following properties:
 - it generates square output frames of desired size and .json annotations files.
 - it generates one frame per object, but multiple objects can coexist in the same frame, in such case they will appear multiple times in the dataset.
 - it only accounts for objects whose size is in specified size range.
 - it does image augmentation.
 - it makes sure that objects do not cross frame border.
 - it can use one or multiple classes.

First, all images are written into a temporary directory.
Then a train/validation/test split is done, stratified upon gray level so that clean and noisy frames are spreaded into all splits.

Code inspired from:
- Create COCO Annotations From Scratch: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch (for mask processing)
- How to build custom datasets for Detectron2: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

"""


import numpy as np
import pandas as pd
import os
import shutil
import itertools
import lycon
import matplotlib.pyplot as plt
import skimage
import skimage.measure
import hashlib                    
from shapely.geometry import Polygon, MultiPolygon 
import json
import cv2
import glob

import lib.dataset_functions as dataset_functions # my functions
#from importlib import reload


######################################## Settings ########################################
# Directory where to write output frames
output_dir = 'data/detectron2_dataset_400'
# If it doesn't exist, create it
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Switch for plotting slices
plot = False
    
# Switch to delete all files in output_dir
delete = False
if delete:
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)

# Create temporary dir
if not os.path.exists(os.path.join(output_dir, 'temporary')):
    os.mkdir(os.path.join(output_dir, 'temporary'))            
            
# Output size and shape of images 
output_size = 524
output_shape = [output_size, output_size]

# Size of particles to consider
min_area = 50
max_area = 400

# Data augmentation
zoom_range = 0.2 # maximum zoom range (e.g. 0.2 will allow zoom in or out of 20%)
shift_range  = 0.2 # maximum shift range

# Particle properties to extract to match with ecotaxa particles (for multiclass dataset)
my_props = [
    'label',
    'area', 
    'bbox',
    'slice',
    'local_centroid',
]

multiclass = False # whether to generate multiclass or uniclass dataset

split = [70, 15, 15] # split for training / validation / test

##########################################################################################

## Process objects exported from ecotaxa
# Read exported from EcoTaxa
eco_exp = pd.read_csv('data/ecotaxa_export_training_set.csv')

# Ignore objects outside of size range
eco_exp = eco_exp[(eco_exp['area'] > min_area) & (eco_exp['area'] < max_area)].reset_index(drop=True)

# Rename a few columns to match name of extracted properties
eco_exp = eco_exp.rename(columns = {
    'taxon': 'classif',
    'bbox0': 'bbox-0',
    'bbox1': 'bbox-1',
    'bbox2': 'bbox-2',
    'bbox3': 'bbox-3',
})
# Select usefull columns
eco_exp = eco_exp[['area', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'classif', 'classif_id']]



## Create frames
# Create empty list to store segmentation data. Will contain one dict for each frame. 
frames_data = []

# Directory where to read manual stacks
psd_files = glob.glob('data/manual_stack/**/**/Sans*.psd')
print(f'Found {len(psd_files)} psd files to process.')

# Initiate count to zero for large ignored particles
too_bigs = 0
# Initiate count to zero for empty frames
empty = 0

# Loop over manual stacks
for p in range(len(psd_files)):
    
    # Flag to show processing
    if (p+1) % 10 == 0:
        print(f'Done with {p+1} psd files out of {len(psd_files)}.')
    
    # Get path to psd file
    path = psd_files[p]
    
    # Extract mask and background from PSD stack
    mask, back = dataset_functions.psd_to_array(
        psd_file = path, 
        min_area = min_area, 
        max_area = max_area,
    )

    # Extract mask and background from PSD stack including large particles (wb = with bigs)
    mask_wb, _ = dataset_functions.psd_to_array(
        psd_file = path, 
        min_area = min_area, 
        max_area = 10000,
    )

    # Get image dimensions
    img_height, img_width = back.shape
    
    # Extract particles and their properties from mask and back
    parts, props = dataset_functions.extract_particles(
        mask = mask,
        back = back,
        props = my_props,
    )
    
    _, props_wb = dataset_functions.extract_particles(
        mask = mask_wb,
        back = back,
        props = my_props,
    )

    # Convert properties to dataframe
    df_props = pd.DataFrame(props)
    
    # If objects are present, process them
    # Add an index column and name of image
    if len(df_props) > 0:
        df_props['img_name'] = path
        
        # Convert parts from dict to list of array
        parts_arr = [x for x in parts.values()]
        
        # Join extracted particles with taxonomy
        df_props = pd.merge(df_props, eco_exp, how = 'inner', on=['area', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3'])       
        # Label particles
        mask_lab = skimage.measure.label(mask, background=False, connectivity=2) 
        mask_lab_wb = skimage.measure.label(mask_wb, background=False, connectivity=2)   
            
        # Loop over particles and extract objects   
        for i in df_props.index:
            #break
            # Generate slice for particle
            my_slice = dataset_functions.generate_slice(
                index = i, 
                df_props = df_props, 
                large_mask_id = mask_lab, 
                back = back, 
                output_size = output_size, 
                zoom_range = zoom_range, 
                shift_range = shift_range, 
                plot = False,
            )
            
            # If a slice was found
            if my_slice[0] is not None:
            
                # Extract back slice and mask slice
                back_slice = back[my_slice] # is back image
                mask_lab_slice = mask_lab[my_slice].astype(np.int32) # mask with particle labels (from skimage). Will be used after.
                mask_lab_wb_slice = mask_lab_wb[my_slice].astype(np.int32) # mask with particle labels (from skimage) including large (> max_area) particles
                
                # Resize mask and back to output_shape
                back_slice = cv2.resize(back_slice, (output_size, output_size))
                mask_lab_slice = cv2.resize(mask_lab_slice.astype('uint8'), (output_size, output_size), interpolation = cv2.INTER_NEAREST)
                mask_lab_wb_slice = cv2.resize(mask_lab_wb_slice.astype('uint8'), (output_size, output_size), interpolation = cv2.INTER_NEAREST)
                    
                if plot:
                    plt.figure(figsize=(10,10))
                    plt.imshow(back_slice, cmap='gray')
                    plt.show()
                    
                    plt.figure(figsize=(10,10))
                    plt.imshow(mask_id_slice)
                    plt.colorbar()
                    plt.show()
                    
                    plt.figure(figsize=(10,10))
                    plt.imshow(mask_lab_slice)
                    plt.colorbar()
                    plt.show()
                
                
                # Compute image_id (frame name) from hashing 
                image_id = hashlib.md5(back_slice.tobytes()).hexdigest()
                file_name = os.path.join(output_dir, 'temporary', image_id + '.png')
                
                # Set no presence of ignored large objects in image
                too_big_in_frame = False
                
                # Compute mean gray level
                mean_gray = back_slice.mean()
                # Convert to float64 to make json happy
                mean_gray = np.float64(mean_gray)
            
            
                ## Create the annotations for each objects in the frame
                # Initiate empty list for frame annotations
                annotations = []
                # Generate submasks
                # Use mask with particle label and not taxa id because we need one submask per instance and not one submask per taxon.
                sub_masks = dataset_functions.create_sub_masks(mask_lab_slice) 
                sub_masks_wb = dataset_functions.create_sub_masks(mask_lab_wb_slice)
                
                # If more objects in mask with large objects, large objects were ignored. Keep track of this.
                if len(sub_masks_wb) > len(sub_masks):
                    too_big_in_frame = True
                
                # Loop over submasks and extract properties
                for lab, sub_mask in sub_masks.items():
                    # Case of multiclass dataset
                    if multiclass:
                        # Find category_id matching index in dataframe of properties
                        category_id = int(df_props.classif_id[df_props.index == i])
                    
                    # Case of uniclass dataset
                    else:
                        # Set all categories to 0 to have only 1 category
                        category_id = 0
                        
                    # Generate annotation from submask
                    annotation = dataset_functions.create_sub_mask_annotation(sub_mask, image_id, category_id)
                    
                    # If annotation is not empty, append to list of annotations
                    if len(annotation) > 0:
                        annotations.append(annotation)
                
                    # Create entry for frame
                    frame_data = {
                        'file_name': file_name,
                        'orig_image': path.split('/')[-2],
                        'height': output_size,
                        'width': output_size,
                        'image_id': image_id,
                        'mean_gray': mean_gray,
                        'annotations': annotations
                    }
                    
                    # Count frames with objects larger than max_area
                    if too_big_in_frame:
                        too_bigs = too_bigs + 1   
                
                # Append dict for this frame to the list of segmentation data
                frames_data.append(frame_data)
                
                # Count empty frames. This happens if shift or zoom was to strong. Keep track to correct if needed.
                if len(frame_data['annotations']) == 0:
                    empty = empty + 1
                    
                    # Store list of empty frames
                    with open('empty_frames.txt', 'a') as f:
                        f.writelines('%s\n' % L for L in [path, 'i = ' + str(i), df_props['object_id'][i]])
                    
                # Write back image to output (temporary dir)
                cv2.imwrite(os.path.join(file_name), back_slice*255)

print(too_bigs, 'frames with ignored big particles')       
print(empty, 'empty frames')   
print('Done writting temporary images')    

# Write frame annotations to json format
with open(os.path.join(output_dir, 'images_data.json'), 'w') as fp:
    json.dump(frames_data, fp)

## Dataset split
# Split dataset into training, validation and testing data
df_frames = dataset_functions.split_dataset(frames_data, split, show=True)

# Create dirs for train, valid and test set
if not os.path.exists(os.path.join(output_dir, 'train')):
    os.mkdir(os.path.join(output_dir, 'train'))
if not os.path.exists(os.path.join(output_dir, 'valid')):
    os.mkdir(os.path.join(output_dir, 'valid'))
if not os.path.exists(os.path.join(output_dir, 'test')):
    os.mkdir(os.path.join(output_dir, 'test'))

## Loop over frames and move them in appropriate set
dataset_functions.move_frames(df_frames, frames_data, output_dir)

# Check that temporary dir is now empty and delete it
files = glob.glob(os.path.join(output_dir, 'temporary', '*'))

if len(files) == 0:
    shutil.rmtree(os.path.join(output_dir, 'temporary'))
else:
    print('Images left in temporary dir.')

# Delete temporary annotations file
os.remove(os.path.join(output_dir, 'images_data.json'))

print('Done')




