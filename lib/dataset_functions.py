import numpy as np
import pandas as pd
from psd_tools import PSDImage
from PIL import Image
import pycocotools.mask
import skimage.measure
import skimage.io
import hashlib
import os
import random
import shutil
import itertools
import matplotlib.pyplot as plt
import json


def fast_particle_area(x):
    return(np.sum(x._label_image[x._slice] == x.label))


def psd_to_array(psd_file, alpha_thres = 100, min_area = 20, max_area = 300):
    """
    Return mask and background image from photoshop file
    
    Args:
        psd_file (str): name of photoshop file from which to extract mask and background
        alpha_thres (int): alpha level; pixels above threshold will be considered as particles. Default = 100
        min_area (int): minimum size of particles in pixels. Default = 20 
        max_area (int): maximum size of particles in pixels. Default = 300 
    
    Returns:
        mask (ndarray): mask encoded as 0 and 1, 1 = particle, 0 = background
        back (ndarray): background image, 0 = black, 1 = white
    """
    
    ## Read PSD file
    psd = PSDImage.open(psd_file)
    
    ## Mask processing
    # Extract mask layer
    mask = psd[1]
    # Convert to numpy array
    mask_np = mask.numpy()
    
    # Extract alpha channel of mask layer
    mask_alpha = mask_np[:, :, 3]
    
    # Keep alpha channel > alpha_thres
    mask_ar = mask_alpha > alpha_thres/255 # mask_alpha is in [0, 1], so divide alpha threshold by 255

    # Label particles
    mask_lab = skimage.measure.label(mask_ar, background=False, connectivity=2)
    
    # Measure labelled particles
    regions = skimage.measure.regionprops(mask_lab)
    # Keep only regions with size between min_area and max_area
    large_regions = [r for r in regions if (fast_particle_area(r) > min_area) & (fast_particle_area(r) <= max_area)]
    
    # Initiate empty mask for selected particles
    mask_lab_large = np.zeros_like(mask_lab)
    # Loop over regions and write particle masks as ones
    for r in large_regions:
        mask_lab_large[r._slice] = (mask_lab[r._slice] == r.label)
    
    ## Background processing
    # Extract background layer
    back = psd[0]
    # Convert to numpy array and keep first channel
    back_np = back.numpy()[:, :, 0]
    
    return(mask_lab_large, back_np)


def extract_particles(mask, back, props, path = None):
    """
    Extract and measure particles from background image associated with a mask
    
    Args:
        mask (ndarray): mask, array of booleans, TRUE = particle, FALSE = background
        back (ndarray): background image, 0 = black, 1 = white
        props (list): properties to extract
    
    Returns:
        particles (dict): extracted particles
        particles_properties (dict): properties of extracted particles
    """
    
    # Label mask particles
    mask_label = skimage.measure.label(mask)
    
    # Extract masked particles
    particles_properties = skimage.measure.regionprops(mask_label, back)
    # number of particles
    n_part = len(particles_properties)
    
    # if particles are present in image, process them
    if n_part > 0:
        # for each particle:
        # - construct an image of the particle over blank space
        # - extract the measurements of interest
        particles = []
        
        # prepare a mask over the whole image on which retained particles will be shown
        particles_mask = np.ones_like(back, dtype=int)
        
        for x in particles_properties :
            # extract the particle
            particle = back[x._slice]
            # and its mask
            particle_mask = mask_label[x._slice]
            # blank out the pixels outside the particle
            particle = np.where(particle_mask == x.label, particle, 1.)
            particles = particles + [particle]
        
        
        # particles and their properties as dict
        particles = {hashlib.md5(p).hexdigest():p for p in particles}
        
        # store this as their first property
        particle_props = {'object_id': list(particles.keys())}
        
        # add frame name 
        #if path is not None:
        #    particle_props['frame'] = [path] * n_part
        
        # NB: append so that the md5 column is the first one
        particle_props.update(skimage.measure._regionprops._props_to_dict(particles_properties, properties=props))
        
        # convert to dataframe
        #particle_props = pd.DataFrame(particle_props)

    # If no particles were found
    else:
        # return an empty dict for particles and particle_props
        particles = {}
        particle_props = {}
    
    return(particles, particle_props)


def generate_slice(index, df_props, large_mask_id, back, output_size, zoom_range = 0.2, shift_range = 0.2, plot = False):
    """
    Generate a slice to extract a given object if object is not on border of large image. 
    Applies random zoom and shift around objects and makes sure object is not on border of sliced frame.
    
    Args:
        index (int): object index in properties dataframe
        df_props (dataframe): dataframe of objects properties
        large_mask_id (array): array of mask with particles
        back (array): array of background image
        output_size (int): size of output frame in px
        zoom_range (float): range of possible values for zoom. 0 for no zoom.
        shift_range (float): range of possible values for shift. 0 for no shift.
        plot (bool): whether to plot intermediate step frames
        
    Returns:
        my_slice (tuple): tuple containing top-left and bottom-right values of frame bbox. (None) if object is on border of large image.
 
    """
    
    ## Generate output_shape from output_size
    output_shape = [output_size, output_size]
    
    ## Check if objet is on image border
    # Get image width and height
    img_height = large_mask_id.shape[0]
    img_width = large_mask_id.shape[1]
    
    # Get object bbox
    bbox0 = df_props['bbox-0'][index]
    bbox1 = df_props['bbox-1'][index]
    bbox2 = df_props['bbox-2'][index]
    bbox3 = df_props['bbox-3'][index]
    
    ## If bbox does not touch image border
    if (bbox0 > 0) & (bbox1 > 0) & (bbox2 < img_height) & (bbox3 < img_width):
        # Compute object center
        obj_center = [
            df_props['bbox-0'][index] + df_props['local_centroid-0'][index],
            df_props['bbox-1'][index] + df_props['local_centroid-1'][index],
        ]
        # Define Top-Left (TL) and Bottom-Right (BR) of frame output, with object centered in frame
        TL = np.add(obj_center, -np.multiply(output_shape, 0.5)).astype(int)
        BR = np.add(obj_center, np.multiply(output_shape, 0.5)).astype(int)
        
        # Plots
        if plot:
            plt.figure(figsize = (10,10))
            plt.imshow(back[(slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))], cmap="gray")
            plt.title('Initial frame')
            plt.show()
            
            plt.figure(figsize = (10,10))
            plt.imshow(large_mask_id[(slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))], cmap="gray")
            plt.title('Initial frame')
            plt.show()
        
        
        ## Random zoom
        # Manual zoom in and out. This also shifts position of objects within frame but only on diagonal.
        # The same change is applied to the two values in TL, same for BR.
        # Draw random zoom in or out in zoom_range
        rand_z = [random.uniform(-zoom_range, zoom_range) for _ in range(2)]
        # Convert this to pixel values for top-left and bottom)right
        px_z = [round(r * output_size) for r in rand_z]
        TL_zoom = np.array((px_z[0], px_z[0]))
        BR_zoom = np.array((px_z[1], px_z[1]))
        # Apply changes to TL and BR values
        TL = TL + TL_zoom
        BR = BR + BR_zoom
        
        # Plots
        if plot:
            plt.figure(figsize = (10,10))
            plt.imshow(back[(slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))], cmap="gray")
            plt.title('After zoom')
            plt.show()
            
            plt.figure(figsize = (10,10))
            plt.imshow(large_mask_id[(slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))], cmap="gray")
            plt.title('After zoom')
            plt.show()
            

        ## Random shift
        # Manual shift. The same change is applied to the first value in TL and BR, and a different change for second values.
        # Draw random shift in shift_range
        rand_sh = [random.uniform(-shift_range, shift_range) for _ in range(2)]
        # Convert this to pixel values for top-left and bottom)right
        px_sh = [round(r * output_size) for r in rand_sh]
        TL_shift = np.array((px_sh[0], px_sh[1]))
        BR_shift = np.array((px_sh[0], px_sh[1]))
        # Apply changes to TL and BR values
        TL = TL + TL_shift
        BR = BR + BR_shift
        
        # Plots
        if plot:
            plt.figure(figsize = (10,10))
            plt.imshow(back[(slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))], cmap="gray")
            plt.title('After shift')
            plt.show()
            
            plt.figure(figsize = (10,10))
            plt.imshow(large_mask_id[(slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))], cmap="gray")
            plt.title('After shift')
            plt.show()
            
        
        ## Frame displacement if frame crosses image border
        # Check whether frame crosses image border
        if np.logical_or(np.any(TL<0), np.any(BR>np.array(large_mask_id.shape))):
            # Case of intersection with top or left border
            if np.any(TL<0):
                BR[TL<0] = BR[TL<0] + np.abs(TL[TL<0])
                TL[TL<0] = TL[TL<0] + np.abs(TL[TL<0])
                
            # Case of intersection with bottom or right border
            if np.any(BR>np.array(large_mask_id.shape)):
                TL[BR>np.array(large_mask_id.shape)] = TL[BR>np.array(large_mask_id.shape)] - \
                np.abs((np.array(large_mask_id.shape) - BR)[BR>np.array(large_mask_id.shape)])
                
                BR[BR>np.array(large_mask_id.shape)] = BR[BR>np.array(large_mask_id.shape)] - \
                np.abs((np.array(large_mask_id.shape) - BR)[BR>np.array(large_mask_id.shape)])
    
        # Plots
        if plot:
            plt.figure(figsize = (10,10))
            plt.imshow(back[(slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))], cmap="gray")
            plt.title("Final")
            plt.show()
            
            plt.figure(figsize = (10,10))
            plt.imshow(large_mask_id[(slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))], cmap="gray")
            plt.title("Final")
            plt.show()
            
    
        # Define slice from Top-Left and Bottom-Right limits
        my_slice = (slice(TL[0], BR[0], None), slice(TL[1], BR[1], None))
        
    # if object bbox is on image border, ignore object and set slice to None
    else:
        my_slice = (None,)
        
    return my_slice


## Functions for COCO format dataset
def create_sub_masks(mask_lab):
    """
    Create submasks for particles present in mask.
    Adapted from XXX.
    
    Args:
        mask_lab (ndarray): mask with labelled particles.
    
    Returns:
        sub_masks (dict): created submasks, one entry per particle label present in mask with key as str of particle label.  
    """
    
    # Get width and height of mask
    width, height = mask_lab.shape

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    
    # Scan pixels
    for x in range(height):
        for y in range(width):
            
            # Get the value of the pixel
            pixel = mask_lab[x,y]

            # If the pixel is not black...
            if pixel > 0:
                #print(x)
                #print(y)
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                
                # if no sub_mask created yet
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    #sub_masks[pixel_str] = Image.new('1', (width+2, height+2))
                    # No need to do such thing because frames are built in such way that particles are not on borders,
                    # and if a particle is on the border of the large image, it will be ignored when generating annotations
                    sub_masks[pixel_str] = Image.new('1', (width, height))

                # Set the pixel value to 1 (default is 0), accounting for padding
                #sub_masks[pixel_str].putpixel((y+1, x+1), 1)
                # No need to do such thing because frames are built in such way that particles are not on borders, 
                # and if a particle is on the border of the large image, it will be ignored when generating annotations
                sub_masks[pixel_str].putpixel((y, x), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, bbox_mode=1):
    """
    Create annotations from submask, ignoring objects on borders.
    Adapted from XXX.
    
    Args:
        sub_mask (PIL Image): submask from create_sub_masks function.
        image_id (str): name of frame.
        category_id (int): particle class is (1 in case of one class dataset, classif_id in case of multiclass dataset).
        bbox_mode (int): mode for bbox values; 0 for (x0, y0, x1, y1), 1 for (x0, y0, w, h). 
        
    Returns:
        annotation (dict): annotation for submask; including bbox, bbox_mode, category_id and segmentation contour
    """
    
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    
    # Convert PIL image mask to array 
    gt = np.array(sub_mask, dtype=np.uint8)
    fortran_gt = np.asfortranarray(gt)
    encoded_gt = pycocotools.mask.encode(fortran_gt)
    
    # Compute bbox and contours
    bbox = pycocotools.mask.toBbox(encoded_gt).tolist()
    contours = skimage.measure.find_contours(gt, 0.5)
    
    segmentations = []
    border = False
    for contour in contours:
        # Check is object is on border
        if np.logical_or(np.any(contour <= 0), np.any(contour >= sub_mask.size[0]-1)):
            border = True
        else:
            # if object is not on border, extract contour and compute segmentation
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            segmentations.append(segmentation)
    
    # if object is not on border, generate annotation
    if not border:
        annotation = {
            'bbox': bbox,
            'bbox_mode': bbox_mode,
            'category_id': category_id,
            'segmentation': segmentations
        }
    # if object is on border, generate an empty annotation
    else:
        annotation = {}

    return annotation


def split_dataset(frames_data, split = [70, 15, 15], show = False):
    """
    Split dataset into training, validation and testing sets; strasified on frame gray level.
    
    Args:
        frames_data (dict): frames annotations 
        split (list): split for training / validation / test. Sum should equal 100. Default is [70, 15, 15].
        
    Returns:
        df_frames (dataframe): dataframe with assigned split for each frame
        
    """
    
    # Initiate empty lists for frame names, psd files, mean gray level and category
    frames = []
    mean_gray_vals = []
    
    # Loop over list of frames data
    for i in range(len(frames_data)):
        # Store frame name and mean gray level
        frames.extend([frames_data[i]['image_id']]) 
        mean_gray_vals.extend([frames_data[i]['mean_gray']]) 
        
    # Convert to a DataFrame
    df_frames = pd.DataFrame({
        'frame': frames,
        'mean_gray': mean_gray_vals
    })
    # Round mean gray level to 0.01
    df_frames['mean_gray_round'] = round(df_frames.mean_gray, 2)
    
    # Group by rounded mean gray level and shuffle rows inside group
    df_frames = df_frames.groupby(['mean_gray_round']).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
    grouped = df_frames.groupby(['mean_gray_round'])
    
    # Compute rank of each occurrence within groups
    df_frames['rank'] = grouped.cumcount()+1
    # Compute number of occurrences per group
    df_frames['count'] = df_frames.groupby(['mean_gray_round'])['rank'].transform(len)
    # Compute percentile from rank and count
    df_frames['rank_prop'] = df_frames['rank']/df_frames['count']
    
    # Compute splits as percentages for training and training + validation
    split_1 = split[0]/100
    split_2 = (split[0] + split[1])/100
    df_frames['set'] = ['train' if x <= split_1 else 'valid' if x <= split_2 else 'test' for x in df_frames['rank_prop']]

    # Delete temporary columns
    df_frames = df_frames.drop(['rank', 'count', 'rank_prop'], axis = 1)
    
    # Display counts
    if show:
        print(df_frames.groupby(["set", "mean_gray_round"]).size())
    
    return df_frames


def move_frames(df_frames, frames_data, output_dir):
    """
    Move frames and associated annotations into assigned split directory.
    
    Args:
        df_frames (dataframe): dataframe with assigned split for each frame
        frames_data (dict): frames annotations 
        output_dir (str): path to output directory
        
    Returns:
        nothing
        
    """
    # Create empty lists for frames data for train, valid and test datasets     
    train_frames_data = []
    valid_frames_data = []
    test_frames_data = []
    
    # Loop over frames
    for idx in df_frames.index:
        # Get split name
        set_split = df_frames.set[idx]
        # Get frame name
        frame_name = df_frames.frame[idx]
        
        # Extract data associated with frame
        vec = [frames_data[i]['image_id'] ==  frame_name for i in range(len(frames_data))]
        frame_data = list(itertools.compress(frames_data, vec))
        
        if len(frame_data) > 1:
            print('Pb of image id: multiple dicts associated with one frame name.')
        
        # frame_data is a single element list, extract it
        frame_data = frame_data[0]
        
        # Compute old (temporary dir) and new (set split dir) path to image    
        old_path = frame_data['file_name']
        new_path = old_path
        new_path = new_path.replace('temporary', set_split)
        
        # Replace file_name entry in dict with new path
        frame_data['file_name'] = new_path
        
        # Append data to the relevant set
        if set_split == 'train':
            train_frames_data.append(frame_data)
        elif set_split == 'valid':
            valid_frames_data.append(frame_data)
        elif set_split == 'test':
            test_frames_data.append(frame_data)
        
        # Move image from temporary directory to split set directory
        shutil.move(old_path, new_path)
        
    # Write images data for train, valid and test sets
    with open(os.path.join(output_dir, 'train', 'images_data.json'), 'w') as fp:
        json.dump(train_frames_data, fp)
    with open(os.path.join(output_dir, 'valid', 'images_data.json'), 'w') as fp:
        json.dump(valid_frames_data, fp)
    with open(os.path.join(output_dir, 'test', 'images_data.json'), 'w') as fp:
        json.dump(test_frames_data, fp)

    