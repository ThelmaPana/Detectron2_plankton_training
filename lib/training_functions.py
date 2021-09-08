import os
import json
import logging
import numpy as np
import pandas as pd
import time
import datetime
import torch
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator, inference_context, DatasetEvaluator
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm


## Adapt bbox format for Detectron2
def format_bbox(data_dir):
    """
    Change bbox format in datasets to match Detectron2 input format
    
    Args
        data_dir (str): directory where to find dataset json file
        
    Returns
        data_dict (dict): dataset dict with proper bbox format
    """
    
    # Open json file in data_dir
    json_file = os.path.join(data_dir, 'images_data.json')
    with open(json_file) as f:
        data_dicts = json.load(f)
        
        # Loop over images
        for d in range(len(data_dicts)):
            # Change bbox_mode to be recognized by detectron2
            annot = data_dicts[d]['annotations']
            for a in range(len(annot)):
                annot[a]['bbox_mode'] = BoxMode.XYWH_ABS
            
    return data_dicts


## Custom training to monitor validation loss
# from https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        print(self.cfg) # to make sure that customized cfg is read by MyTrainer
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


## Hook to compute validation loss
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

        
def check_bbox_overlap(bb1, bb2):
    """
    Check if two bbox overlap or not.
    Args:
        bb1 (list): coordinates of 1st bbox as [x1, y1, x2, y2] 
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 (list): coordinates of 2nd bbox as [x1, y1, x2, y2] 
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
    Returns:
        bool: TRUE if there is an intersection, FALSE if not
    """
    # Determine the coordinates of the intersection rectangle
    x_left   = max(bb1[0], bb2[0])
    y_top    = max(bb1[1], bb2[1])
    x_right  = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        inter = False
    else:
        inter = True
    
    return(inter)


def bbox_iou(bb1, bb2):
    """
    Compute iou of two bbox.
    Args:
        bb1 (list): coordinates of 1st bbox as [x1, y1, x2, y2] 
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 (list): coordinates of 2nd bbox as [x1, y1, x2, y2] 
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
    Returns:
        iou (flt): iou value
    """
    # Determine the coordinates of the intersection rectangle
    x_left   = max(bb1[0], bb2[0])
    y_top    = max(bb1[1], bb2[1])
    x_right  = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        iou = 0
    
    else:
        # compute area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # compute area of both bbox
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        # union area is the sum of bbox areas minus intersection area
        union_area = bb1_area + bb2_area - intersection_area

        # compute iou
        iou = intersection_area / union_area
        
    return(iou)


## Custom evaluator (https://detectron2.readthedocs.io/en/latest/tutorials/evaluation.html)
class my_evaluator(DatasetEvaluator):
    def reset(self):
        self.pred_bbox = {
            'image_id': [],   # unique frame identifier
            'bbox': [],       # predicted bbox
            'score': [],      # prediction score
        }
        self.gt_bbox = {
            'image_id': [],   # unique frame identifier
            'bbox': [],       # gt bbox
        }
        self.inter_bbox = {
            'image_id': [],   # unique frame identifier
            'gt_bbox': [],    # gt bbox
            'pred_bbox': [],  # predicted bbox
            'score': [],      # prediction score
            'iou': [],        # gt and predicted bbox iou
        }
        self.ap = 0
        
        
    def process(self, image_id, gt_annots, outputs):
        """
        Extract ground truth bbox and predicted bbox with score.
        Args:
            image_id (str): unique image identifier
            gt_annots (list(dict)): ground truth annotations 
            outputs (list(dict)): model predictions for this image
            
        """
        for annot in gt_annots:
            # Get bbox
            bbox = np.around(annot['bbox']).astype('int')
            # Change bbox from XYWH to XYXY
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[3] + bbox[1]
            self.gt_bbox['bbox'].append(bbox)
            self.gt_bbox['image_id'].append(image_id)  

        # Extract instances
        instances = outputs[0]['instances']
        # Loop over prediction instances
        for i in range(len(instances)):
        
            # Extract instance
            instance = instances[i]
            
            # Extract prediction score
            score = instance.scores.cpu().detach().numpy()[0]
            # Extract bbox
            bbox = instance.pred_boxes.tensor.cpu().detach().numpy()[0]
            # Round and convert bbox values to int
            bbox = np.around(bbox).astype('int')
    
            # Store predictions
            self.pred_bbox['score'].append(score)
            self.pred_bbox['bbox'].append(bbox)
            self.pred_bbox['image_id'].append(image_id)
                
    def evaluate(self, output_dir):
        """
        Compute precision, recall (from https://github.com/rafaelpadilla/Object-Detection-Metrics)
        """
        # Convert dicts to df
        self.pred_bbox = pd.DataFrame(self.pred_bbox)
        self.gt_bbox = pd.DataFrame(self.gt_bbox)
        
        # List images ids
        image_ids = list(set(self.gt_bbox['image_id']))
        
        # Loop over predicted images
        for image_id in image_ids:
            # Extract prediction bbox in this image
            pred_bbox_img = self.pred_bbox[self.pred_bbox['image_id'] == image_id].reset_index(drop = True)
            # Extract gt bbox in this image
            gt_bbox_img = self.gt_bbox[self.gt_bbox['image_id'] == image_id].reset_index(drop = True)
            
            # Loop over gt bbox
            for i in gt_bbox_img.index:
                # Loop over pred bbox
                for j in pred_bbox_img.index:
                    # Check for bbox intercept
                    #inter = check_bbox_overlap(bb1 = gt_bbox_img['bbox'][i], bb2 = pred_bbox_img['bbox'][j])
                    iou = bbox_iou(bb1 = gt_bbox_img['bbox'][i], bb2 = pred_bbox_img['bbox'][j])
                    # In case of intercept, store it
                    if iou > 0:
                        #print(image_id)
                        self.inter_bbox['image_id'].append(image_id)
                        self.inter_bbox['gt_bbox'].append(gt_bbox_img['bbox'][i])
                        self.inter_bbox['pred_bbox'].append(pred_bbox_img['bbox'][j])
                        self.inter_bbox['score'].append(pred_bbox_img['score'][j])
                        self.inter_bbox['iou'].append(iou)
        
        # Convert dict to df
        self.inter_bbox = pd.DataFrame(self.inter_bbox)
        
        # If one gt bbox is matched with multiple pred bbox, keep the match with higher iou
        idx = self.inter_bbox.groupby(['image_id'])['iou'].transform(max) == self.inter_bbox['iou']
        self.inter_bbox = self.inter_bbox[idx].reset_index(drop = True)
        
        # Define each prediction as a TP or FP
        # Initially tag them all as FP
        self.pred_bbox['TP'] = 0
        
        for i in self.pred_bbox.index:
            # Check if prediction was matched with a gt bbox
            # if image_id and bbox are found in the same row in table of intersects
            if (self.pred_bbox['image_id'][i] == self.inter_bbox['image_id']).tolist() == [(self.pred_bbox['bbox'][i] == x).all() for x in self.inter_bbox['pred_bbox'].tolist()]:
                # it’s a TP
                self.pred_bbox.loc[i, 'TP'] = 1

        # Order predictions by desc score
        self.pred_bbox = self.pred_bbox.sort_values('score', ascending=False).reset_index(drop = True)
        # Generate a FP column
        self.pred_bbox['FP'] = 1 - self.pred_bbox['TP']
        
        # Cumulative sum of TP and FP
        self.pred_bbox['acc_TP'] = self.pred_bbox['TP'].cumsum()
        self.pred_bbox['acc_FP'] = self.pred_bbox['FP'].cumsum()
        
        # To compute recall we need the number of gt bbox
        self.pred_bbox['precision'] = self.pred_bbox['acc_TP'] / (self.pred_bbox['acc_TP'] + self.pred_bbox['acc_FP'])
        self.pred_bbox['recall'] = self.pred_bbox['acc_TP'] / (self.pred_bbox['acc_TP'] + len(self.gt_bbox))
        
        # Compute average precision
        ap, int_precision, int_recall, _ = self.CalculateAveragePrecision(
            rec = self.pred_bbox['recall'].tolist(), 
            prec = self.pred_bbox['precision'].tolist(),
        )
        self.ap = ap
        
        # Plot it
        self.plot(
            recall = self.pred_bbox['recall'].tolist(), 
            precision = self.pred_bbox['precision'].tolist(), 
            int_recall = int_recall,
            int_precision = int_precision,
            scores = self.pred_bbox['score'].tolist(), 
            ap = ap,
            output_dir = output_dir
        )
        
        
        return self.ap

    
    def plot(self, recall, precision, int_recall, int_precision, scores, ap, output_dir):
        """
        Plot precision x recall curve
        """
        plt.figure(figsize = (10, 6))
        plt.plot(int_recall, int_precision, 'k--', label='Precision')
        plt.scatter(recall, precision, c = scores)
        cbar = plt.colorbar()
        cbar.set_label('Prediction score')
        #plt.xlim(0, 1)
        #plt.ylim(0, 1)
        plt.clim(min(scores), 1)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Precision x Recall curve, AP: %s' % ('{0:.1f}%'.format(ap * 100)))
        #plt.show()
        plt.savefig(os.path.join(output_dir, 'precision_recall.png'))

        pass
    
    def CalculateAveragePrecision(self, rec, prec):
        """
        From https://github.com/rafaelpadilla/Object-Detection-Metrics
        """
        
        mrec = rec.copy()
        mrec.insert(0, 0)
        mrec.append(max(rec))
        mpre = prec.copy()
        mpre.insert(0, 0)
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1+i] != mrec[i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def get_all_inputs_outputs(test_loader, test_dataset_dicts, model):
    """
    Generate inputs and outputs.
    """
    for i, data in enumerate(test_loader):
            image_id = data[0]['image_id'] # get image id
            gt_annots = test_dataset_dicts[i]['annotations'] # get GT annotations
            yield image_id, gt_annots, model(data)
