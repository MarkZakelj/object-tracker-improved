from pytracking.tracker.base import BaseTracker
# from pytracking.tracker.base import D3SBaseTracker
import torch
import torch.nn.functional as F
import torch.nn
import math
# import time
import numpy as np
import cv2
import copy

import ltr.data.processing_utils as prutils
# import ltr.data.d3sprocessing_utils as prutils
from ltr import load_network

from pytracking.mask_to_output import save_mask,show_mask,box2mask
# from pytracking.mask_to_disk import save_mask,show_mask,box2mask

class Segm(BaseTracker):
# class Segm(D3SBaseTracker):
    def loop_track(self, image, box)-> dict:
        
        self.pos=torch.Tensor([box[1]+(box[3]-1)/2,box[0]+(box[2]-1)/2])#?
        self.target_sz=torch.Tensor([box[3],box[2]])
        
        return self.track(image)
        

    def initialize(self, image, state, init_mask=None, *args, **kwargs):

        # Initialize some stuff
        self.frame_num = 1

        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # if len(state) == 4:
        state[0] -= 1
        state[1] -= 1
        # Get position and size
        self.pos = torch.Tensor([state[1] + state[3] / 2, state[0] + state[2] / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])
        self.gt_poly = np.array([state[0], state[1],
                                 state[0] + state[2] - 1, state[1],
                                 state[0] + state[2] - 1, state[1] + state[3] - 1,
                                 state[0], state[1] + state[3] - 1])

        if self.params.use_segmentation:
            self.init_segmentation(image, state, init_mask=init_mask)


    def track(self, image):

        self.frame_num += 1


        # Convert image
        new_pos = self.pos       

            
        # just a sanity check so that it does not get out of image
        if new_pos[0] < 0:
            new_pos[0] = 0
        if new_pos[1] < 0:
            new_pos[1] = 0
        if new_pos[0] >= image.shape[0]:
            new_pos[0] = image.shape[0] - 1
        if new_pos[1] >= image.shape[1]:
            new_pos[1] = image.shape[1] - 1

        pred_segm_mask = None
        if self.segmentation_task or (self.params.use_segmentation ):
            pred_segm_mask = self.segment_target(image, new_pos, self.target_sz)
            # pred_segm_region = self.segment_target(image, new_pos, self.target_sz)
            if pred_segm_mask is None:
                self.pos = new_pos.clone()
        else:
            self.pos = new_pos.clone()
        
        # Return new state
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))
        
        if self.params.use_segmentation:
            if pred_segm_mask is not None:
                # return pred_segm_region
                outputs = {'mask': self.maskresult,'bbox': new_state.tolist(), 'best_score': 1}
                return outputs

        # return new_state.tolist()
        self.maskresult=box2mask(new_state.tolist(),image)
        outputs = {'mask': self.maskresult,'bbox': new_state.tolist(), 'best_score': 1}
        return outputs

    def create_dist(self, width, height, cx=None, cy=None):

        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2

        x_ = np.linspace(1, width, width) - cx
        y_ = np.linspace(1, width, width) - cy
        X, Y = np.meshgrid(x_, y_)

        return np.sqrt(np.square(X) + np.square(Y)).astype(np.float32)

    def create_dist_gauss(self, map_sz, w, h, cx=None, cy=None, p=4, sz_weight=0.7):
        # create a square-shaped distance map with a Gaussian function which can be interpreted as a distance
        # to the given bounding box (center [cx, cy], width w, height h)
        # p is power of a Gaussian function
        # sz_weight is a weight of a bounding box size in Gaussian denominator
        if cx is None:
            cx = map_sz / 2
        if cy is None:
            cy = map_sz / 2

        x_ = np.linspace(1, map_sz, map_sz) - 1 - cx
        y_ = np.linspace(1, map_sz, map_sz) - 1 - cy
        X, Y = np.meshgrid(x_, y_)
        # 1 - is needed since we need distance-like map (not Gaussian function)
        return 1 - np.exp(-((np.power(X, p) / (sz_weight * w ** p)) + (np.power(Y, p) / (sz_weight * h ** p))))

    def init_segmentation(self, image, bb, init_mask=None):

        init_patch_crop, f_ = prutils.sample_target(image, np.array(bb), self.params.segm_search_area_factor,
                                                    output_sz=self.params.segm_output_sz)

        self.segmentation_task = False
        if init_mask is not None:
            mask = copy.deepcopy(init_mask).astype(np.float32)
            self.segmentation_task = True
            # self.params.segm_optimize_polygon = True
            # segmentation videos are shorter - therefore larger scale change factor can be used
            self.params.min_scale_change_factor = 0.9
            self.params.max_scale_change_factor = 1.1
            self.params.segm_mask_thr = 0.2
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            if hasattr(self, 'gt_poly'):
                p1 = self.gt_poly[:2]
                p2 = self.gt_poly[2:4]
                p3 = self.gt_poly[4:6]
                p4 = self.gt_poly[6:]
                cv2.fillConvexPoly(mask, np.array([p1, p2, p3, p4], dtype=np.int32), 1)
                mask = mask.astype(np.float32)
            else:
                p1 = bb[:2]
                p2 = [bb[0] + bb[2]-1, bb[1]]
                p3 = [bb[0] + bb[2]-1, bb[1] + bb[3]-1]
                p4 = [bb[0], bb[1] + bb[3]-1]
                cv2.fillConvexPoly(mask, np.array([p1, p2, p3, p4], dtype=np.int32), 1)
                mask = mask.astype(np.float32)

        init_mask_patch_np, patch_factor_init = prutils.sample_target(mask, np.array(bb),
                                                                      self.params.segm_search_area_factor,
                                                                      output_sz=self.params.segm_output_sz, pad_val=0)

        # network was renamed therefore we need to specify constructor_module and constructor_fun_name
        segm_net, _ = load_network(self.params.segm_net_path, backbone_pretrained=False,
                                   constructor_module='ltr.models.segm.segm',
                                   constructor_fun_name='segm_resnet50')

        if self.params.use_gpu:
            segm_net.cuda()
        segm_net.eval()

        for p in segm_net.segm_predictor.parameters():
            p.requires_grad = False

        self.params.segm_normalize_mean = np.array(self.params.segm_normalize_mean).reshape((1, 1, 3))
        self.params.segm_normalize_std = np.array(self.params.segm_normalize_std).reshape((1, 1, 3))

        # normalize input image
        init_patch_norm_ = init_patch_crop.astype(np.float32) / float(255)
        init_patch_norm_ -= self.params.segm_normalize_mean
        init_patch_norm_ /= self.params.segm_normalize_std

        # create distance map for discriminative segmentation
        if self.params.segm_use_dist:
            if self.params.segm_dist_map_type == 'center':
                # center-based dist map
                dist_map = self.create_dist(init_patch_crop.shape[0], init_patch_crop.shape[1])
            elif self.params.segm_dist_map_type == 'bbox':
                # bbox-based dist map
                dist_map = self.create_dist_gauss(self.params.segm_output_sz, bb[2] * patch_factor_init,
                                                  bb[3] * patch_factor_init)
            else:
                print('Error: Unknown distance map type.')
                exit(-1)

            dist_map = torch.Tensor(dist_map)

        # put image patch and mask to GPU
        init_patch = torch.Tensor(init_patch_norm_)
        init_mask_patch = torch.Tensor(init_mask_patch_np)
        if self.params.use_gpu:
            init_patch = init_patch.to(self.params.device)
            init_mask_patch = init_mask_patch.to(self.params.device)
            if self.params.segm_use_dist:
                dist_map = dist_map.to(self.params.device)
                dist_map = torch.unsqueeze(torch.unsqueeze(dist_map, dim=0), dim=0)
                test_dist_map = [dist_map]
            else:
                test_dist_map = None

        # reshape image for the feature extractor
        init_patch = torch.unsqueeze(init_patch, dim=0).permute(0, 3, 1, 2)
        init_mask_patch = torch.unsqueeze(torch.unsqueeze(init_mask_patch, dim=0), dim=0)

        # extract features (extracting twice on the same patch - not necessary)
        train_feat = segm_net.extract_backbone_features(init_patch)

        # prepare features in the list (format for the network)
        train_feat_segm = [feat for feat in train_feat.values()]
        test_feat_segm = [feat for feat in train_feat.values()]
        train_masks = [init_mask_patch]

        if init_mask is None:
            iters = 0
            while iters < 1:
                # Obtain segmentation prediction
                segm_pred = segm_net.segm_predictor(test_feat_segm, train_feat_segm, train_masks, test_dist_map)

                # softmax on the prediction (during training this is done internaly when calculating loss)
                # take only the positive channel as predicted segmentation mask
                mask = F.softmax(segm_pred, dim=1)[0, 0, :, :].cpu().numpy()
                mask = (mask > self.params.init_segm_mask_thr).astype(np.float32)

                if hasattr(self, 'gt_poly'):
                    # dilate polygon-based mask
                    # dilate only if given mask is made from polygon, not from axis-aligned bb (since rotated bb is much tighter)
                    dil_kernel_sz = max(5, int(round(0.05 * min(self.target_sz).item() * f_)))
                    kernel = np.ones((dil_kernel_sz, dil_kernel_sz), np.uint8)
                    mask_dil = cv2.dilate(init_mask_patch_np, kernel, iterations=1)
                    mask = mask * mask_dil
                else:
                    mask = mask * init_mask_patch_np

                target_pixels = np.sum((mask > 0.5).astype(np.float32))
                self.segm_init_target_pixels = target_pixels

                if self.params.save_mask:
                    segm_crop_sz = math.ceil(math.sqrt(bb[2] * bb[3]) * self.params.segm_search_area_factor)
                    # save_mask(None, mask, segm_crop_sz, bb, image.shape[1], image.shape[0],
                              # self.params.masks_save_path, self.sequence_name, self.frame_name)
                    self.maskresult=show_mask(None, mask, segm_crop_sz, bb, image.shape[1], image.shape[0],image)
                mask_gpu = torch.unsqueeze(torch.unsqueeze(torch.tensor(mask), dim=0), dim=0).to(self.params.device)
                train_masks = [mask_gpu]

                iters += 1
        else:
            init_mask_patch_np = (init_mask_patch_np > 0.1).astype(np.float32)
            target_pixels = np.sum((init_mask_patch_np).astype(np.float32))
            self.segm_init_target_pixels = target_pixels

            mask_gpu = torch.unsqueeze(torch.unsqueeze(torch.tensor(init_mask_patch_np), dim=0), dim=0).to(
                self.params.device)

        # store everything that is needed for later
        self.segm_net = segm_net
        self.train_feat_segm = train_feat_segm
        self.init_mask_patch = mask_gpu
        if self.params.segm_use_dist:
            self.dist_map = dist_map

        self.mask_pixels = np.array([np.sum(mask)])

    def segment_target(self, image, pos, sz):
        # pos and sz are in the image coordinates
        # construct new bounding box first
        tlx_ = pos[1] - sz[1] / 2
        tly_ = pos[0] - sz[0] / 2
        w_ = sz[1]
        h_ = sz[0]
        bb = [tlx_.item(), tly_.item(), w_.item(), h_.item()]

        # extract patch
        patch, f_ = prutils.sample_target(image, np.array(bb), self.params.segm_search_area_factor,
                                          output_sz=self.params.segm_output_sz)

        segm_crop_sz = math.ceil(math.sqrt(bb[2] * bb[3]) * self.params.segm_search_area_factor)

        # normalize input image
        init_patch_norm_ = patch.astype(np.float32) / float(255)
        init_patch_norm_ -= self.params.segm_normalize_mean
        init_patch_norm_ /= self.params.segm_normalize_std

        # put image patch and mask to GPU
        patch_gpu = torch.Tensor(init_patch_norm_)
        if self.params.use_gpu:
            patch_gpu = patch_gpu.to(self.params.device)

            # reshape image for the feature extractor
            patch_gpu = torch.unsqueeze(patch_gpu, dim=0).permute(0, 3, 1, 2)

        # extract features (extracting twice on the same patch - not necessary)
        test_feat = self.segm_net.extract_backbone_features(patch_gpu)

        # prepare features in the list (format for the network)
        test_feat_segm = [feat for feat in test_feat.values()]
        train_masks = [self.init_mask_patch]
        if self.params.segm_use_dist:
            if self.params.segm_dist_map_type == 'center':
                # center-based distance map
                test_dist_map = [self.dist_map]
            elif self.params.segm_dist_map_type == 'bbox':
                # bbox-based distance map
                D = self.create_dist_gauss(self.params.segm_output_sz, w_.item() * f_, h_.item() * f_)
                test_dist_map = [torch.unsqueeze(torch.unsqueeze(torch.Tensor(D).to(self.params.device), dim=0), dim=0)]
        else:
            test_dist_map = None

            # Obtain segmentation prediction
        segm_pred = self.segm_net.segm_predictor(test_feat_segm, self.train_feat_segm, train_masks, test_dist_map)

        # softmax on the prediction (during training this is done internaly when calculating loss)
        # take only the positive channel as predicted segmentation mask
        mask = F.softmax(segm_pred, dim=1)[0, 0, :, :].cpu().numpy()
        if self.params.save_mask:
            mask_real = copy.copy(mask)
        mask = (mask > self.params.segm_mask_thr).astype(np.uint8)

        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]

        if self.segmentation_task:
            mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, contours, -1, 1, thickness=-1)

            # save mask to disk
            # Note: move this below if evaluating on VOT
            if self.params.save_mask:
                # save_mask(None, mask_real, segm_crop_sz, bb, image.shape[1], image.shape[0],
                          # self.params.masks_save_path, self.sequence_name, self.frame_name)
                self.maskresult=show_mask(None, mask_real, segm_crop_sz, bb, image.shape[1], image.shape[0],image)
                   
        return self.maskresult