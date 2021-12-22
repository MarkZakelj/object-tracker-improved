
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import cv2
import math
import torch
import pickle
import importlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from siamreppoints.core.config import cfg
from siamreppoints.tracker.base_tracker import SiameseTracker
from siamreppoints.tracker.optim import ConvProblem, FactorizedConvProblem

from pytracking.features import augmentation
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features.preprocessing import numpy_to_torch, sample_patch, torch_to_numpy
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2

class SiamReppointsTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamReppointsTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.reshape(-1)
        self.model = model
        self.model.eval()
        
        self.mem_step = cfg.TRACK.MEM_STEP
        self.mem_len = cfg.TRACK.MEM_LEN
        self.st_mem_coef = cfg.TRACK.ST_MEM_COEF
        self.mem_sink_idx = cfg.TRACK.MEM_SINK_IDX

        ##Next part for Online Classification
        param_module = importlib.import_module('pytracking.parameter.segm.default_params')
        self.params = param_module.parameters()
        #self.fparams = self.params.features.get_fparams('feature_params')

        self.frame_num = 0
        
    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        
        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)
        
        self.features = [self.model.zf] * self.mem_len
        
        
        ##Next part for initilizing the Online Classification
        self.target_scale = 1.0 / scale_z
        self.pos = torch.Tensor([bbox[1] + (bbox[3] - 1)/2, bbox[0] + (bbox[2] - 1)/2])
        self.target_sz = torch.Tensor([bbox[3], bbox[2]])
        self.base_target_sz = self.target_sz / self.target_scale #the size of target in 255*255 image crop
        
        self.img_sample_sz = torch.Tensor([cfg.TRACK.INSTANCE_SIZE * 1.0, cfg.TRACK.INSTANCE_SIZE * 1.0])
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = TensorList([torch.Tensor([self.score_size * 1.0, self.score_size * 1.0])])
        self.output_sz = torch.Tensor([self.score_size * 1.0, self.score_size * 1.0])
        #import pdb
        #pdb.set_trace()
        #self.kernel_size = self.fparams.attribute('kernel_size')
        self.kernel_size = TensorList([self.params.kernel_size])
        self.visdom = None
        
        self.params.precond_learning_rate = TensorList([self.params.learning_rate])
        if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - max(self.params.precond_learning_rate))**self.params.CG_forgetting_rate
        
        self.init_learning()
        
        im = numpy_to_torch(img)

        x = self.generate_init_samples(im)

        self.init_projection_matrix(x)
              
        init_y = self.init_label_function(x)

        self.init_memory(x)

        self.init_optimization(x, init_y)  
        
        ##Next part for doing some strategy about motion
        self.frame_num = 0
        
        self.dist = []
        self.dist_x = []
        self.dist_y = []
        self.speed = 0.0
        self.speed_x = 0.0
        self.speed_y = 0.0
        self.disturbance_close_to_target=False
        self.disturbance_away_from_target=False
        self.disturbance_in_target=False
        self.previous_t_d_distance = 0
        self.inside_target_pos_x=0
        self.inside_target_pos_y=0
    
    def post_processing(self, score, pred_bbox, scale_z):
    
        pscore = score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        
        final_lr_score = score[best_idx]
        final_bbox = pred_bbox[best_idx, :]
        final_score = pscore[best_idx]
        lr = cfg.TRACK.LR * final_lr_score
        if final_lr_score > 0.35:
            lr = max(lr, 0.50)
        bbox = final_bbox
        box = np.array([0.0, 0.0, 0.0, 0.0])
        box[0] = (bbox[0] + bbox[2]) / 2 - cfg.TRACK.INSTANCE_SIZE // 2
        box[1] = (bbox[1] + bbox[3]) / 2 - cfg.TRACK.INSTANCE_SIZE // 2
        box[2] = (bbox[2] - bbox[0] + 1)
        box[3] = (bbox[3] - bbox[1] + 1)
        
        return box, final_score, lr, final_lr_score, best_idx
    
    def motion_strategy(self, score, pred_bbox, scale_z):
        pscore = score.copy()
        score = score.reshape(self.score_size, self.score_size)
        
        inside_weight = np.zeros_like(score)
        inside_width = cfg.TRACK.SCORE_INSIDE_WIDTH
        inside_right_idx = int(self.score_size - (self.score_size - inside_width) / 2)
        inside_left_idx = int(self.score_size - (self.score_size - inside_width) / 2 - inside_width)
        inside_weight[inside_left_idx:inside_right_idx,inside_left_idx:inside_right_idx] = np.ones((inside_width,inside_width))

        outside_width = cfg.TRACK.SCORE_OUTSIDE_WIDTH
        outside_right_idx = int(self.score_size - (self.score_size - outside_width) / 2)
        outside_left_idx = int(self.score_size - (self.score_size - outside_width) / 2 - outside_width)
        
        outside_weight = np.zeros_like(score)
        outside_weight[outside_left_idx:outside_right_idx,outside_left_idx:outside_right_idx] = np.ones((outside_width,outside_width))
        outside_weight = outside_weight - inside_weight
        
        inside_score = score * inside_weight
        outside_score = score * outside_weight

        flag = False
        
        if outside_score.max() > 0.3 and inside_score.max() > 0.4:
            inside_score = inside_score.reshape(-1)
            outside_score = outside_score.reshape(-1)
            inside_box, final_score, lr, final_lr_score, best_idx_inside = self.post_processing(inside_score, pred_bbox, scale_z)
            inside_pos_x = self.center_pos[0] + inside_box[0] / scale_z
            inside_pos_y = self.center_pos[1] + inside_box[1] / scale_z
            
            outside_box, final_score, lr, final_lr_score, best_idx_outside = self.post_processing(outside_score, pred_bbox, scale_z)
            outside_pos_x = self.center_pos[0] + outside_box[0] / scale_z
            outside_pos_y = self.center_pos[1] + outside_box[1] / scale_z
            
            target_disturbance_distance = np.sqrt((outside_pos_x - inside_pos_x)**2+(outside_pos_y - inside_pos_y)**2)
            
            if self.previous_t_d_distance == 0:
                self.previous_t_d_distance = target_disturbance_distance
            else:
                if target_disturbance_distance - self.previous_t_d_distance < 0:
                    self.disturbance_close_to_target = True

                    self.inside_target_pos_x = inside_pos_x
                    self.inside_target_pos_y = inside_pos_y
                    self.t_d_reset_count = 0
                elif target_disturbance_distance - self.previous_t_d_distance > 0 and self.disturbance_in_target is True:
                    self.disturbance_away_from_target = True

            box = target_box = inside_box
            flag = True
        else:
            box, final_score, lr, final_lr_score, best_idx_else = self.post_processing(pscore, pred_bbox, scale_z)
            if self.disturbance_close_to_target is True:
                self.disturbance_in_target = True
                self.previous_t_d_distance = 0
                inside_box = box
                inside_pos_x = self.center_pos[0] + inside_box[0] / scale_z
                inside_pos_y = self.center_pos[1] + inside_box[1] / scale_z
                self.t_d_reset_count = self.t_d_reset_count + 1
                if self.t_d_reset_count == 10:
                    self.disturbance_close_to_target = False
                    self.disturbance_in_target = False
                    self.disturbance_away_from_target = False  
            inside_box = box
            outside_box = box

        if flag:
            best_idx = best_idx_inside
        else:
            best_idx = best_idx_else

        if self.disturbance_away_from_target is True:
            inside_pos_x = self.center_pos[0] + inside_box[0] / scale_z
            inside_pos_y = self.center_pos[1] + inside_box[1] / scale_z
            target_inside_distance = np.sqrt((self.inside_target_pos_x - inside_pos_x)**2 + (self.inside_target_pos_y - inside_pos_y)**2)
            outside_pos_x = self.center_pos[0] + outside_box[0] / scale_z
            outside_pos_y = self.center_pos[1] + outside_box[1] / scale_z
            target_outside_distance = np.sqrt((outside_pos_x - self.inside_target_pos_x)**2 + (outside_pos_y - self.inside_target_pos_y)**2)
            if target_inside_distance > target_outside_distance:
                disturbance_box = inside_box
                target_box = outside_box
                if flag:
                    best_idx = best_idx_outside
                else:
                    best_idx = best_idx_else
            else:
                disturbance_box = outside_box
                target_box = inside_box
                if flag:
                    best_idx = best_idx_inside
                else:
                    best_idx = best_idx_else
                
            self.disturbance_close_to_target = False
            self.disturbance_in_target = False
            self.disturbance_away_from_target = False
            box = target_box
            
        return box, final_score, lr, final_lr_score, best_idx
            
    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        self.frame_num += 1
        
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
       
        pred_bbox = []
        scores_siamese = []
        
        for idx in range(self.mem_len):
            with torch.no_grad():
                if idx == 0:
                    outputs = self.model.track(x_crop, self.features[idx], cfg.TRACK.INSTANCE_SIZE)
                else:
                    outputs = self.model.tracking(outputs['search_feat'], self.features[idx], cfg.TRACK.INSTANCE_SIZE)
                scores_siamese.append(outputs['score'].view(-1).cpu().detach().numpy())
                pred_bbox.append(outputs['bbox'].cpu().detach().numpy().squeeze(0))
                
        if self.mem_len > 1:
            fuse_func = lambda x: x[0] * self.st_mem_coef + np.stack(x[1:], axis=0).mean(axis=0) * (1 - self.st_mem_coef)
            scores_siamese = fuse_func(scores_siamese)
        else:
            scores_siamese = scores_siamese[0]
        pred_bbox = pred_bbox[0]  ##using the 1st template to predict boundingbox
        
        test_feature = TensorList(outputs['feature'].unsqueeze(0))
        test_x = self.project_sample(test_feature)
        scores_match = self.apply_filter(test_x)[0].view(self.score_size, self.score_size).cpu().detach().numpy()
        scores_match = scores_match.reshape(-1, 1).flatten()
        scores_match = np.clip(scores_match, 0, 0.999)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz((pred_bbox[:, 2]-pred_bbox[:, 0]), (pred_bbox[:, 3]-pred_bbox[:, 1])) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     ((pred_bbox[:, 2]-pred_bbox[:, 0])/(pred_bbox[:, 3]-pred_bbox[:, 1])))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * scores_siamese
        
        pscore = pscore * (1 - cfg.TRACK.ONLINE_CLASSIFICATION_INFLUENCE) + scores_match * cfg.TRACK.ONLINE_CLASSIFICATION_INFLUENCE
        
        if self.speed_x > cfg.TRACK.SPEED_INFLUENCE * self.size[0] or self.speed_y > cfg.TRACK.SPEED_INFLUENCE * self.size[1] or \
            self.speed > cfg.TRACK.SPEED_INFLUENCE * max(self.size):
            cfg.TRACK.WINDOW_INFLUENCE = cfg.TRACK.WINDOW_INFLUENCE_FAST
        elif self.speed_x > self.size[0] or self.speed_y > self.size[1] or self.speed > max(self.size):
            cfg.TRACK.WINDOW_INFLUENCE = cfg.TRACK.WINDOW_INFLUENCE_MEDIUM
        else:
            cfg.TRACK.WINDOW_INFLUENCE = cfg.TRACK.WINDOW_INFLUENCE_SLOW

        box, best_score, lr, final_lr_score, best_idx = self.motion_strategy(pscore.copy(), pred_bbox, scale_z)
        
        final_bbox = box
        bbox = box / scale_z
        
        cx = self.center_pos[0] + bbox[0]
        cy = self.center_pos[1] + bbox[1]
        
        self.dist.append(math.sqrt(bbox[0]**2 + bbox[1]**2))
        self.dist_x.append(np.abs(bbox[0]))
        self.dist_y.append(np.abs(bbox[1]))

        if len(self.dist) < cfg.TRACK.SPEED_LAST_CALC:
            self.speed = max(self.dist)
            self.speed_x = max(self.dist_x)
            self.speed_y = max(self.dist_y)
        else:
            self.speed = max(self.dist[-cfg.TRACK.SPEED_LAST_CALC:])
            self.speed_x = max(self.dist_x[-cfg.TRACK.SPEED_LAST_CALC:])
            self.speed_y = max(self.dist_y[-cfg.TRACK.SPEED_LAST_CALC:])
        
        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        ##Next part for updating Online Classification and SiameseReppointsTracker template
        if (final_lr_score > 0.3):
            shift = torch.Tensor([final_bbox[1], final_bbox[0]])
            train_y = self.get_label_function(shift)
            self.update_memory(test_x, train_y, None)
        
        if (self.frame_num - 1) % self.params.train_skipping == 0 and self.frame_num > 1: 
            self.filter_optimizer.run(self.params.CG_iter)
        
        if (self.frame_num - 1) % self.mem_step == 0 and self.frame_num > 1 and final_lr_score > 0.45 and self.mem_len > 1:
            w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            s_z = round(np.sqrt(w_z * h_z))
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

            # get crop
            z_crop = self.get_subwindow(img, self.center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        s_z, self.channel_average)
            with torch.no_grad():
                feat = self.model.get_template(z_crop)
            self.features.pop(self.mem_sink_idx)
            self.features.append(feat)
        
        return {
                'bbox': bbox,
                'best_score': best_score,
                'best_idx': best_idx
               }
    
    
    def init_optimization(self, train_x, init_y):
        # Initialize filter
        filter_init_method = getattr(self.params, 'filter_init_method', 'zeros')
        
        self.filter = TensorList(
            [x.new_zeros(1, cdim, sz[0], sz[1]) for x, cdim, sz in zip(train_x, self.compressed_dim, self.kernel_size)])
        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'randn':
            for f in self.filter:
                f.normal_(0, 1/f.numel())
        else:
            raise ValueError('Unknown "filter_init_method"')

        # Get parameters
        self.params.update_projection_matrix = getattr(self.params, 'update_projection_matrix', True) and self.params.use_projection_matrix
        optimizer = getattr(self.params, 'optimizer', 'GaussNewtonCG')
        # Setup factorized joint optimization
        if self.params.update_projection_matrix:
            self.joint_problem = FactorizedConvProblem(self.init_training_samples, init_y, self.filter_reg,
                                                       TensorList([self.params.projection_reg]), self.params, self.init_sample_weights,
                                                       self.projection_activation, self.response_activation)

            # Variable containing both filter and projection matrix
            joint_var = self.filter.concat(self.projection_matrix)
            # Initialize optimizer
            analyze_convergence = getattr(self.params, 'analyze_convergence', False)
            if optimizer == 'GaussNewtonCG':
                self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var, debug=(self.params.debug >= 1),
                                                     plotting=(self.params.debug >= 3), analyze=analyze_convergence,
                                                     visdom=self.visdom)
            elif optimizer == 'GradientDescentL2':
                self.joint_optimizer = GradientDescentL2(self.joint_problem, joint_var, self.params.optimizer_step_length, self.params.optimizer_momentum, plotting=(self.params.debug >= 3), debug=(self.params.debug >= 1),
                                                         visdom=self.visdom)

            # Do joint optimization
            if isinstance(self.params.init_CG_iter, (list, tuple)):
                self.joint_optimizer.run(self.params.init_CG_iter)
            else:
                self.joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)

            if analyze_convergence:
                opt_name = 'CG' if getattr(self.params, 'CG_optimizer', True) else 'GD'
                for val_name, values in zip(['loss', 'gradient'], [self.joint_optimizer.losses, self.joint_optimizer.gradient_mags]):
                    val_str = ' '.join(['{:.8e}'.format(v.item()) for v in values])
                    file_name = '{}_{}.txt'.format(opt_name, val_name)
                    with open(file_name, 'a') as f:
                        f.write(val_str + '\n')
                raise RuntimeError('Exiting')

        # Re-project samples with the new projection matrix
        compressed_samples = self.project_sample(self.init_training_samples, self.projection_matrix)
        #compressed_samples = operation.channel_attention(compressed_samples, self.attention1, self.attention2, self.attention3)
        for train_samp, init_samp in zip(self.training_samples, compressed_samples):
            train_samp[:init_samp.shape[0],...] = init_samp

        self.hinge_mask = None

        # Initialize optimizer
        self.conv_problem = ConvProblem(self.training_samples, self.y, self.filter_reg, self.sample_weights, self.response_activation)

        if optimizer == 'GaussNewtonCG':
            self.filter_optimizer = ConjugateGradient(self.conv_problem, self.filter, fletcher_reeves=self.params.fletcher_reeves,
                                                      direction_forget_factor=self.params.direction_forget_factor, debug=(self.params.debug>=1),
                                                      plotting=(self.params.debug>=3), visdom=self.visdom)
        elif optimizer == 'GradientDescentL2':
            self.filter_optimizer = GradientDescentL2(self.conv_problem, self.filter, self.params.optimizer_step_length,
                                                      self.params.optimizer_momentum, debug=(self.params.debug >= 1),
                                                      plotting=(self.params.debug>=3), visdom=self.visdom)

        # Transfer losses from previous optimization
        if self.params.update_projection_matrix:
            self.filter_optimizer.residuals = self.joint_optimizer.residuals
            self.filter_optimizer.losses = self.joint_optimizer.losses

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)

        # Free memory
        del self.init_training_samples
        if self.params.use_projection_matrix:
            del self.joint_problem, self.joint_optimizer
            
    def init_learning(self):
        # Get window function
        self.feature_window = TensorList([dcf.hann2d(sz).cuda() for sz in self.feature_sz])

        # Filter regularization
        self.filter_reg = TensorList([self.params.filter_reg])

        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = getattr(self.params, 'projection_activation', 'none')
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.projection_activation = torch.nn.ReLU(inplace=True)
        elif projection_activation == 'elu':
            self.projection_activation = torch.nn.ELU(inplace=True)
        elif projection_activation == 'mlu':
            self.projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = getattr(self.params, 'response_activation', 'none')
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.response_activation = lambda x: x
        elif response_activation == 'relu':
            self.response_activation = torch.nn.ReLU(inplace=True)
        elif response_activation == 'elu':
            self.response_activation = torch.nn.ELU(inplace=True)
        elif response_activation == 'mlu':
            self.response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')
            
    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Generate augmented initial samples."""
        # Compute augmentation size(511*511) double the search area
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift operator
        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor).long().tolist()

        # Create transofmations
        self.transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.params.augmentation:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation['shift']])
        if 'relativeshift' in self.params.augmentation:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.params.augmentation['relativeshift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.params.augmentation:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.params.augmentation['blur']])
        if 'scale' in self.params.augmentation:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.params.augmentation['scale']])
        if 'rotate' in self.params.augmentation:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.params.augmentation['rotate']])

        ## Generate initial samples
        im_patch, _ = sample_patch(im, self.pos, self.target_scale*aug_expansion_sz, aug_expansion_sz)
        im_patches = torch.cat([T(im_patch) for T in self.transforms])

        with torch.no_grad():
            init_samples = TensorList([self.model.get_feature(im_patches.cuda())])

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(TensorList([self.params.use_augmentation])):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        # Add dropout samples
        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            for i, use_aug in enumerate(TensorList([self.params.use_augmentation])):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i], F.dropout2d(init_samples[i][0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        return init_samples

    def init_projection_matrix(self, x):
        # Set if using projection matrix
        self.params.use_projection_matrix = getattr(self.params, 'use_projection_matrix', True)

        if self.params.use_projection_matrix:
            self.compressed_dim = TensorList([self.params.compressed_dim])

            proj_init_method = getattr(self.params, 'proj_init_method', 'pca')
            if proj_init_method == 'pca':
                x_mat = TensorList([e.permute(1, 0, 2, 3).reshape(e.shape[1], -1).clone() for e in x])
                x_mat -= x_mat.mean(dim=1, keepdim=True)
                cov_x = x_mat @ x_mat.t()
                self.projection_matrix = TensorList(
                    [None if cdim is None else torch.svd(C)[0][:, :cdim].t().unsqueeze(-1).unsqueeze(-1).clone() for C, cdim in
                     zip(cov_x, self.compressed_dim)])
            elif proj_init_method == 'randn':
                self.projection_matrix = TensorList(
                    [None if cdim is None else ex.new_zeros(cdim,ex.shape[1],1,1).normal_(0,1/math.sqrt(ex.shape[1])) for ex, cdim in
                     zip(x, self.compressed_dim)])
        else:
            self.compressed_dim = x.size(1)
            self.projection_matrix = TensorList([None]*len(x))
       
    def init_label_function(self, train_x):
        # Allocate label function
        self.y = TensorList([x.new_zeros(self.params.sample_memory_size, 1, self.score_size, self.score_size) for x in train_x])
        
        # Output sigma factor
        output_sigma_factor = TensorList([self.params.output_sigma_factor])
        self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized coords
        target_center_norm = (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)

        # Generate label functions
        for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz, self.kernel_size, train_x):
            center_pos = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * sz
                y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center)
        
        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0], ...] for y, x in zip(self.y, train_x)])

    def init_memory(self, train_x):
        # Initialize first-frame training samples
        self.num_init_samples = train_x.size(0)
        self.init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        self.init_training_samples = train_x

        # Sample counters and weights
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, cdim, x.shape[2], x.shape[3]) for x, cdim in
             zip(train_x, self.compressed_dim)])
        self.training_samples = self.training_samples

    def project_sample(self, x: TensorList, proj_matrix = None):
        # Apply projection matrix
        if proj_matrix is None:
            proj_matrix = self.projection_matrix
        return operation.conv2d(x, proj_matrix, mode='same').apply(self.projection_activation)
        
    def apply_filter(self, sample_x: TensorList):
        return operation.conv2d(sample_x, self.filter, mode=None).apply(self.response_activation)
        
    def get_label_function(self, shift):
        # Generate label function
        train_y = TensorList()
        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            center = shift / self.img_support_sz * sz
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y
        
    def update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate = None):
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, TensorList([self.params]), learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y
        if self.hinge_mask is not None:
            for m, y, ind in zip(self.hinge_mask, sample_y, replace_ind):
                m[ind:ind+1,...] = (y >= self.params.hinge_threshold).float()
        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams, learning_rate = None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind
    
    