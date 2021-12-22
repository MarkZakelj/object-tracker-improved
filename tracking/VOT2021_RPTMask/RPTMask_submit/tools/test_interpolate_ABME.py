from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import vot
from vot import VOT
import sys
import time
import cv2
import numpy as np
import torch
import argparse
import os
import random
from pathlib import Path


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


seed_torch(1000000007)

filepath = os.path.abspath(__file__)
filelocation = os.path.split(filepath)[0]
rootpath = os.path.dirname(filelocation)
sys.path.append(rootpath)

from siamreppoints.core.config import cfg
from siamreppoints.models.model_builder import ModelBuilder
from siamreppoints.tracker.tracker_builder import build_tracker
from siamreppoints.utils.bbox import get_axis_aligned_bbox
from siamreppoints.utils.model_load import load_pretrain

torch.set_num_threads(4)
from pytracking.evaluation import Tracker as Trackers
from pytracking.refine_modules.refine_module import RefineModule

parser = argparse.ArgumentParser(description='SiamRP tracker')

parser.add_argument('--config', default=os.path.join(rootpath, 'experiments/siamreppoints/config.yaml'), type=str,
                    help='config file')
parser.add_argument('--snapshot', default=os.path.join(rootpath, '../models/siamreppoints.model'), type=str,
                    help='tracker model name')
parser.add_argument('--armodel', default=os.path.join(rootpath, '../models/SEcmnet_ep0040.pth.tar'), type=str,
                    help='segmentation model name')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
args = parser.parse_args()


def rect_from_mask(mask):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def get_ar(img, init_box, mask, ar_path):
    """ set up Alpha-Refine """
    selector_path = 0
    sr = 2.0
    input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, selector_path, search_factor=sr, input_sz=input_sz)
    RF_module.initialize(img, np.array(init_box), mask)
    return RF_module


class NCCTracker(object):

    def __init__(self, image, mask):

        # generate bbox as vot2019
        rotated_bbox = self._mask_post_processing(mask)
        rotated_bbox = np.array(
            [rotated_bbox[0][0], rotated_bbox[0][1], rotated_bbox[1][0], rotated_bbox[1][1], rotated_bbox[2][0],
             rotated_bbox[2][1], rotated_bbox[3][0], rotated_bbox[3][1]])
        cx, cy, w, h = self.get_axis_aligned_bbox(rotated_bbox)
        gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]

        # load config
        cfg.merge_from_file(args.config)

        # create model
        model = ModelBuilder()

        # load model
        model = load_pretrain(model, args.snapshot).cuda().eval()

        # build tracker
        self.tracker = build_tracker(model)

        # build segmentation net
        # tracker_name='segm'
        # tracker_param='default_params'
        # run_id=None
        # d3strackers = [Trackers(tracker_name, tracker_param, run_id)]#tracker_name tracker_param,run_id
        # params=d3strackers[0].get_parameters()
        # params.segm_net_path=args.d3smodel
        # self.d3stracker=d3strackers[0].tracker_class(params)

        # init SiamRP
        self.tracker.init(image, gt_bbox_)
        # d3simg=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # self.d3stracker.initialize(d3simg,gt_bbox_,mask*255)
        ar_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ar_init_bbox = rect_from_mask(mask)
        self.RF_module = get_ar(ar_img, ar_init_bbox, mask, args.armodel)
        self.thres_outer = 0.75
        self.thres_inner = 0.55
        self.distance_threshold = 64

    def _manhattan_distance(self, bbox, best_idx):
        center1 = np.array(bbox, dtype=np.int)[2:]
        center2 = np.array([best_idx // 25, best_idx % 25], dtype=np.int)
        return np.abs(center1 - center2).sum()

    def track(self, image):

        outputs = self.tracker.track(image)
        pred_bbox = outputs['bbox']
        score = outputs['best_score']
        best_idx = outputs['best_idx']

        # d3simg=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # outputs=self.d3stracker.loop_track(d3simg,outputs['bbox'])

        ar_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = self.RF_module.refine(ar_img, np.array(pred_bbox), mode='all', test=True)
        refined_bbox = output['corner']
        rb = [int(refined_bbox[0]), int(refined_bbox[1]),
              int(refined_bbox[0] + refined_bbox[2]), int(refined_bbox[1] + refined_bbox[3])]
        mask = (output['mask'] > self.thres_outer).astype(np.uint8)
        mask[rb[1]:rb[3], rb[0]:rb[2]] = (output['mask'][rb[1]:rb[3], rb[0]:rb[2]] > self.thres_inner).astype(np.uint8)

        pred_bbox_rect = np.array(
            [pred_bbox[0], pred_bbox[1], pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]]).astype(np.int32)
        refined_bbox_rect = np.array([refined_bbox[0], refined_bbox[1], refined_bbox[0] + refined_bbox[2],
                                      refined_bbox[1] + refined_bbox[3]]).astype(np.int32)
        cross_box = np.array(
            [max(pred_bbox_rect[0], refined_bbox_rect[0]), max(pred_bbox_rect[1], refined_bbox_rect[1]),
             min(pred_bbox_rect[2], refined_bbox_rect[2]), min(pred_bbox_rect[3], refined_bbox_rect[3])])
        mask[cross_box[1]:cross_box[3], cross_box[0]:cross_box[2]] = (
                    output['mask'][cross_box[1]:cross_box[3], cross_box[0]:cross_box[2]] > 0.45).astype(np.uint8)
        # if self._manhattan_distance(refined_bbox, best_idx) < self.distance_threshold:
        #     w = np.zeros(mask.shape, dtype=np.uint8)
        #     w[refined_bbox[1]: refined_bbox[1] + refined_bbox[3],
        #     refined_bbox[0]: refined_bbox[2] + refined_bbox[2]] = 1
        #     mask *= w
        # mask = (mask > self.thres).astype(np.uint8)

        # make sure no empty mask output
        # mask = (outputs['mask']>0.3).astype(np.uint8)
        if mask.astype(np.float).sum() / (refined_bbox[2] * refined_bbox[3]) > 0.1:
            mask_bbox = rect_from_mask(mask)
            return mask, pred_bbox, refined_bbox, mask_bbox, score, best_idx
        else:
            img = np.zeros(mask.shape, dtype=np.uint8)
            img[int(refined_bbox[1]):int(refined_bbox[1] + refined_bbox[3]),
            int(refined_bbox[0]):int(refined_bbox[0] + refined_bbox[2])] = 1
            mask = img
            mask_bbox = refined_bbox
        return mask, pred_bbox, refined_bbox, mask_bbox, score, best_idx

    def _mask_post_processing(self, mask):
        target_mask = (mask > 0.5)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 50:  # cnt_area=100
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)

            ## the following code estimate the shape angle with ellipse
            ## then fit a axis-aligned bounding box on the rotated image

            ellipseBox = cv2.fitEllipse(polygon)
            # get the center of the ellipse and the angle
            angle = ellipseBox[-1]
            # print(angle)
            center = np.array(ellipseBox[0])
            axes = np.array(ellipseBox[1])

            # get the ellipse box
            ellipseBox = cv2.boxPoints(ellipseBox)

            # compute the rotation matrix
            rot_mat = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)

            # rotate the ellipse box
            one = np.ones([ellipseBox.shape[0], 3, 1])
            one[:, :2, :] = ellipseBox.reshape(-1, 2, 1)
            ellipseBox = np.matmul(rot_mat, one).reshape(-1, 2)

            # to xmin ymin xmax ymax
            xs = ellipseBox[:, 0]
            xmin, xmax = np.min(xs), np.max(xs)
            ys = ellipseBox[:, 1]
            ymin, ymax = np.min(ys), np.max(ys)
            ellipseBox = [xmin, ymin, xmax, ymax]

            # rotate the contour
            one = np.ones([polygon.shape[0], 3, 1])
            one[:, :2, :] = polygon.reshape(-1, 2, 1)
            polygon = np.matmul(rot_mat, one).astype(int).reshape(-1, 2)

            # remove points outside of the ellipseBox
            logi = polygon[:, 0] <= xmax
            logi = np.logical_and(polygon[:, 0] >= xmin, logi)
            logi = np.logical_and(polygon[:, 1] >= ymin, logi)
            logi = np.logical_and(polygon[:, 1] <= ymax, logi)
            polygon = polygon[logi, :]

            x, y, w, h = cv2.boundingRect(polygon)
            bRect = [x, y, x + w, y + h]

            # get the intersection of ellipse box and the rotated box
            x1, y1, x2, y2 = ellipseBox[0], ellipseBox[1], ellipseBox[2], ellipseBox[3]
            tx1, ty1, tx2, ty2 = bRect[0], bRect[1], bRect[2], bRect[3]
            xx1 = min(max(tx1, x1, 0), target_mask.shape[1] - 1)
            yy1 = min(max(ty1, y1, 0), target_mask.shape[0] - 1)
            xx2 = max(min(tx2, x2, target_mask.shape[1] - 1), 0)
            yy2 = max(min(ty2, y2, target_mask.shape[0] - 1), 0)

            rotated_mask = cv2.warpAffine(target_mask, rot_mat, (target_mask.shape[1], target_mask.shape[0]))

            # refinement
            alpha_factor = 0.2583  # cfg.TRACK.FACTOR
            while True:
                if np.sum(rotated_mask[int(yy1):int(yy2), int(xx1)]) < (yy2 - yy1) * alpha_factor:
                    temp = xx1 + (xx2 - xx1) * 0.02
                    if not (temp >= target_mask.shape[1] - 1 or xx2 - xx1 < 1):
                        xx1 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy1):int(yy2), int(xx2)]) < (yy2 - yy1) * alpha_factor:
                    temp = xx2 - (xx2 - xx1) * 0.02
                    if not (temp <= 0 or xx2 - xx1 < 1):
                        xx2 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy1), int(xx1):int(xx2)]) < (xx2 - xx1) * alpha_factor:
                    temp = yy1 + (yy2 - yy1) * 0.02
                    if not (temp >= target_mask.shape[0] - 1 or yy2 - yy1 < 1):
                        yy1 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy2), int(xx1):int(xx2)]) < (xx2 - xx1) * alpha_factor:
                    temp = yy2 - (yy2 - yy1) * 0.02
                    if not (temp <= 0 or yy2 - yy1 < 1):
                        yy2 = temp
                    else:
                        break
                else:
                    break

            prbox = np.array([[xx1, yy1], [xx2, yy1], [xx2, yy2], [xx1, yy2]])

            # inverse of the rotation matrix
            M_inv = cv2.invertAffineTransform(rot_mat)
            # project the points back to image coordinate
            one = np.ones([prbox.shape[0], 3, 1])
            one[:, :2, :] = prbox.reshape(-1, 2, 1)
            prbox = np.matmul(M_inv, one).reshape(-1, 2)

            rbox_in_img = prbox
        else:  # empty mask
            # location = cxy_wh_2_rect(self.center_pos, self.size)
            location = [0, 0, 1, 1]
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img

    def get_axis_aligned_bbox(self, region):
        """ convert region to (cx, cy, w, h) that represent by axis aligned box
        """
        nv = region.size
        if nv == 8:
            cx = np.mean(region[0::2])
            cy = np.mean(region[1::2])
            x1 = min(region[0::2])
            x2 = max(region[0::2])
            y1 = min(region[1::2])
            y2 = max(region[1::2])
            A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
                 np.linalg.norm(region[2:4] - region[4:6])
            A2 = (x2 - x1) * (y2 - y1)
            s = np.sqrt(A1 / A2)
            w = s * (x2 - x1) + 1
            h = s * (y2 - y1) + 1
        else:
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]
            cx = x + w / 2
            cy = y + h / 2
        return cx, cy, w, h

    def _rect_from_mask(self, mask):
        '''
        create an axis-aligned rectangle from a given binary mask
        mask in created as a minimal rectangle containing all non-zero pixels
        '''
        # print(mask)
        x_ = np.sum(mask, axis=0)
        y_ = np.sum(mask, axis=1)
        x0 = np.min(np.nonzero(x_))
        x1 = np.max(np.nonzero(x_))
        y0 = np.min(np.nonzero(y_))
        y1 = np.max(np.nonzero(y_))
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def make_full_size(x, offset, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - int(x.shape[1] + offset[0])
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0

    front_pad_x = output_sz[0] - pad_x - x.shape[1]
    if front_pad_x < 0:
        x = x[:, x.shape[1] + front_pad_x:]
        front_pad_x = 0

    pad_y = output_sz[1] - int(x.shape[0] + offset[1])
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    front_pad_y = output_sz[1] - pad_y - x.shape[0]
    if front_pad_y < 0:
        x = x[x.shape[0] + front_pad_y:, :]
        front_pad_y = 0
    # print(front_pad_y+pad_y+x.shape[0],output_sz[1])
    # print(front_pad_x+pad_x+x.shape[1],output_sz[0])
    assert ((front_pad_y + pad_y + x.shape[0]) == output_sz[1])
    assert ((front_pad_x + pad_x + x.shape[1]) == output_sz[0])
    return np.pad(x, ((front_pad_y, pad_y), (front_pad_x, pad_x)), 'constant', constant_values=0)


def get_interpolated_filename(filename0, filename1, interpolation_method):
    """
        get filename (path) of interpolated frame between frame with filename0 and filename1
        (if original filename0 is 00000001.jpg and filename1 is 00000002.jpg return interpolation between 00000001.jpg and 00000002.jpg)
    """
    f0 = int(Path(filename0).stem)
    f1 = int(Path(filename1).stem)
    filename_name = '{:08}'.format(min(f0, f1)) + '.jpg'
    p = Path(filename0)
    pp = p.parent.parent
    # check if interpolation exists
    int_filename = Path(pp, 'interpolated', interpolation_method, filename_name)
    if int_filename.exists():
        return int_filename.as_posix()
    else:
        raise FileNotFoundError('No interpolated frame found')




# handle = vot.VOT("mask")
handle = VOT("mask")
selection, offset = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

frame_t0 = imagefile

image = cv2.imread(imagefile)
# mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
mask = make_full_size(selection, offset, (image.shape[1], image.shape[0]))

if selection.max() == 0:
    raise ('selection.max() is 0')
if mask.max() == 0:
    raise ('mask.max()==0')
color = np.array([0, 255, 0], dtype='uint8')
color2 = np.array([255, 0, 0], dtype='uint8')
tracker = NCCTracker(image, mask)

INTERPOLATE = True  # don't set to false
SHOW_FRAMES = False
T_FRAMES = 20  # time between frames showing when SHOW_FRAMES = True
T_FRAMES = round(T_FRAMES / (INTERPOLATE + 1))  # account for when using interpolation -
INTERPOLATION_METHOD = 'ABME'

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    frame_t1 = imagefile
    if INTERPOLATE:
        # track on interpolated frame before tracking on new frame
        int_imagefile = get_interpolated_filename(frame_t0, frame_t1, INTERPOLATION_METHOD)
        int_image = cv2.imread(int_imagefile)
        m_int, pred_bbox_int, refined_bbox_int, mask_bbox_int, confidence_int, best_idx_int = tracker.track(int_image)

    image = cv2.imread(imagefile)
    m, pred_bbox, refined_bbox, mask_bbox, confidence, best_idx = tracker.track(image)
    if SHOW_FRAMES:
        if INTERPOLATE:
            masked_int_img = np.where(m_int[..., None], color, int_image)
            out_int = cv2.addWeighted(int_image, 0.6, masked_int_img, 0.4, 0)
            cv2.imshow('test', out_int)
            cv2.waitKey(T_FRAMES)
        masked_img = np.where(m[..., None], color, image)
        out = cv2.addWeighted(image, 0.6, masked_img, 0.4, 0)
        cv2.imshow('test', out)
        cv2.waitKey(T_FRAMES)

    handle.report(m, confidence)

    # shift frames for next iteration
    frame_t0 = frame_t1
