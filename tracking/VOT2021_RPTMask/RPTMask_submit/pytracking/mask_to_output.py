import os

import cv2
import numpy as np


def save_mask(mask, mask_real, segm_crop_sz, bb, img_w, img_h, masks_save_path, sequence_name, frame_name):
    if mask is not None:
        M_sel = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
        mask_resized = (cv2.resize((M_sel * mask_real).astype(np.float32), (segm_crop_sz, segm_crop_sz),
                                   interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        mask_resized = (cv2.resize(mask_real.astype(np.float32), (segm_crop_sz, segm_crop_sz),
                                   interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    image_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # patch coordinates
    xp0 = 0
    yp0 = 0
    xp1 = mask_resized.shape[0]
    yp1 = mask_resized.shape[1]
    # image coordinates
    xi0 = int(round((bb[0] + bb[2] / 2) - mask_resized.shape[1] / 2))
    yi0 = int(round((bb[1] + bb[3] / 2) - mask_resized.shape[0] / 2))
    xi1 = int(round(xi0 + mask_resized.shape[1]))
    yi1 = int(round(yi0 + mask_resized.shape[0]))
    if xi0 < 0:
        xp0 = -1 * xi0
        xi0 = 0
    if xi0 > img_w:
        xp0 = (mask_resized.shape[1]) - (xi0 - img_w)
        xi0 = img_w
    if yi0 < 0:
        yp0 = -1 * yi0
        yi0 = 0
    if yi0 > img_h:
        yp0 = (mask_resized.shape[0]) - (yi0 - img_h)
        yi0 = img_h
    if xi1 < 0:
        xp1 = -1 * xi1
        xi1 = 0
    if xi1 > img_w:
        xp1 = (mask_resized.shape[1]) - (xi1 - img_w)
        xi1 = img_w
    if yi1 < 0:
        yp1 = -1 * yi1
        yi1 = 0
    if yi1 > img_h:
        yp1 = (mask_resized.shape[0]) - (yi1 - img_h)
        yi1 = img_h

    image_mask[yi0:yi1, xi0:xi1] = mask_resized[yp0:yp1, xp0:xp1]
    
    mask_save_dir = os.path.join(masks_save_path, sequence_name)
    if not os.path.exists(mask_save_dir):
        os.mkdir(mask_save_dir)
    mask_save_path = os.path.join(mask_save_dir, '%s.png' % frame_name)
    cv2.imwrite(mask_save_path, image_mask)
    
def show_mask(mask, mask_real, segm_crop_sz, bb, img_w, img_h, image):
    if mask is not None:
        M_sel = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
        mask_resized = (cv2.resize((M_sel * mask_real).astype(np.float32), (segm_crop_sz, segm_crop_sz),
                                   interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        mask_resized = (cv2.resize(mask_real.astype(np.float32), (segm_crop_sz, segm_crop_sz),
                                   interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    image_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # patch coordinates
    xp0 = 0
    yp0 = 0
    xp1 = mask_resized.shape[0]
    yp1 = mask_resized.shape[1]
    # image coordinates
    xi0 = int(round((bb[0] + bb[2] / 2) - mask_resized.shape[1] / 2))
    yi0 = int(round((bb[1] + bb[3] / 2) - mask_resized.shape[0] / 2))
    xi1 = int(round(xi0 + mask_resized.shape[1]))
    yi1 = int(round(yi0 + mask_resized.shape[0]))
    if xi0 < 0:
        xp0 = -1 * xi0
        xi0 = 0
    if xi0 > img_w:
        xp0 = (mask_resized.shape[1]) - (xi0 - img_w)
        xi0 = img_w
    if yi0 < 0:
        yp0 = -1 * yi0
        yi0 = 0
    if yi0 > img_h:
        yp0 = (mask_resized.shape[0]) - (yi0 - img_h)
        yi0 = img_h
    if xi1 < 0:
        xp1 = -1 * xi1
        xi1 = 0
    if xi1 > img_w:
        xp1 = (mask_resized.shape[1]) - (xi1 - img_w)
        xi1 = img_w
    if yi1 < 0:
        yp1 = -1 * yi1
        yi1 = 0
    if yi1 > img_h:
        yp1 = (mask_resized.shape[0]) - (yi1 - img_h)
        yi1 = img_h

    image_mask[yi0:yi1, xi0:xi1] = mask_resized[yp0:yp1, xp0:xp1]
    # import pdb
    # pdb.set_trace()
    
    # masktmp = (image_mask*255).astype(np.uint8)
    # mask_final = np.stack([masktmp, masktmp*255, masktmp]).transpose(1, 2, 0)
    # imagewithmask = cv2.addWeighted(image, 0.77, mask_final, 0.23, -1)
    # imagewithmask=cv2.cvtColor(imagewithmask, cv2.COLOR_RGB2BGR)
    # cv2.imshow('d3smask',imagewithmask)
    # cv2.waitKey(1)
    return image_mask
    
def box2mask(bb, image):
    
    # import pdb
    # pdb.set_trace()
    img_w, img_h=image.shape[0],image.shape[1]
    image_mask = np.zeros((img_w, img_h), dtype=np.uint8)
    x0,y0,w,h=bb
    if x0<0:
        w=w+x0
        x0=0
    if y0<0:
        h=h+y0
        y0=0
    mask_resized=np.ones((int(h), int(w)), dtype=np.uint8)
    y1=min(int(y0)+int(h),img_w)
    dy=int(y1)-int(y0)
    x1=min(int(x0)+int(w),img_h)
    dx=int(x1)-int(x0)
    if image_mask[int(y0):(int(y1)), int(x0):(int(x1))].shape[0] !=dy:
        # import pdb
        # pdb.set_trace()
        # a=1
        return image_mask
    if image_mask[int(y0):(int(y1)), int(x0):(int(x1))].shape[1] !=dx:

        # import pdb
        # pdb.set_trace()
        # a=1
        return image_mask
    image_mask[int(y0):(int(y1)), int(x0):(int(x1))] = mask_resized[0:dy,0:dx]
    # image_mask[int(y0):(int(y0)+int(h)), int(x0):(int(x0)+int(w))] = mask

    # import pdb
    # pdb.set_trace()
    
    # masktmp = (image_mask*255).astype(np.uint8)
    # mask_final = np.stack([masktmp, masktmp*255, masktmp]).transpose(1, 2, 0)
    # imagewithmask = cv2.addWeighted(image, 0.77, mask_final, 0.23, -1)
    # imagewithmask=cv2.cvtColor(imagewithmask, cv2.COLOR_RGB2BGR)
    # cv2.imshow('d3smask',imagewithmask)
    # cv2.waitKey(1)
    return image_mask