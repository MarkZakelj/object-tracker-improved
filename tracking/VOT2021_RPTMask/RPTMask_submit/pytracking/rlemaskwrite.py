import cv2
import numpy as np
def mask_to_rle(m):
    """
    # Input: 2-D numpy array
    # Output: list of numbers (1st number = #0s, 2nd number = #1s, 3rd number = #0s, ...)
    """
    # reshape mask to vector
    v = m.reshape((m.shape[0] * m.shape[1]))
    if v.size == 0:
        return[0]
    # output is empty at the beginning
    rle = []
    # index of the last different element
    last_idx = 0
    # check if first element is 1, so first element in RLE (number of zeros) must be set to 0
    if v[0] > 0:
        rle.append(0)

    # go over all elements and check if two consecutive are the same
    for i in range(1, v.size):
        if v[i] != v[i - 1]:
            rle.append(i - last_idx)
            last_idx = i

    if v.size > 0:
        # handle last element of rle
        if last_idx < v.size - 1:
            # last element is the same as one element before it - add number of these last elements
            rle.append(v.size - last_idx)
        else:
            # last element is different than one element before - add 1
            rle.append(1)

    return rle
    
def mask2bbox(mask):
    """
    mask: 2-D array with a binary mask
    output: coordinates of the top-left and bottom-right corners of the minimal axis-aligned region containing all positive pixels
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rows_i = np.where(rows)[0]
    cols_i = np.where(cols)[0]
    if len(rows_i) > 0 and len(cols_i) > 0:
        rmin, rmax = rows_i[[0, -1]]
        cmin, cmax = cols_i[[0, -1]]
        return (cmin, rmin, cmax, rmax)
    else:
        # mask is empty
        return (None, None, None, None)
        
def encode_mask(mask):
    """
    mask: input binary mask, type: uint8
    output: full RLE encoding in the format: (x0, y0, w, h), RLE
    first get minimal axis-aligned region which contains all positive pixels
    extract this region from mask and calculate mask RLE within the region
    output position and size of the region, dimensions of the full mask and RLE encoding
    """
    # calculate coordinates of the top-left corner and region width and height (minimal region containing all 1s)
    x_min, y_min, x_max, y_max = mask2bbox(mask)

    # handle the case when the mask empty
    if x_min is None:
        return (0, 0, 0, 0), [0]
    else:
        tl_x = x_min
        tl_y = y_min
        region_w = x_max - x_min + 1
        region_h = y_max - y_min + 1

        # extract target region from the full mask and calculate RLE
        # do not use full mask to optimize speed and space
        target_mask = mask[tl_y:tl_y+region_h, tl_x:tl_x+region_w]
        rle = mask_to_rle(np.array(target_mask))

        return (tl_x, tl_y, region_w, region_h), rle

def mask_mask_overlap(m1, m2):
    m_inter_sum = np.sum(m1 * m2)
    union_sum = np.sum(m1) + np.sum(m2) - m_inter_sum
    return float(m_inter_sum) / float(union_sum) if union_sum > 0 else float(0)

def masktostring(mask):
    tmpline=encode_mask(mask)
    offset_str='%d,%d' % (tmpline[0][0],tmpline[0][1])
    region_sz_str = '%d,%d' % (tmpline[0][2], tmpline[0][3])
    rle_str = ','.join([str(el) for el in tmpline[1]])
    textline='m%s,%s,%s\n' % (offset_str, region_sz_str, rle_str)
    return textline

def main():    
    mask = cv2.imread('./mask.png', cv2.IMREAD_GRAYSCALE)
    # mask.max() needs to be 1
    mask=mask/255
    # import pdb
    # pdb.set_trace()
    textline=masktostring(mask)
    maskzero= np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8) 
    import pdb
    pdb.set_trace()

    print(mask_mask_overlap(mask,mask))

    video_path='./'
    video_name='text'
    import os
    result_path = os.path.join(video_path, '{}_001.txt'.format(video_name))
    with open(result_path, 'w') as f:
        if isinstance(textline, str):
            f.write("{}".format(textline))

if __name__ == '__main__':
    main()