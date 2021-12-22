import cv2
import main_utils as utils
import numpy as np

def main():
    masks = utils.load_masks_raster('tiger')
    a = masks[0]
    x = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) * 255
    print(x.max())


if __name__ == '__main__':
    main()
