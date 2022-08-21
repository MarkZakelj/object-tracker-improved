import cv2
import os

img_dir = 'black_circles'
mask_dir = 'black_circles_masks'
for img_name in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir, img_name), 0)
    binary_image = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imwrite(os.path.join(mask_dir, img_name), binary_image)
    # print(binary_image)
    # cv2.imshow('a', binary_image)
    # cv2.waitKey(0)
