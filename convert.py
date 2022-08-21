from vot.region import io
from vot.region import shapes
import cv2
import os
import shutil
from pathlib import Path

img_dir = 'black_circles'
mask_dir = 'black_circles_masks'

base_img = cv2.imread(os.path.join(mask_dir, 'base.jpg'), 0)
base_mask = shapes.Mask(base_img)
for mask_name in os.listdir(mask_dir):
    if mask_name.startswith('base'):
        continue
    moved_img = cv2.imread(os.path.join(mask_dir, mask_name), 0)
    moved_mask = shapes.Mask(moved_img)
    sequence_name = mask_name.rstrip('.jpg')
    sequence_dir = os.path.join('TransT_M/votspace/sequences', sequence_name)
    Path(sequence_dir).mkdir(exist_ok=True)
    Path(sequence_dir, 'color').mkdir(exist_ok=True)
    io.write_trajectory(os.path.join(sequence_dir, 'groundtruth.txt'), [base_mask, moved_mask])

    with open(os.path.join(sequence_dir, 'anchor.value'), 'w') as fp:
        print('1.0', file=fp)
        print('0.0', file=fp)

    with open(os.path.join(sequence_dir, 'sequence'), 'w') as fp:
        fp.writelines(['channels.color=color/%08d.jpg\n',
                       'format=default\n',
                       'fps=30\n',
                      f'name={sequence_name}\n'])

    shutil.copyfile(os.path.join(img_dir, 'base.jpg'), os.path.join(sequence_dir, 'color', '00000001.jpg'))
    shutil.copyfile(os.path.join(img_dir, mask_name), os.path.join(sequence_dir, 'color', '00000002.jpg'))


