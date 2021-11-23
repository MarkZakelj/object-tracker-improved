import cv2
import os
from pathlib import Path

SEQUENCE_DIR = 'my_workspace/sequences'


def main():
    for seq in Path(SEQUENCE_DIR).iterdir():
        if not seq.is_dir():
            continue
        print(seq)
        video = cv2.VideoCapture(os.path.join(str(seq), 'color', "%08d.jpg"), cv2.CAP_IMAGES)
        ok, frame = video.read()
        with Path(seq, 'groundtruth.txt').open() as f:
            line = f.readline()
            nums = [int(float(e)) for e in line.split(',')]
            left = min(nums[::2])
            top = min(nums[1::2])
            right = max(nums[::2])
            bottom = max(nums[1::2])
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=3)
            cv2.imshow('test', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
