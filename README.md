# object-tracker-improved
Improve object tracking with Video interpolation techniques

# Installation
Prerequisites:
 - 8GB nvidia gpu
 - python conda

First install interpolation algorithms each one into separate environment, as specified in the repositories.
Create new folder `interpolation` and clone the following repos inside:
 - https://github.com/JunHeum/ABME
 - https://github.com/megvii-research/ECCV2022-RIFE

download the pretrained models into proper directories inside repos.

Then install object tracking algorithms (into root) from https://www.votchallenge.net/vot2021/trackers.html, namely
 - TransT_M
 - RPTMask

and install dependencies each into separate python environment.

Create VOT workspace and initialize it to vot2021.

In `main_utils.py` set up workspace directory.

## Experiments

Run `interpolation_main.py` twice, each time with a different interpolation algorithm.

Then run each object tracking algorithm on interpolated and non-interpolated sequences.



