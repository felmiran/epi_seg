# epi_seg
Thesis: Automatic segmentation of epithelium in cervical whole slide images

TODO

nvidia GeForce GTX 960M, driver 417.71
CUDA version: 10.0.130
cuDNN version: 7.4.1 for CUDA 10


Installing dependencies (windows 10):
- install anaconda 
- install gitbash
- (in root directory of script) run setup.sh ('./setup.sh')


Openslide is used for handling of NDP images and annotations. These can be downloaded in https://openslide.org/download/. This project uses the 64-bit windows binary from 2017-11-22. Add bin to path and you are done. 

NOTE 1: apparently there was a dependency issue between an openslide and a cv2 dll file. I do not remember how I fixed it, but the openslide package in this project works.
NOTE 2: the openslide-python package requires Microsoft Visual C++ v14.0 to run. Be sure to download it.

Common error: https://github.com/openslide/openslide-python/issues/23

