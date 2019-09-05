import os
import shutil
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from cv2 import cvtColor, COLOR_RGB2HSV, normalize, CV_32F, NORM_MINMAX


def create_directory(*args):
    '''
    *args are strings corresponding to directories
    '''
    for directory in args:
        os.mkdir(os.path.dirname(directory))


def build_dirs():
    try:
        shutil.rmtree("data/split/X/")
    except:
        pass

    create_directory("data/split/X/")
    create_directory("data/split/X/-1/",  # background
                     "data/split/X/1/",  # epithelium
                     "data/split/X/0/")  # non-epithelium


def list_files_from_dir(directory="", extension=".ndpi"):
    # TODO> pasar a "utils.py"
    '''
    lists files of extension <extension> in directory.
    It also returns the path relative to the inputed directory
    '''
    # TODO> resolver el bug... cuando corro preprocess la variable glb
    #       tiene que tener "**/*" pero para q funcione con train.py tiene
    #       tiene que tener "/**/*"

    glb = glob.glob(directory + "/**/*" + extension, recursive=True)

    file_list = [os.path.basename(f) for f in glb]
    dir_list = [os.path.dirname(f).replace(directory + "\\", "") for f in glb]
    return file_list, dir_list


def save_np_as_image(np_array, filename):
    im = Image.fromarray(np.uint8(np_array))
    im.save(filename)
    return


def extract_region(np_array, square_top_left_corner, square_height,
                   square_width, draw_region=False):

    '''
    square_top_left_corner: (x,y) tuple
    square_height and square_width: height and width of ROI
    draw_region: if true, the extracted ROI will be previewed
    '''
    region = np_array[square_top_left_corner[1]:
                      square_top_left_corner[1]+square_height,
                      square_top_left_corner[0]:
                      square_top_left_corner[0]+square_width]

    if draw_region:
        plt.figure()
        plt.imshow(region*255, aspect='auto')
        plt.show()

    return region


def to_hsv(image):
    # TODO pasar a "utils.py"
    '''
    Image corresponds to a numpy array.
    function equires original image to come in RGB.
    NDPI images extracted with Openslide come in RGBA,
    so the last channel is dismissed.
    '''
    return cvtColor(image, COLOR_RGB2HSV)


def normalize_image(image):
    # TODO pasar a "utils.py"
    '''
    image corresponds to a numpy array.
    '''
    return normalize(image, None, alpha=0., beta=1.,
                     dtype=CV_32F, norm_type=NORM_MINMAX)


def train_validation_test_partition(file_list, prop=(0.6, 0.4, 0.0)):
    lf = len(file_list)
    indexes = np.arange(lf)
    np.random.shuffle(indexes)
    train_list = [file_list[indexes[i]]
                  for i in range(0, floor(prop[0]*lf))]
    val_list = [file_list[indexes[i]]
                for i in range(floor(prop[0]*lf),
                               floor((prop[0]+prop[1])*lf))]
    test_list = [file_list[indexes[i]]
                 for i in range(floor((prop[0]+prop[1])*lf),
                                floor((prop[0]+prop[1]+prop[2])*lf))]
    return train_list, val_list, test_list


def call_ndpi_ndpa(filename):
    '''
    filename is the name of the ndpi file.
    OpenSlide requires the file to be in the cwd.
    the annotation must end in ".ndpi.ndpa".
    '''
    ndp_image = NDPImage(filename)
    image_annotation_list = ImageAnnotationList(ndp_image, filename + ".ndpa")
    return ndp_image, image_annotation_list
