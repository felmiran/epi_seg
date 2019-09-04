import os
from classes import *
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_RGB2GRAY, COLOR_BGR2RGB, \
    normalize, CV_32F, NORM_MINMAX, imread, calcHist, transpose, flip
from math import ceil, floor
import numpy as np
import json
import shutil
import glob
from tqdm import tqdm

# TODO> rename to "split.py"
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


def list_files_from_dir(directory="", extension=".ndpi"):
    # TODO> pasar a "utils.py"
    '''
    lists files of extension <extension> in directory.
    It also returns the path relative to the inputed directory
    '''
    # TODO> resolver el bug... cuando corro preprocess la variable glb 
    #       tiene que tener "**/*" pero para q funcione con train.py tiene 
    #       tiene que tener "/**/*"
            
    glb = glob.glob(directory + "**/*" + extension, recursive=True)

    file_list = [os.path.basename(f) for f in glb]
    dir_list = [os.path.dirname(f).replace(directory + "\\", "") for f in glb]
    return file_list, dir_list


def data_augmentation(directory="../split/X/1", flip_imgs=True):
    '''
    generates extra images for eah image in directory: 3 addicional
    images corresponding to 90, 180, 270 rotations.
    if flip=True the flipped version of the 4 images (original + 3)
    will be created as well.
    '''
    # TODO: Add tqdm https://www.youtube.com/watch?v=qVHM3ly-Amg (already installed in isic)
    labels = {}

    print("Data augmentation commencing for images labelled 'epithelium'.")
    print("flip_imgs={}".format(flip_imgs))

    file_list, _ = list_files_from_dir(directory=directory, extension=".tif")
    for img_name in tqdm(file_list):

        img = imread(directory + "/" + img_name)

        img_90, labels["90_" + img_name] = flip(transpose(img), 1), 1
        save_np_as_image(cvtColor(img_90, COLOR_BGR2RGB),
                         directory + "/90_" + img_name)

        img_180, labels["180_" + img_name] = flip(img, -1), 1
        save_np_as_image(cvtColor(img_180, COLOR_BGR2RGB),
                         directory + "/180_" + img_name)

        img_270, labels["270_" + img_name] = flip(transpose(img), 0), 1
        save_np_as_image(cvtColor(img_270, COLOR_BGR2RGB),
                         directory + "/270_" + img_name)

        if flip_imgs:
            img_f, labels["f_" + img_name] = flip(img, 1), 1
            save_np_as_image(cvtColor(img_f, COLOR_BGR2RGB),
                             directory + "/f_" + img_name)

            img_90f, labels["90f_" + img_name] = flip(img_90, 1), 1
            save_np_as_image(cvtColor(img_90f, COLOR_BGR2RGB),
                             directory + "/90f_" + img_name)

            img_180f, labels["180f_" + img_name] = flip(img_180, 1), 1
            save_np_as_image(cvtColor(img_180f, COLOR_BGR2RGB),
                             directory + "/180f_" + img_name)

            img_270f, labels["270f_" + img_name] = flip(img_270, 1), 1
            save_np_as_image(cvtColor(img_270f, COLOR_BGR2RGB),
                             directory + "/270f_" + img_name)

    json.dump(labels, open("../split/X/augmentations.txt", "w"))

    pass


def clean_split_files(directory="../split", lista=""):
    '''
    deletes files where mask has no pixel in annotated region
    default directory is "split" from the "raw" folder
    '''
    if lista == "":
        mask_list = os.listdir(directory + "/mask")
        mask_list.remove(".gitkeep")
    else:
        mask_list = [x for x in lista]

    for filename in mask_list:
        mask = imread(directory + "/mask/" + filename)[:, :, 0]

        if np.amax(mask) == 0:
            os.remove(directory + "/mask/" + filename)
            os.remove(directory + "/X/" + filename)


def call_ndpi_ndpa(filename):
    # TODO pasar a "utils.py"
    '''
    filename is the name of the ndpi file.
    OpenSlide requires the file to be in the cwd.
    the annotation must end in ".ndpi.ndpa".
    '''
    ndp_image = NDPImage(filename)
    image_annotation_list = ImageAnnotationList(ndp_image, filename + ".ndpa")
    return ndp_image, image_annotation_list

# TODO> registrar para la tesis como se llego a esta funcion tile_is_background
# 1. Se probo con una funcion que calculaba si el % de pixeles entre
#    220 y 240 era mayor a 90% (usando calcHist). Sin embargo, se observo que
#    en la clase 0 tambien habian tiles que eran claramente background,
#    solo que eran un poco mas oscuras.
# 2. Aunque eran mas oscuras, se observaba que la std del histograma era menor
#    en imagenes que son background. Viendo como se distribuye esta std entre
#    tiles (separandolas con el metodo 1) se observo una separacion clara en
#    std=5. Con esa info se hizo la segundafuncion tile_is_background_2.


def tile_is_background_2(image, threshold=5):
    '''
    returns True if tile (np array) is background. An <image> is classified
    as background if std dev of pixel colors (gray scale, 0-255) is over
    <threshold>.
    '''
    is_bkgnd = False
    std = np.std(cvtColor(image, COLOR_RGB2GRAY))
    if std < threshold:
        is_bkgnd = True
    return std, is_bkgnd


def tile_is_background_1(image, rng=(220, 240), threshold=0.9):
    '''
    returns True if tile (np array) is background. An <image> is classified
    as background if the proportion of pixels colors (gray scale, 0-255) 
    within <rng> is over <threshold>.

    inputs:
     - image: numpy array corresponding to RGB image
     - rng: range of values to evaluate in histogram
     - threshold: if (pixels in rng / total pixels)
                is higher than threshold, images is classified as background
    '''
    is_bkgnd = False
    hist = calcHist(images=[cvtColor(image, COLOR_RGB2GRAY)],
                    channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    if np.sum(hist[rng[0]:rng[1]])/np.sum(hist) > 0.9:
        is_bkgnd = True
    return hist, is_bkgnd


# TODO> TESTS:
# 1- el nombre de los archivos debe ser igual en X y en mask
# 2- el tamano de los archivos debe ser igual para cada nombre
# 3- los archivos ndpi tienen que tener el formato necesario
# 4- validar que los labels se guarden correctamente, y
#    sobreescriban el archivo antetiot
# 5- test if 5000 por imagen son background. (quizás sea mejor separarlas por carpeta)
# *- (borrar los ?xml y ponerle id a los ndpviewstate)


def rectangle_split_ndpi_ndpa(ndp_image, image_annotation_list, split_height,
                              split_width, tohsv=False, path_ndpi="../split/X",
                              path_ndpa="../split/mask", n_bkgnd_tiles=5000):
    '''
    Splits ndpi into tiles and saves lanel as dict in "labels.txt".
    Keys are filenames.
    '''
    merged = image_annotation_list.merge_annotations().mask

    width, height = split_width, split_height
    size_hor, size_ver = ndp_image.width_lvl_0, ndp_image.height_lvl_0
    n_hor, n_ver = ceil(size_hor / width), ceil(size_ver / height)

    filename = ndp_image.filename
    print("filename: " + filename)

    print("size horizontal: " + str(ndp_image.width_lvl_0))
    print("n of horizontal splits: " + str(n_hor))
    print("size vertical: " + str(ndp_image.height_lvl_0))
    print("n of vertical splits: " + str(n_ver))

    labels = {}
    bkgnd_tiles_counter = 0
    tile_class = "0"

    for h in tqdm(range(n_ver)):
        if h == n_ver-1:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):
            if w == n_hor-1:
                width = size_hor - (n_hor - 1) * split_width

            reg_ndpi = np.array(ndp_image.read_region(location=(w * width,
                                                                h * height),
                                                      level=0, size=(width,
                                                                     height)
                                                      ))[:, :, :3]

            reg_ndpa = extract_region(merged,
                                      square_top_left_corner=(w * width,
                                                              h * height),
                                      square_height=height, square_width=width)

            dimensions = "_({},{})_{}x{}".format(w * width, h * height, width,
                                                 height)
            split_filename = filename.replace(".ndpi", "")
            split_filename = split_filename + dimensions + ".tif"

            # _, is_bkgnd = tile_is_background_1(reg_ndpi, rng=(220, 240),
            #                                    threshold=0.9)

            _, is_bkgnd = tile_is_background_2(reg_ndpi, threshold=5)

            if is_bkgnd:
                if bkgnd_tiles_counter < n_bkgnd_tiles:
                    labels[split_filename] = 2
                    bkgnd_tiles_counter += 1
                    tile_class = "-1"
                else:
                    continue
            else:
                if np.sum(reg_ndpa) == height * width:
                    labels[split_filename] = 1
                    tile_class = "1"
                else:
                    labels[split_filename] = 0
                    tile_class = "0"

            if tohsv:
                reg_ndpi = to_hsv(reg_ndpi)

            save_np_as_image(reg_ndpi, path_ndpi + "/" + tile_class +
                             "/" + split_filename)

    json.dump(labels, open(path_ndpi + "/" + filename + ".txt", "w"))


def rectangle_split_ndpi(ndp_image, split_width, split_height,
                         tohsv=False, path="../split/X"):
    '''
    Splits image into smaller, easier to handle images

    input:
    - ndp_image: object of class NDPImage
    - width:
    - height:
    - lvl:
    - tohsv:

    Observations:
    - Images are saved in the "../split/X" folder with .tif extension

    '''
    width = split_width
    height = split_height

    size_hor = ndp_image.width_lvl_0
    n_hor = ceil(size_hor / width)

    size_ver = ndp_image.height_lvl_0
    n_ver = ceil(size_ver / height)

    filename = ndp_image.filename

    print("size horizontal: " + str(ndp_image.width_lvl_0))
    print("size vertical: " + str(ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    lvl = 0

    for h in range(n_ver):
        if h == n_ver-1:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):
            if w == n_hor-1:
                width = size_hor - (n_hor - 1) * split_width

            reg = np.array(ndp_image.read_region(location=(w * width,
                                                           h * height),
                                                 level=lvl,
                                                 size=(width, height))
                           )[:, :, :3]

            if tohsv:
                reg = to_hsv(reg)

            dimensions = "_({},{})_{}x{}".format(w*width, h*height,
                                                 width, height)
            split_filename = filename.replace(".ndpi", "") + dimensions

            save_np_as_image(reg, path + "/" + split_filename + ".tif")


def rectangle_split_ndpa(image_annotation_list, split_width,
                         split_height, value_ones=1, path="../split/mask"):

    '''
    mask from image_annotation_list is expected to be 0s and 1s
    '''

    width = split_width
    height = split_height

    merged = image_annotation_list.merge_annotations().mask * value_ones

    size_hor = image_annotation_list.ndp_image.width_lvl_0
    n_hor = ceil(size_hor / width)

    size_ver = image_annotation_list.ndp_image.height_lvl_0
    n_ver = ceil(size_ver / height)

    filename = image_annotation_list.ndp_image.filename

    print("size horizontal: " +
          str(image_annotation_list.ndp_image.width_lvl_0))
    print("size vertical: " +
          str(image_annotation_list.ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    for h in range(n_ver):
        if h == n_ver-1:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):
            if w == n_hor-1:
                width = size_hor - (n_hor - 1) * split_width

            reg = extract_region(merged,
                                 square_top_left_corner=(w*width, h*height),
                                 square_height=height,
                                 square_width=width)
            dimensions = "_({},{})_{}x{}".format(w*width,
                                                 h*height,
                                                 width,
                                                 height)
            filename = filename.replace(".ndpi", "") + dimensions
            save_np_as_image(reg, path + "/" + filename + ".tif")


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


def main(clean=False):

    '''
    for now, this function grabs a ndpi image, splits the image and the mask
    and saves the splits in the split directory.
    '''
    # os.chdir("data/raw")
    os.chdir("data/test")

    # archivo = ["S04_3441_p16_RTU_ER1_20 - 2016-04-12 15.45.38.ndpi"]
    # for ndpi_file in archivo:

    file_list, _ = list_files_from_dir(extension=".ndpi")
    print(file_list)

    for ndpi_file in file_list:
        print(ndpi_file)
        ndp_image, image_annotation_list = call_ndpi_ndpa(ndpi_file)
        width, height = floor(ndp_image.width_lvl_0/4), floor(ndp_image.height_lvl_0/4)
        rectangle_split_ndpi(ndp_image=ndp_image,
                             split_width=width,
                             split_height=height,
                             tohsv=False,
                             path="grandes_RGB")

        # width, height = 128, 128
        # rectangle_split_ndpi_ndpa(ndp_image=ndp_image,
        #                           image_annotation_list=image_annotation_list,
        #                           split_height=height,
        #                           split_width=width,
        #                           tohsv=False,
        #                           path_ndpi="../split/X",
        #                           path_ndpa="../split/mask")

    data_augmentation()

    if clean:
        clean_split_files()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # build_dirs()

    main(clean=False)
