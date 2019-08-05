from classes import *
from cv2 import *
from math import ceil


def call_ndpi_ndpa(filename):
    '''
    Filename is the name of the ndpi file.
    OpenSlide requires the file to be in the cwd.

    '''
    ndpi_image = NDPImage(filename)
    image_annotation_list = ImageAnnotationList(ndpi_image, filename + ".ndpa")
    return ndpi_image, image_annotation_list


# TODO
def split_ndpi(ndp_image, image_annotation_list, height, width):
    '''
    splits image into smaller, easier to handle images

    '''
    n_hor = ceil(ndp_image.width_lvl_0 / width)
    n_ver = ceil(ndp_image.height_lvl_0 / height)

    pass


def split_ndpa(image_annotation_list, height, width):

    merged = image_annotation_list.merge_annotations()
    n_hor = ceil(image_annotation_list.ndp_image.width_lvl_0 / width)
    n_ver = ceil(image_annotation_list.ndp_image.height_lvl_0 / height)

    print("size horizontal: " + str(image_annotation_list.ndp_image.width_lvl_0))
    print("size vertical: " + str(image_annotation_list.ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    for h in range(n_ver):
        for w in range(n_hor):
            reg = merged.extract_region(square_top_left_corner=(w*width,
                                                                h*height),
                                        square_height=height,
                                        square_width=width)
            filename = image_annotation_list.ndp_image.filename
            filename = "mask_" + filename.replace(".ndpi", "") + "_w{}_h{}.tif".format(w*width, h*height)
            save_mask_as_img(reg, "../split/" + filename)


def to_hsv(image):
    '''
    requires original image to come in BGR. NDPI images come like this
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def main():
    print(os.getcwd()) # borrar
    os.chdir("data/raw")
    filename = "S04_292_p16_RTU_ER1_20 - 2016-04-12 15.42.13.ndpi"
    ndpi_image, image_annotation_list = call_ndpi_ndpa(filename)
    split_ndpa(image_annotation_list, 1280, 1280)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
