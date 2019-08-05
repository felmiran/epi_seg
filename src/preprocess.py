from classes import *


def call_ndpi_ndpa(filename):
    '''
    Filename is the name of the ndpi file.
    OpenSlide requires the file to be in the cwd.

    '''
    ndpi_image = NDPImage(filename)
    image_annotation_list = ImageAnnotationList(ndpi_image, filename)
    return ndpi_image, image_annotation_list


# TODO
def split_ndpi_ndpa(ndp_image, image_annotation_list, height, width):
    '''
    splits image into smaller, easier to handle images

    '''
    merged = image_annotation_list.merge_annotations()

    n_hor = ndp_image.

    pass