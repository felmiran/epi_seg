import sys
# sys.path.append('D:/felipe/software_projects/epi_seg/openslide/openslide-win64-20171122/bin')

import xml.etree.ElementTree as ET
from openslide import OpenSlide
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


class NDPImage(OpenSlide):

    '''
    Ndpi image object.
    '''

    def __init__(self, filename):
        super().__init__(filename)
        self.filename = filename
        self.offset_x, self.offset_y, self.mpp_x, \
            self.mpp_y, self.width_lvl_0, \
            self.height_lvl_0 = self._get_image_properties()

    def _get_image_properties(self):

        '''
            Returns prooperties that are useful for constructing annotations
            from NDPA files:

            -  hamamatsu.XOffsetFromSlideCentre:
            -  hamamatsu.YOffsetFromSlideCentre:
            -  openslide.mpp-x:
            -  openslide.mpp-y:
            -  openslide.level[0].width: image width in its max
                resolution level (usually x40)
            -  openslide.level[0].height: image height in its max
                resolution level (usually x40)
        '''

        offset_x = float(self.properties['hamamatsu.XOffsetFromSlideCentre'])
        offset_y = float(self.properties['hamamatsu.YOffsetFromSlideCentre'])
        mpp_x = float(self.properties['openslide.mpp-x'])
        mpp_y = float(self.properties['openslide.mpp-y'])
        width_lvl_0 = int(self.properties['openslide.level[0].width'])
        height_lvl_0 = int(self.properties['openslide.level[0].height'])
        return offset_x, offset_y, mpp_x, mpp_y, width_lvl_0, height_lvl_0

    def print_image_properties(self):
        '''
        Utility method to view all NDPI properties
        '''

        for i in self.properties:
            print(i + ": " + self.properties[i])
        return


class ImageAnnotationList:
    '''
        Esta clase es el set de anotaciones de una imagen
        (esta imagen esta representada por el filename)
    '''

    def __init__(self, ndp_image, annotation_path):
        self.ndp_image = ndp_image
        self.annotation_path = annotation_path
        self.annotation_list = self._create_annotations()

    def _create_annotations(self):

        '''
         funcion para crear las anotaciones con su id, path, image_path y
         puntos.

         Creates a list of annotations for the NDPI.

         input:
         -  ndp_image: the NDPI corresponding to the annotations
         -  annotation_path: path for the NDPA file

         returns:
         -  List of annotation objects containing:
                -  annotation_id: as described in the NDPA file
                -  annotation_title: as described in the NDPA file
                -  annotation_path: el path del archivo del que se saco la
                   anotacion
                -  image_path: ubicacion de la imagen a la que esta asociada
                   la anotacion
                -  points: la lista con las tuplas de puntos para la anotacion
        '''

        # TODO> reemplazar los loops
        annotations = []

        ndpa = open(self.annotation_path)
        tree = ET.parse(ndpa)

        for ndpviewstate in tree.findall('ndpviewstate'):

            annotation_title = ndpviewstate.find('title').text
            annotation_id = ndpviewstate.get('id')
            pointlist = ndpviewstate.find('annotation').find('pointlist')
            points = []

            for p in pointlist.findall('point'):
                point_x = float(p.find('x').text)
                point_y = float(p.find('y').text)
                point = NDPAnnotationPoint(point_x, point_y, self.ndp_image)

                points.append(point.point_from_physical_to_pixels())

            annotation = Annotation(annotation_id=annotation_id,
                                    annotation_title=annotation_title,
                                    annotation_path=self.annotation_path,
                                    ndp_image=self.ndp_image,
                                    points=points)

            annotations.append(annotation)

        return annotations

    def merge_annotations(self, draw_mask=False):
        '''
        funcion para mergear las anotaciones por ID de anotacion
        inputs:
         - draw_mask: if True, a figure of the mask will be displayed

        outputs:
         - merged: corresponds to the mask that contains all annotations
           of an image
        '''

        masks = [a.get_mask() for a in self.annotation_list]

        merged_annotation = reduce(np.add, masks, 0)
        merged_annotation[merged_annotation > 0] = 1

        if draw_mask:
            plt.figure()
            plt.imshow(merged_annotation*255, aspect='auto')
            plt.show()

        merged = Annotation(annotation_id="merged",
                            annotation_title="merged",
                            annotation_path=self.annotation_path,
                            ndp_image=self.ndp_image)

        merged.set_mask(merged_annotation)

        return merged


class Annotation:
    def __init__(self, annotation_id=None, annotation_title=None,
                 annotation_path=None, ndp_image=None, points=[]):
        # cuando termine de armar esto, deberia dejar como obligatorios
        # todos los campos
        self.annotation_id = annotation_id
        self.annotation_title = annotation_title
        self.annotation_path = annotation_path
        self.ndp_image = ndp_image
        self.points = points

    def count_points(self):
        return len(self.points)

    def print_points(self):
        for i in self.points:
            print(i)
        return

    def get_mask(self, draw_mask=False):

        '''
            Retorna una mascara de la imagen completa, con la region de la
            anotacion marcada en 1s y el resto en 0s.
            output:
                - np.array con mascara
        '''

        polygon = self.points
        width = self.ndp_image.width_lvl_0
        height = self.ndp_image.height_lvl_0

        img = Image.new('L', (width, height), 0)

        ImageDraw.Draw(img).polygon(polygon, outline=0, fill=1)
        mask = np.array(img)

        if draw_mask:
            plt.figure()
            plt.imshow(mask*255, aspect='auto')
            plt.show()

        self.mask = mask

        return mask

    def set_mask(self, mask):
        if hasattr(self, "mask"):
            print("Annotation already has a mask")
        else:
            self.mask = mask
        return

    def draw_mask(self):
            plt.figure()
            plt.imshow(self.mask*255, aspect='auto')
            plt.show()

    def get_mask_area(self):
        # TODO
        # Te retorna el numero de pixeles contenidos en el area.
        # Por ahora incluye el borde, aunque quizas no deberia hacerlo.
        # no se para que me puede servir, pero igual.
        # pendiente: hacer un try except, que depende de si esta o no generada
        # la mask

        try:
            pass
        except expression as identifier:
            pass

        return

    def is_inside_region(self, square_top_left_corner, square_height,
                         square_width):
        # TODO

        '''
        Returns True if a square section is inside of the annotated region
        '''

        # Esta funcion dice si el tile esta o no adentro de la region,
        # a partir de las coordenadas de cada una

        # pendiente: hacer un try except, que depende de si esta o no
        # generada la mask
        return


class Point:
    def __init__(self, x, y):
        self.coord = (x, y)

    def __str__(self):
        return 'x: ' + str(self.coord[0]) + ', y: ' + str(self.coord[1])


class NDPAnnotationPoint(Point):
    def __init__(self, x, y, ndp_image):
        super().__init__(x, y)
        self.ndp_image = ndp_image

    def point_from_physical_to_pixels(self):
        '''
         funcion para pasar los puntos desde coordenadas fisicas tomando el
         escaneo completo a pixeles de la imagen de interes.

         input:
         -  punto (tupla x,y)
         -  image properties object, que contiene:
                -  offset_x
                -  offset_y
                -  mpp_x
                -  mpp_y
                -  width_lvl_0
                -  height_lvl_0

         returns:
         -  tupla en pixeles
        '''

        image_prop_obj = self.ndp_image

        # coordenada del punto (0,0) de la imagen de interes, tomando como eje de referencia
        # el centro de la slide completa. se pasa la coordenada de nm a pixel
        slide_center_x = image_prop_obj.offset_x / (image_prop_obj.mpp_x * 1000)
        slide_center_y = image_prop_obj.offset_y / (image_prop_obj.mpp_y * 1000)
        
        x_0 = slide_center_x - image_prop_obj.width_lvl_0 / 2
        y_0 = slide_center_y - image_prop_obj.height_lvl_0 / 2

        # al restar las coordenadas del punto de interes con las del punto (0,0) de la imagen, 
        # se obtiene la coordenada del punto de interes respecto al eje (0,0)
        x = self.coord[0] / (image_prop_obj.mpp_x * 1000) - x_0
        y = self.coord[1] / (image_prop_obj.mpp_y * 1000) - y_0

        return (round(x), round(y))


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


def save_mask_as_img(numpy_array, filename):
    im = Image.fromarray(np.uint8(numpy_array))
    im.save(filename)
    return


def PIL_to_numpy(PIL_obj):
    pass
