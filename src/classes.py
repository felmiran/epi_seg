import sys
# sys.path.append('C:/Users/felipe/openslide-win64-20171122/bin')

import xml.etree.ElementTree as ET
import openslide
import os
from PIL import Image, ImageDraw
import numpy as np


class ImageProperties:
    def __init__(self,image_path):
        self.image_path = image_path
        self.offset_x, self.offset_y, self.mpp_x, \
            self.mpp_y, self.width_lvl_0, \
                self.height_lvl_0 = self._get_image_parameters()

    def _get_image_parameters(self):
         '''
          esta funcion tiene que retornar los siguientes datos:
          -  hamamatsu.XOffsetFromSlideCentre: 
          -  hamamatsu.YOffsetFromSlideCentre: 
          -  openslide.mpp-x: 
          -  openslide.mpp-y: 
          -  openslide.level[0].width: image width in its max 
             resolution level (usually x40)
          -  openslide.level[0].height: image height in its max 
             resolution level (usually x40)


            ESTO QUIZAS PODRIA SER UNA FUNCION NO MAS... 
            NO NECESITA SER UNA CLASE EN REALIDAD

         '''

         path, filename = os.path.split(self.image_path)

         os.chdir(path)

         slide = openslide.OpenSlide(filename)

         offset_x = float(slide.properties['hamamatsu.XOffsetFromSlideCentre'])
         offset_y = float(slide.properties['hamamatsu.YOffsetFromSlideCentre'])
         mpp_x = float(slide.properties['openslide.mpp-x'])
         mpp_y = float(slide.properties['openslide.mpp-y'])
         width_lvl_0 = int(slide.properties['openslide.level[0].width'])
         height_lvl_0 = int(slide.properties['openslide.level[0].height'])

         return offset_x, offset_y, mpp_x, mpp_y, width_lvl_0, height_lvl_0


class ImageAnnotationList:
    '''
        Esta clase es el set de anotaciones de una imagen 
        (esta imagen esta representada por el filename)
    '''
    def __init__(self, associated_image, annotation_path):
        self.associated_image = associated_image
        self.annotation_path = annotation_path
        self.annotation_list = self.create_annotations()

    def create_annotations(self):

        '''
         funcion para crear las anotaciones con su id, path, image_path y 
         puntos.

         input:
         -  image properties object
         -  annotation_path

         returns:
         -  lista de annotation objects, cada una de las cuales contiene:
                -  annotation_id: el id de la anotacion para esta imagen 
                -  annotation_title: el titulo (epithelium u otro)
                -  annotation_path: el path del archivo del que se saco la 
                   anotacion
                -  image_path: ubicacion de la imagen a la que esta asociada 
                   la anotacion
                -  points: la lista con las tuplas de puntos para la anotacion
        '''
        
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
                point = Point(point_x,point_y)

                points.append(point_from_physical_to_pixels(point_obj=point, \
                    image_prop_obj=self.associated_image))

            annotation = Annotation(annotation_id=annotation_id, \
                                    annotation_title=annotation_title, \
                                    annotation_path=self.annotation_path, \
                                    associated_image=self.associated_image, \
                                    points=points)

            annotations.append(annotation)

        return annotations


    def merge_annotations(self):
        '''
            funcion para mergear las anotaciones por ID de anotacion 
            (quizas tambien seria bueno hacerlo por "title")
        '''
        return



class Annotation:
    def __init__(self,annotation_id=None, annotation_title=None,annotation_path=None, associated_image=None, points=[]):
        # cuando termine de armar esto, deberia dejar como obligatorios 
        # todos los campos
        self.annotation_id = annotation_id
        self.annotation_title = annotation_title
        self.annotation_path = annotation_path
        self.associated_image = associated_image
        self.points = points
        
    def count_points(self):
        return len(self.points)

    def get_mask(self):
        '''
            Retorna una mascara de la imagen completa, con la region de la 
            anotacion marcada en 1s y el resto en 0s. 
            output:
                - np.array con mascara
        '''

        polygon = self.points
        width = self.associated_image.width_lvl_0
        height = self.associated_image.height_lvl_0

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=0, fill=1)
        mask = np.array(img)

        return mask

    def get_mask_area(self):
        # TODO
        # Te retorna el numero de pixeles contenidos en el area. 
        # Por ahora incluye el borde, aunque quizas no deberia hacerlo.
        # no se para que me puede servir, pero igual

        return

    def is_inside_region(self,tile):
        # TODO
        # Esta funcion dice si el tile esta o no adentro de la region, 
        # a partir de las coordenadas de cada una
        return



class Point:
    def __init__(self,x, y):
        self.coord = (x,y)

    def __str__(self):
        return 'x: ' + str(self.coord[0]) + ', y: ' + str(self.coord[1])


# TODO: Crear una nueva clase "AnnotationPoint" que herede la clase "Point"
# y agregue dos cosas:
#       -  funcion "point_from_physical_to_pixels", que no tiene que estar 
#          afuera ya que solo la ocupa esta clase
#       -  numero de punto: esto es importante ya que los puntos tienen que 
#          seguir un orden, igual que los juegos de connect the dots.
# 
# Pendiente arreglar el resto del codigo acorde a este cambio.
# PRIORITY: Low.


def point_from_physical_to_pixels(point_obj, image_prop_obj=None):
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

        # coordenada del punto (0,0) de la imagen de interes, tomando como eje de referencia 
        # el centro de la slide completa. se pasa la coordenada de nm a pixel
        slide_center_x = image_prop_obj.offset_x / (image_prop_obj.mpp_x * 1000)
        slide_center_y = image_prop_obj.offset_y / (image_prop_obj.mpp_y * 1000)
        
        x_0 = slide_center_x - image_prop_obj.width_lvl_0 / 2
        y_0 = slide_center_y  - image_prop_obj.height_lvl_0 / 2

        # al restar las coordenadas del punto de interes con las del punto (0,0) de la imagen, 
        # se obtiene la coordenada del punto de interes respecto al eje (0,0)
        x = point_obj.coord[0] / (image_prop_obj.mpp_x * 1000) - x_0
        y = point_obj.coord[1] / (image_prop_obj.mpp_y * 1000) - y_0

        return (round(x),round(y))



def save_mask_as_img(numpy_array, filename):
    im = Image.fromarray(np.uint8(numpy_array*255))
    im.save(filename)
    return






# annotation_path = "D:/felipe/ndpi/prueba1.ndpi.ndpa"
# image_path = 'D:/felipe/ndpi/prueba1.ndpi' 
# imagen = ImageProperties(image_path)
# print('offset_x, offset_y, mpp_x, mpp_y, width_lvl_0, height_lvl_0')
# print(imagen._get_image_parameters())
# print('--------------------------------------------------')


# annotations = create_annotations(image_prop_obj=imagen, annotation_path=annotation_path)

# i = 2
# print(annotations[i].points)
# print(annotations[i].annotation_title)
# print(annotations[i].annotation_path)
# print(annotations[i].image_path)
# print(annotations[i].count_points())
# print('--------------------------------------------------')


# max_x = 0
# min_x = 10000000000000

# max_y = 0
# min_y = 10000000000000
# for i in annotations[i].points:
#     max_x = max(max_x,i[0])
#     min_x = min(min_x,i[0])

#     max_y = max(max_y,i[1])
#     min_y = min(min_y,i[1])

# print('max x: {}, min x: {}, max y: {}, min y: {}'.format(max_x, min_x, max_y, min_y))


# ////////////////////////////////////////////////////////////////////////////////////////////////////////


# annotation_id = None
# annotation_path = None

# if (annotation_id != None and annotation_path != None):
#     print("OK")

# puntos = []

# p1 = Point(1,2)
# p2 = Point(3,4)
# p3 = Point(5,6)

# puntos.append(p1.coord)
# puntos.append(p2.coord)
# puntos.append(p3.coord)

# print(puntos)
# print(len(puntos))





