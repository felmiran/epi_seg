import openslide
import os


# para que funcione este script, el path al bin de este archivo debe estar dentro de los paths en las variables de entorno

slide_dir_path = 'D:/felipe/ndpi' 
ndpi_filename = "prueba1.ndpi"

os.chdir(slide_dir_path)

slide = openslide.OpenSlide(ndpi_filename)

# print("XOffsetFromSlideCentre (distancia en nm al centro ): " + slide.properties['hamamatsu.XOffsetFromSlideCentre'])
# print("YOffsetFromSlideCentre: " + slide.properties['hamamatsu.YOffsetFromSlideCentre'])

# print("mpp-x (Pixel Width in micrometers): " + slide.properties['openslide.mpp-x'])
# print("mpp-y (Pixel Height in micrometers): " + slide.properties['openslide.mpp-y'])

# for i in range(0,9):
    

#     print("width [{}]: ".format(i) + slide.properties['openslide.level[{}].width'.format(i)])
#     print("height [{}]: ".format(i) + slide.properties['openslide.level[{}].height'.format(i)])
#     print("downsample [{}]: ".format(i) + slide.properties['openslide.level[{}].downsample'.format(i)])
#     print("tile-height [{}]: ".format(i) + slide.properties['openslide.level[{}].tile-height'.format(i)])
#     print("tile-width [{}]: ".format(i) + slide.properties['openslide.level[{}].tile-width'.format(i)])



for i in slide.properties:
     print(i + ": " + slide.properties[i])
     print("-------------------------")





