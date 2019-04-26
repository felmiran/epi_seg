import classes
import numpy as np

image_path = 'D:/felipe/ndpi/prueba1.ndpi' 
imagen = classes.ImageProperties(image_path)

annotation_path = "D:/felipe/ndpi/prueba1.ndpi.ndpa"
annotations = classes.create_annotations(image_prop_obj=imagen, annotation_path=annotation_path)

annotation_prueba = annotations[0]
print(type(annotation_prueba.points))


from PIL import Image, ImageDraw

# polygon = annotation_prueba.points
# width = imagen.width_lvl_0
# height = imagen.height_lvl_0

# img = Image.new('L', (width, height), 0)
# ImageDraw.Draw(img).polygon(polygon, outline=0, fill=1)
# # img.save('mask.jpeg')

# mask = np.array(img)
# # print(mask)

mask = annotation_prueba.get_mask()

# im = Image.fromarray(np.uint8(mask*255))

# im.save('mask.jpeg')

classes.save_mask_as_img(mask,'mask.jpeg')

