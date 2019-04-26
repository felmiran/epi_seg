import xml.etree.ElementTree as ET
import os
import io

from src.classes.classes import ImageProperties, Point, Annotation


os.chdir("D:/felipe/ndpi")

# path = "D:/felipe/ndpi/prueba1_1.xml"
# print(os.path.split(path)) <-----------------esto sirve para separar el nombre del archivo del resto del path

ndpa = open("prueba1.ndpi.ndpa")
tree = ET.parse(ndpa)

# root = tree.getroot()

# print("-----------------------------")
# print(root)
# print("-----------------------------")
# print(str(root.tag) + " - " + str(root.attrib))
# print("-----------------------------")
# for child in root:
#     if child.tag == "ndpviewstate" and child.attrib['id'] == 1:
#         print(str(child.tag) + " - " + str(child.attrib))
#         print("-----------------------------")



for ndpviewstate in tree.findall('ndpviewstate'):
    print('***********************************')
    print(ndpviewstate.find('title').text)
    print(ndpviewstate.get('id'))

    for point in ndpviewstate.find('annotation').find('pointlist').findall('point'):
        point_x = float(point.find('x').text)
        point_y = float(point.find('y').text)

        print(str(point_x) + ', ' + str(point_y))
    print('***********************************')




