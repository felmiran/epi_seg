from subprocess import call
import os

os.chdir("D:/felipe/ndpi")


# este es un ejemplo de script para cortar una imagen
# "ndpisplit -Ex40,z0,3600,13000,5000,5000,muestra_1 prueba1.ndpi"
# generalizando, es lo que escribio juanjo (que dios le pague):
# ndpisplit -E{magnification},{zlevel},{x},{y},{width},{height},{label} /
# {ndpi_path}
# ojo que el comando es sensible a espacios. Las comas tienen que ir sin
# espacios
# magnification: por ahora siempre "x40"
# zlevel: por ahora siempre "z0"
# x, y, width, height son en pixeles.
# 
#  
# 
#             (-)
#              |  (0,0) <--- esq. superior izq de la imagen completa
#              | /
# (-) ---------|----------------> x(+)
#              |               
#              |   (x,y) <--- esq. superior izq del pedazo de imagen que quiero
#              |  /___________
#              | |           | (x + width, y + height)
#              | |___________|/
#              v
#            y(+) 


# Este es para agregarle un label al final a la imagen:
# subprocess.call("ndpisplit -Ex40,z0,3600,13000,50000,5000,holi prueba1.ndpi",
# shell=True)

# ojo que el command es sensible a espacios!


def extract_region_from_ndpi(magnification, zlevel, x, y, width, height, label,
                             ndpi_path, has_parameters=True):

    parameters = "ndpisplit " +  \
            "-E{magnification},{zlevel},{x},{y},{width},{height},{label} " + \
            "{ndpi_path}"

    if has_parameters:
        command = parameters.format(magnification=magnification,
                                    zlevel=zlevel, x=x, y=y, width=width,
                                    height=height, label=label,
                                    ndpi_path=ndpi_path)

    else:
        command = ndpi_path

    call(command, shell=True)
    print("end")
    return


magnification = "x40"
zlevel = "z0"
x = 0
y = 0
width = 15000  # original is 28672
height = 17152
label = "full"
ndpi_path = "prueba2.ndpi"
has_parameters = True

extract_region_from_ndpi(magnification, zlevel, x, y, width, height, label,
                         ndpi_path, has_parameters=True)
