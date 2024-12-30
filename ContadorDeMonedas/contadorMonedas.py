#importamos las librerias
import cv2
import numpy as np

#variables
valorGauss = 3
ValorKernel = 3

#lee la imagen
original = cv2.imread("monedas.jpg")

#convertir la imagen a escala de grises
grises = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

#aplicar el filtro gaussiano, desenfoque
gauss = cv2.GaussianBlur(grises, (valorGauss, valorGauss),0)

#canny para reducir el ruido, bordes
canny = cv2.Canny(gauss, 60,100)

#eliminar el ruido interno
kernel = np.ones((ValorKernel, ValorKernel),np.uint8)
cierre=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel)

contornos, jerarqias=cv2.findContours(cierre.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Monedas encontradas: {}".format(len(contornos)))

cv2.drawContours(original, contornos, -1, (251,60,50),3)
#Mostrar Resultados
#cv2.imshow("Imagen gauss", gauss)
#cv2.imshow("Imagen grises", grises)
#cv2.imshow("Canny", canny)

cv2.imshow("Imagen original", original)
cv2.waitKey(0)