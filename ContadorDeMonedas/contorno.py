import cv2
#lee la imagen
imagen = cv2.imread("contorno.jpg")
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

#aplica el umbral binario
#hay que ajustar el valor del umbral minimo para que se vea bien
_,umbral = cv2.threshold(grises,100,255, cv2.THRESH_BINARY)

#encuentra los contornos
#CHAIN_APPROX_SIMPLE no consume mucha memoria
contornos,jerarquia = cv2.findContours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#dibujar los contornos
cv2.drawContours(imagen, contornos, -1, (251, 60, 50),3)

#muestra la imagen
cv2.imshow("imagen original",imagen)
#muestra la imagen en grises
#cv2.imshow("imagen en gris",grises)
#muestra la imagen con el umbral binario
#cv2.imshow("imagen del umbral",umbral)

# Esperar a que se presione una tecla
cv2.waitKey(0)
# Cerrar todas las ventanas abiertas
cv2.destroyAllWindows()