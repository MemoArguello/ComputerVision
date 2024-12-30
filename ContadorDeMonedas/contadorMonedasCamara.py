import cv2 #Biblioteca OpenCV para procesamiento de imágenes y visión por computadora.
import numpy as np #Para operaciones matemáticas, como transformar y ordenar puntos.

def ordenarpuntos(puntos):
    # Asegurar que los puntos sean un array de NumPy
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    # Ordenar por coordenada y
    y_order = sorted(n_puntos, key=lambda punto: punto[1])
    # Dividir en grupos: superiores e inferiores
    x1_order = sorted(y_order[:2], key=lambda punto: punto[0])  # Puntos superiores (x)
    x2_order = sorted(y_order[2:], key=lambda punto: punto[0])  # Puntos inferiores (x)
    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def alineamiento(imagen,ancho,alto):
    imagen_alineada=None
    grises=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    _,umbral=cv2.threshold(grises,150,255,cv2.THRESH_BINARY)
    #cv2.imshow("Umbral", umbral)
    contorno=cv2.findContours(umbral,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contorno = contorno[0] if len(contorno) == 2 else contorno[1]
    contorno=sorted(contorno,key=cv2.contourArea,reverse=True)[:1]
    for c in contorno:
        epsilon=0.01*cv2.arcLength(c, True)
        aproximacion=cv2.approxPolyDP(c,epsilon, True)
        if len(aproximacion)==4:
            puntos=ordenarpuntos(aproximacion)
            puntos1=np.float32(puntos)
            puntos2=np.float32([[0,0],[ancho,0],[0,alto],[ancho,alto]])
            M = cv2.getPerspectiveTransform(puntos1, puntos2)
            imagen_alineada=cv2.warpPerspective(imagen, M,(ancho,alto))
    return imagen_alineada
capturavideo=cv2.VideoCapture(1)

while True:
    tipocamara,camara=capturavideo.read()
    if tipocamara==False:
        break
    imagen_A6 = alineamiento(camara,ancho=480,alto=677) 
    if imagen_A6 is not None:
        puntos=[]
        imagen_gris=cv2.cvtColor(imagen_A6,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(imagen_gris,(5,5),1)
        _,umbral2=cv2.threshold(blur,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        #cv2.imshow("umbral",umbral2)
        contorno2=cv2.findContours(umbral2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(imagen_A6,contorno2,-1,(255,0,0),2)
        suma1=0.0
        suma2=0.0
        for c2 in contorno2:
            area=cv2.contourArea(c2)
            momentos=cv2.moments(c2)
            if (momentos["m00"]==0):
                momentos["m00"]=1.0
            x=int(momentos["m10"]/momentos["m00"])
            y=int(momentos["m01"]/momentos["m00"])

            if area<18377 and area>17000:
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6,"500Gs",(x,y),font, 0.75,(0,255,0),2)
                suma1+=500
            if area<21700 and area>20700:
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6,"1000",(x,y),font, 0.75,(0,255,0),2)
                suma2+=1000
        total=suma1+suma2
        print("Monedas de 500Gs: "+str(suma1/500))
        print("Monedas de 1000: "+str(suma2/1000))
        cv2.imshow("Imagen A6",imagen_A6)
        cv2.imshow("Camara",camara)
    if cv2.waitKey(1) == ord("q"):
        break
capturavideo.release()
cv2.destroyAllWindows()
print(type(contorno2), len(contorno2))