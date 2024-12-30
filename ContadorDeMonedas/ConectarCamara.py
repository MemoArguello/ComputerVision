import cv2 as cv
captureVideo=cv.VideoCapture(1)

if not captureVideo.isOpened():
    print("No se encontró una cámara")
    exit()
while True:
    tipocamara,camara = captureVideo.read()
    grises=cv.cvtColor(camara,cv.COLOR_BGR2GRAY)
    cv.imshow("En vivo",grises)
    if cv.waitKey(1)==ord('q'):
        break
captureVideo.release()
cv.destroyAllWindows()