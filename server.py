from flask import Flask
from flask import render_template
from flask import Response
from detecto.core import Model
from datetime import datetime
import os.path

import cv2
app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Función que se encarga de representar las cajas de detección en una imagen
#que se envía al servidor web para mostrarse en tiempo real
def visualizar():
  #Se fija un límite de puntuación para la detección
  score_filter=0.6
  #Se carga el modelo
  model = Model.load('modelo.pth', ['Galleta', 'Galleta_rota'])
  #Se carga el fichero de errores para poner el contador de errores
  if os.path.exists("errors.txt") == True:
    with open("errors.txt","r") as f:
      errores = len(f.readlines())
  else:
    errores=0

  while True:
    ret, frame = cap.read()
    if ret:
      #Se procesa la imagen para obtener las coordenadas de las cajas de detección de los objetos
      labels, boxes, scores = model.predict(frame)
      cv2.putText(frame, 'Errores:{}'.format(errores), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
      for i in range(boxes.shape[0]):
        if scores[i] < score_filter:
          continue
        box = boxes[i]
        #Se dibuja un rectángulo que contiene al objeto que se ha detectado
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
        label = labels[i]
        #Se comprueba si se ha detectado un objeto
        if label:
          fail_detected = False
          #Si el objeto detectado es una galleta rota se calcula su centro para seguirla
          if label == "Galleta_rota":
            fail_box_x_min = int(box[0])
            fail_box_x_max = int(box[2])
            fail_box_center_x = int((fail_box_x_min + fail_box_x_max) / 2)
            fail_box_y_min = int(box[1])
            fail_box_y_max = int(box[3])
            fail_box_center_y = int((fail_box_y_min + fail_box_y_max) / 2)
            radius = 10
            cv2.circle(frame, (fail_box_center_x, fail_box_center_y), radius, (0, 0, 255), 3)

            #Se coloca una línea imaginaria en el centro de la pantalla para comprobar que galletas la cruzan
            #y en el caso de que sea un agalleta con defectos, aumentar el contador de errores
            if fail_box_center_x < 325 and fail_box_center_x > 315 and not fail_detected:
              fail_detected = True

              #Se obtiene la fecha y hora actuales y se guardan en el fichero de errores
              now = datetime.now()
              error_text="Error detectado:{}\n".format(now)
              with open('errors.txt','a') as f:
                f.write(error_text)
              errores=errores+1
            else:
              fail_detected = False
          #Se coloca la etiqueta del objeto detectado y la precisión con la que se ha detectado
          cv2.putText(frame, '{}: {}'.format(label, round(scores[i].item(), 2)), (int(box[0]), int(box[1] - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
      (flag, encodedImage) = cv2.imencode(".jpg", frame)
      if not flag:
        continue
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +bytearray(encodedImage) + b'\r\n')

#Funcion que carga el index.html además de poblar el area de texto con los errores del fichero errors
@app.route("/")
def index():
  with open('errors.txt','r') as f:
    t = f.read()
  return render_template("index.html",t=t)

#Función que se encarga de mostrar el cuadro con las imágenes en la plantilla del index.html
@app.route("/video_feed")
def video_feed():
     return Response(visualizar(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
     app.run(debug=False)

cap.release()