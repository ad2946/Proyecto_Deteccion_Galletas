from flask import Flask
from flask import render_template
from flask import Response
from detecto.core import Model
from datetime import datetime


import cv2
app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def visualizar():
  score_filter=0.6
  model = Model.load('modelo.pth', ['Galleta', 'Galleta_rota'])
  with open("errors.txt","r") as f:
    errores = len(f.readlines())
  while True:
    ret, frame = cap.read()
    if ret:
      labels, boxes, scores = model.predict(frame)
      cv2.putText(frame, 'Errores:{}'.format(errores), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
      for i in range(boxes.shape[0]):
        if scores[i] < score_filter:
          continue
        box = boxes[i]
        x_min=int(box[0])
        y_min=int(box[1])
        x_max=int(box[2])
        y_max=int(box[3])
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
        label = labels[i]
        if label:
          fail_detected = False
          if label == "Galleta_rota":
            fail_box_x_min = int(box[0])
            fail_box_x_max = int(box[2])
            fail_box_center_x = int((fail_box_x_min + fail_box_x_max) / 2)
            fail_box_y_min = int(box[1])
            fail_box_y_max = int(box[3])
            fail_box_center_y = int((fail_box_y_min + fail_box_y_max) / 2)
            radius = 10
            cv2.circle(frame, (fail_box_center_x, fail_box_center_y), radius, (0, 0, 255), 3)
            if fail_box_center_x < 325 and fail_box_center_x > 315 and not fail_detected:
              fail_detected = True
              now = datetime.now()
              error_text="Error detectado:{}\n".format(now)
              with open('errors.txt','a') as f:
                f.write(error_text)
              errores=errores+1
            else:
              fail_detected = False
          cv2.putText(frame, '{}: {}'.format(label, round(scores[i].item(), 2)), (int(box[0]), int(box[1] - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

      (flag, encodedImage) = cv2.imencode(".jpg", frame)
      if not flag:
        continue
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
  with open('errors.txt','r') as f:
    t = f.read()
  return render_template("index.html",t=t)

@app.route("/video_feed")
def video_feed():
     return Response(visualizar(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
     app.run(debug=False)

cap.release()