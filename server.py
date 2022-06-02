from flask import Flask
from flask import render_template
from flask import Response
from detecto.core import Model

import cv2
app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def visualizar():
  score_filter=0.6
  model = Model.load('modelo.pth', ['Galleta', 'Galleta_rota'])

  while True:
    ret, frame = cap.read()
    if ret:
      cv2.putText(frame, 'Errores:{}'.format('0'), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
      labels, boxes, scores = model.predict(frame)
      timer = cv2.getTickCount()
      for i in range(boxes.shape[0]):
        if scores[i] < score_filter:
          continue

        box = boxes[i]
        tracker = cv2.legacy.TrackerMOSSE_create()

        ok = tracker.init(frame,(int(box[0]),int(box[1]),int(box[2]),int(box[3])))
        ok, bbox = tracker.update(frame)

        #cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)

        if ok:
          # Tracking success
          p1 = (int(bbox[0]), int(bbox[1]))
          p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
          cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
          # Tracking failure
          cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        if labels:
          fail_detected = False
          if 'Galleta_rota' in labels:
            fail_box_x_min = int(box[0])
            fail_box_x_max = int(box[2])
            fail_box_center_x = int((fail_box_x_min + fail_box_x_max) / 2)
            fail_box_y_min = int(box[1])
            fail_box_y_max = int(box[3])
            fail_box_center_y = int((fail_box_y_min+fail_box_y_max)/2)
            radius = 10
            cv2.circle(frame,(fail_box_center_x,fail_box_center_y),radius,(0,0,255),3)
            if fail_box_center_x < 325 and fail_box_center_x > 315 and not fail_detected:
              fail_detected = True
            else:
              fail_detected = False
          cv2.putText(frame, '{}: {}'.format(labels[i], round(scores[i].item(), 2)), (int(box[0]), int(box[1] - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
      (flag, encodedImage) = cv2.imencode(".jpg", frame)
      if not flag:
        continue
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
  errores=0
  return render_template("index.html")

@app.route("/video_feed")
def video_feed():
     return Response(visualizar(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
     app.run(debug=False)

cap.release()