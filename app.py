import numpy as np
from flask import Flask, render_template,request,Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from camera import Video
import cv2
import cvlib as cv
import numpy as np
from waitress import serve
app = Flask(__name__)
model = load_model('ouput/model_pretrain.h5')
label_map = ['1-4', '5-14', '15-21', '22-24', '25-30', '31-34']
@app.route('/')
def index():
    return render_template('base.html')

@app.route('/index')
def Webcam():
    return render_template('index.html')

@app.route('/after', methods = ['GET','POST'])
def after():
    img = request.files['file1']
    img.save('static/file.jpg')
    image = cv2.imread('static/file.jpg')
    image = cv2.resize(image, (200, 200))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    label_map = ['1-4','5-14','15-21','21-24','25-30','31-34','35-45']
    prediction = np.argmax(prediction)
    final_prediction = label_map[prediction]
    return render_template('after.html',data=final_prediction)


def gen():
    webcam = cv2.VideoCapture(0)
    while True:
        status, frame = webcam.read()

        face, confidence = cv.detect_face(frame)

        # loop through detected faces
        for idx, f in enumerate(face):

            # get corner points of face rectangle
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (200, 200))

            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

            # get label with max accuracy
            idx = np.argmax(conf)
            label = label_map[idx]

            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        cv2.imshow("ageetection", frame)
        ret,buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
        b'Content-Type:  image/jpeg\r\n\r\n' + frame +
        b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(gen(),
    mimetype ='multipart/x-mixed-replace; boundary=frame')


if __name__ ==  '__main__':
    app.run(debug=True)