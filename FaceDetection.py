from __future__ import division

import numpy as np
import cv2

from scipy.ndimage import zoom
import dlib

from tensorflow.keras.models import load_model
from imutils import face_utils

import time
import keyboard

def work():
    shape_x = 48
    shape_y = 48

    model = load_model('data/video.h5')
    face_detect = dlib.get_frontal_face_detector()

    video_capture = cv2.VideoCapture(0)

    max_seconds = 10
    start_time = time.time()
    read = False

    prediction = ""
    position = 0
    while True:

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)

        # Read the emotion
        if time.time() - start_time < max_seconds and not read:
            for (i, rect) in enumerate(rects):

                # Face coordinates
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                face = gray[y:y + h, x:x + w]

                # Zoom on face
                face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
                face = face.astype(np.float32)

                # Scale
                face /= float(face.max())
                face = np.reshape(face.flatten(), (1, 48, 48, 1))

                # Make Prediction
                prediction = model.predict(face)
                prediction_result = np.argmax(prediction)

                # Annotate main image with a label
                if prediction_result == 0:
                    prediction = "Angry"
                elif prediction_result == 1:
                    prediction = "Disgust"
                elif prediction_result == 2:
                    prediction = "Fear"
                elif prediction_result == 3:
                    prediction = "Happy"
                elif prediction_result == 4:
                    prediction = "Sad"
                elif prediction_result == 5:
                    prediction = "Surprise"
                else:
                    prediction = "Neutral"

                position = 180*i
                cv2.putText(frame, prediction, (40, 140 + position),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)

                read = True

        # Waiting for reading has ended, read again
        elif time.time() - start_time >= max_seconds:
            cv2.putText(frame, prediction, (40, 140 + position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            start_time = time.time()
            read = False

        # Just display emotion
        else:
            cv2.putText(frame, prediction, (40, 140 + position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # R = repeat reading the emotion
        if keyboard.is_pressed('r'):
            start_time = time.time()
            read = False

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    work()