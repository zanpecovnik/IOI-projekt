from __future__ import division

import numpy as np
import cv2

from scipy.ndimage import zoom
import dlib

from tensorflow.keras.models import load_model
from imutils import face_utils


def work():
    shape_x = 48
    shape_y = 48

    model = load_model('data/video.h5')
    face_detect = dlib.get_frontal_face_detector()

    video_capture = cv2.VideoCapture(0)

    while True:

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)

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

            # Rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

            # Annotate main image with a label
            if prediction_result == 0:
                cv2.putText(frame, "Angry", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 1:
                cv2.putText(frame, "Disgust", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 2:
                cv2.putText(frame, "Fear", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 3:
                cv2.putText(frame, "Happy", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 4:
                cv2.putText(frame, "Sad", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif prediction_result == 5:
                cv2.putText(frame, "Surprise", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Neutral", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    work()