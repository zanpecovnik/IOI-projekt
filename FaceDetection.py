from __future__ import division

import numpy as np
import cv2

from scipy.ndimage import zoom
from tensorflow.keras.models import load_model

import time
import keyboard

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def selectedEmotion(frame, emotion, size=50, alpha=0.3):
    image_dir = "images"

    happy_position = (10, 10)
    neutral_position = (size + 10, 10)
    surprise_position = (2*size + 10, 10)
    angry_position = (3*size + 10, 10)
    fear_position = (4*size + 10, 10)
    sad_position = (5*size + 10, 10)
    disgust_position = (6*size + 10, 10)

    # Happy emotion
    happy = cv2.resize(cv2.imread(f'{image_dir}/happy.png'), (size, size))
    ret_happy, mask_happy = cv2.threshold(cv2.cvtColor(happy, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    frame_happy = frame[happy_position[0] + 10:happy_position[0] + size + 10,
                  happy_position[1] + 10:happy_position[1] + size + 10]
    frame_happy[np.where(mask_happy)] = 0
    if emotion != "Happy":
        cv2.addWeighted(happy, 0.3, frame_happy, 0, 0, happy)
    frame_happy += happy

    # Neutral emotion
    netrual = cv2.resize(cv2.imread(f'{image_dir}/neutral.png'), (size, size))
    _, mask_neutral = cv2.threshold(cv2.cvtColor(netrual, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    frame_netrual = frame[neutral_position[0] + 10:neutral_position[0] + size + 10,
                  neutral_position[1] + 10:neutral_position[1] + size + 10]
    frame_netrual[np.where(mask_neutral)] = 0
    if emotion != "Neutral":
        cv2.addWeighted(netrual, 0.3, frame_netrual, 0, 0, netrual)
    frame_netrual += netrual

    # Surprise emotion
    surprise = cv2.resize(cv2.imread(f'{image_dir}/surprise.png'), (size, size))
    _, mask_surprise = cv2.threshold(cv2.cvtColor(surprise, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    frame_surprise = frame[surprise_position[0] + 10:surprise_position[0] + size + 10,
                    surprise_position[1] + 10:surprise_position[1] + size + 10]
    frame_surprise[np.where(mask_surprise)] = 0
    if emotion != "Surprise":
        cv2.addWeighted(surprise, 0.3, frame_surprise, 0, 0, surprise)
    frame_surprise += surprise

    # Angry emotion
    angry = cv2.resize(cv2.imread(f'{image_dir}/angry.png'), (size, size))
    _, mask_angry = cv2.threshold(cv2.cvtColor(angry, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    frame_angry = frame[angry_position[0] + 10:angry_position[0] + size + 10,
                     angry_position[1] + 10:angry_position[1] + size + 10]
    frame_angry[np.where(mask_angry)] = 0
    if emotion != "Angry":
        cv2.addWeighted(angry, 0.3, frame_angry, 0, 0, angry)
    frame_angry += angry

    # Fear emotion
    fear = cv2.resize(cv2.imread(f'{image_dir}/fear.png'), (size, size))
    _, mask_fear = cv2.threshold(cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    frame_fear = frame[fear_position[0] + 10:fear_position[0] + size + 10,
                  fear_position[1] + 10:fear_position[1] + size + 10]
    frame_fear[np.where(mask_fear)] = 0
    if emotion != "Fear":
        cv2.addWeighted(fear, 0.3, frame_fear, 0, 0, fear)
    frame_fear += fear

    # Sad emotion
    sad = cv2.resize(cv2.imread(f'{image_dir}/sad.png'), (size, size))
    _, mask_sad = cv2.threshold(cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    frame_sad = frame[sad_position[0] + 10:sad_position[0] + size + 10,
                 sad_position[1] + 10:sad_position[1] + size + 10]
    frame_sad[np.where(mask_sad)] = 0
    if emotion != "Sad":
        cv2.addWeighted(sad, 0.3, frame_sad, 0, 0, sad)
    frame_sad += sad

    # Disgust emotion
    disgust = cv2.resize(cv2.imread(f'{image_dir}/disgust.png'), (size, size))
    _, mask_disgust = cv2.threshold(cv2.cvtColor(disgust, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    frame_disgust = frame[disgust_position[0] + 10:disgust_position[0] + size + 10,
                disgust_position[1] + 10:disgust_position[1] + size + 10]
    frame_disgust[np.where(mask_disgust)] = 0
    if emotion != "Disgust":
        cv2.addWeighted(disgust, 0.3, frame_disgust, 0, 0, disgust)
    frame_disgust += disgust


def work():
    dataDir = "data"
    shape_x = 48
    shape_y = 48

    face_classifier = cv2.CascadeClassifier(f'{dataDir}/haarcascade_frontalface_default.xml')
    model = load_model(f'{dataDir}/video.h5', compile=False)

    cam_width, cam_height = 1280, 720
    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    prediction = ""
    max_seconds = 30
    start_time = time.time()
    read = False
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        # Read the emotion
        if time.time() - start_time < max_seconds and not read:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]

                # Zoom on face
                if face.shape[0] != 0 and face.shape[1] != 0:
                    face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
                    face = face.astype(np.float32)

                # Scale
                face /= float(face.max())
                face = np.reshape(face.flatten(), (1, 48, 48, 1))

                # Make Prediction
                model_prediction = model.predict(face)
                prediction_result = np.argmax(model_prediction)

                # Annotate main image with a label
                if prediction_result == 0:
                    prediction = "Angry"
                    selectedEmotion(frame, prediction)
                elif prediction_result == 1:
                    prediction = "Disgust"
                    selectedEmotion(frame, prediction)
                elif prediction_result == 2:
                    prediction = "Fear"
                    selectedEmotion(frame, prediction)
                elif prediction_result == 3:
                    prediction = "Happy"
                    selectedEmotion(frame, prediction)
                elif prediction_result == 4:
                    prediction = "Sad"
                    selectedEmotion(frame, prediction)
                elif prediction_result == 5:
                    prediction = "Surprise"
                    selectedEmotion(frame, prediction)
                else:
                    prediction = "Neutral"
                    selectedEmotion(frame, prediction)

                read = True

        # Waiting for reading has ended, read again
        elif time.time() - start_time >= max_seconds:
            selectedEmotion(frame, prediction)
            start_time = time.time()
            read = False

        # Just display emotion
        else:
            selectedEmotion(frame, prediction)

        cv2.imshow('Emotion Detector', frame)

        # Q = Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # R = repeat reading the emotion
        if keyboard.is_pressed('r'):
            start_time = time.time()
            read = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    work()