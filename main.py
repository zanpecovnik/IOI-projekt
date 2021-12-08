from __future__ import division
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model

import os
import cv2
import time
import ctypes
import keyboard
import numpy as np
import HandDetectionModule as hdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def getAveragePositionFromLandmarks(lmList):
    if len(lmList) != 0:
        x = int(sum(map(lambda subList: subList[1], lmList)) / len(lmList))
        y = int(sum(map(lambda subList: subList[2], lmList)) / len(lmList))
        return {'x': x, 'y': y}
    return {'x': -999, 'y': -999}

def findImageInFrontOfHand(avgPosition, overlayImagePositions, overlayImageSize):
    for imagePosition in overlayImagePositions:
        if avgPosition['x'] >= imagePosition['x'] and avgPosition['x'] <= imagePosition['x'] + overlayImageSize and avgPosition['y'] >= imagePosition['y'] and avgPosition['y'] <= imagePosition['y'] + overlayImageSize:
            return imagePosition
    return None

def getQuoteForImage(position, overlayImagePositions, emotion="Happy"):
    idx = overlayImagePositions.index(position)
    return quotes[emotion][idx], idx


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

quotes = {
    "Happy": [
        "Prvi vesel", "drugi vesel", "tretji vesel", "cetrti vesel", "peti vesel"
    ],
    "Angry": [
        "Prvi jezen", "drugi jezen", "tretji jezen", "cetrti jezen", "peti jezen"
    ],
    "Sad": [
        "Prvi zalosten", "drugi zalosten", "tretji zalosten", "cetrti zalosten", "peti zalosten"
    ],
    "Neutral": [
        "Prvi nevtralen", "drugi nevtralen", "tretji nevtralen", "cetrti nevtralen", "peti nevtralen"
    ],
    "Disgust": [
        "Prvi disgust", "drugi disgust", "tretji disgust", "cetrti disgust", "peti disgust"
    ],
    "Fear": [
        "Prvi strah", "drugi strah", "tretji strah", "cetrti strah", "peti strah"
    ],
    "Surprise": [
        "Prvi presenecen", "drugi presenecen", "tretji presenecen", "cetrti presenecen", "peti presenecen"
    ]
}

def work():
    imageDir = "images"
    WINDOW_NAME = 'Full'
    dataDir = "data"
    shape_x = 48
    shape_y = 48

    overlayImageSize = 180
    overlayImage = cv2.imread(f'{imageDir}/square.jpg')
    overlayImagePositions = [
        {'x': 120, 'y': 60},
        {'x': 300, 'y': 440},
        {'x': 520, 'y': 270},
        {'x': 760, 'y': 150},
        {'x': 890, 'y': 540}
    ]

    face_classifier = cv2.CascadeClassifier(f'{dataDir}/haarcascade_frontalface_default.xml')
    model = load_model(f'{dataDir}/video.h5', compile=False)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    detector = hdm.handDetector(detectionCon=0.75)
    fingerTipIds = [4, 8, 12, 16, 20]
    isMovingImage = False
    prevTotalFingers = -1
    
    prediction = ""
    max_seconds = 30
    start_time = time.time()
    read = False

    while True:
        user32 = ctypes.windll.user32
        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        ret, frame = cap.read()
        frame_height, frame_width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        scale_width = float(screen_width) / float(frame_width)
        scale_height = float(screen_height) / float(frame_height)

        if scale_height > scale_width:
            imgScale = scale_width

        else:
            imgScale = scale_height

        full_x, full_y = frame.shape[1] * imgScale, frame.shape[0] * imgScale
        frame = cv2.resize(frame, (int(full_x), int(full_y)))

        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)
        totalFingers = -1
        quote = None
        idx = -1

        
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

        if len(lmList) != 0:
            fingers = []

            # Thumb
            if lmList[fingerTipIds[0]][1] > lmList[fingerTipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lmList[fingerTipIds[id]][2] < lmList[fingerTipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)

        averagePosition = getAveragePositionFromLandmarks(lmList)
        imagePosition = findImageInFrontOfHand(averagePosition, overlayImagePositions, overlayImageSize)

        if imagePosition != None:
            if isMovingImage == False and totalFingers == 0 and prevTotalFingers > 0:
                isMovingImage = True

            if isMovingImage and totalFingers == 5:
                isMovingImage = False

            if isMovingImage:
                overlayImagePositions = list(filter(lambda imPos: imPos != imagePosition, overlayImagePositions))

                newX = int(averagePosition['x'] - overlayImageSize / 2)
                if newX < 0:
                    newX = 0
                if newX >= full_x - overlayImageSize:
                    newX = int(full_x - overlayImageSize)

                newY = int(averagePosition['y'] - overlayImageSize / 2)
                if newY < 0:
                    newY = 0
                if newY >= full_y - overlayImageSize:
                    newY = int(full_y - overlayImageSize)

                overlayImagePositions.insert(0, {'x': newX, 'y': newY})

            if isMovingImage == False and totalFingers > 0:
                quote, idx = getQuoteForImage(imagePosition, overlayImagePositions, prediction)

        for i, position in enumerate(overlayImagePositions):
            if idx > -1 and i == idx:
                cv2.putText(frame, quote, (position['x'], int(position['y'] + overlayImageSize / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            else:
                frame[position['y'] : position['y'] + overlayImageSize, position['x'] : position['x'] + overlayImageSize] = overlayImage

        prevTotalFingers = totalFingers
        
        cv2.imshow(WINDOW_NAME, frame)
        
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