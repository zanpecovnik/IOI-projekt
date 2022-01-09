from __future__ import division


from scipy.ndimage import zoom
from tensorflow.keras.models import load_model

import os
import cv2
import math
import ctypes
import keyboard
import mouse
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

def getQuoteForImage(position, overlayImagePositions, emotion, bingo):
    res = "default"
    if bingo == "true":
        res = emotion + "_correct"
    elif bingo == "false":
        res = emotion + "_incorrect"

    idx = overlayImagePositions.index(position)
    return quotes[res][idx], idx


def pointerInside(x, y, r):

    happy = (80, 60)
    neutral = (226, 60)
    surprise = (372, 60)
    angry = (519, 60)
    fear = (666, 60)
    sad = (812, 60)
    disgust = (960, 60)

    result = ""
    sqrtHappy = math.sqrt(((x - happy[0]) ** 2) + ((y - happy[1]) ** 2))
    sqrtNeutral = math.sqrt(((x - neutral[0]) ** 2) + ((y - neutral[1]) ** 2))
    sqrtSurprise = math.sqrt(((x - surprise[0]) ** 2) + ((y - surprise[1]) ** 2))
    sqrtAngry = math.sqrt(((x - angry[0]) ** 2) + ((y - angry[1]) ** 2))
    sqrtFear = math.sqrt(((x - fear[0]) ** 2) + ((y - fear[1]) ** 2))
    sqrtSad = math.sqrt(((x - sad[0]) ** 2) + ((y - sad[1]) ** 2))
    sqrtDisgust = math.sqrt(((x - disgust[0]) ** 2) + ((y - disgust[1]) ** 2))

    if (sqrtHappy < r):
        result = "happy"
    if (sqrtNeutral < r):
        result = "neutral"
    if (sqrtSurprise < r):
        result = "surprise"
    if (sqrtAngry < r):
        result = "angry"
    if (sqrtFear < r):
        result = "fear"
    if (sqrtSad < r):
        result = "sad"
    if (sqrtDisgust < r):
        result = "disgust"

    return result


def topBarInfo(frame, position, correct, total, bingo, guessing, guessedEmotion, actualEmotion, size=100):
    image_dir = "images"

    happy_position = (0, 0)
    neutral_position = (0, size + 10)
    surprise_position = (0, 2*size + 20)
    angry_position = (0, 3*size + 30)
    fear_position = (0, 4*size + 40)
    sad_position = (0, 5*size + 50)
    disgust_position = (0, 6*size + 60)

    total_score_position = (7 * size + 100, 95)
    miss_score_position = (8 * size - 2, 50)
    actual_prediction_position = (12, 8 * size)

    cv2.putText(frame, "Total score: " + str(correct) + "/" + str(total), total_score_position, cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (0, 0, 0), 2)

    if guessing:
        x = position[0]
        y = position[1]
        inside = pointerInside(x, y, 60)

        # Happy emotion
        happy = cv2.resize(cv2.imread(f'{image_dir}/happy.png'), (size, size))
        ret_happy, mask_happy = cv2.threshold(cv2.cvtColor(happy, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_happy = frame[happy_position[0] + 10:happy_position[0] + size + 10,
                      happy_position[1] + 10:happy_position[1] + size + 10]
        if inside != "happy":
            cv2.addWeighted(happy, 0.3, frame_happy, 0, 0, happy)
        frame_happy[np.where(mask_happy)] = 0
        frame_happy += happy

        # Neutral emotion
        netrual = cv2.resize(cv2.imread(f'{image_dir}/neutral.png'), (size, size))
        _, mask_neutral = cv2.threshold(cv2.cvtColor(netrual, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_netrual = frame[neutral_position[0] + 10:neutral_position[0] + size + 10,
                        neutral_position[1] + 10:neutral_position[1] + size + 10]
        if inside != "neutral":
            cv2.addWeighted(netrual, 0.3, frame_netrual, 0, 0, netrual)
        frame_netrual[np.where(mask_neutral)] = 0
        frame_netrual += netrual

        # Surprise emotion
        surprise = cv2.resize(cv2.imread(f'{image_dir}/surprise.png'), (size, size))
        _, mask_surprise = cv2.threshold(cv2.cvtColor(surprise, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_surprise = frame[surprise_position[0] + 10:surprise_position[0] + size + 10,
                         surprise_position[1] + 10:surprise_position[1] + size + 10]
        if inside != "surprise":
            cv2.addWeighted(surprise, 0.3, frame_surprise, 0, 0, surprise)
        frame_surprise[np.where(mask_surprise)] = 0
        frame_surprise += surprise

        # Angry emotion
        angry = cv2.resize(cv2.imread(f'{image_dir}/angry.png'), (size, size))
        _, mask_angry = cv2.threshold(cv2.cvtColor(angry, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_angry = frame[angry_position[0] + 10:angry_position[0] + size + 10,
                      angry_position[1] + 10:angry_position[1] + size + 10]
        if inside != "angry":
            cv2.addWeighted(angry, 0.3, frame_angry, 0, 0, angry)
        frame_angry[np.where(mask_angry)] = 0
        frame_angry += angry

        # Fear emotion
        fear = cv2.resize(cv2.imread(f'{image_dir}/fear.png'), (size, size))
        _, mask_fear = cv2.threshold(cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_fear = frame[fear_position[0] + 10:fear_position[0] + size + 10,
                     fear_position[1] + 10:fear_position[1] + size + 10]
        if inside != "fear":
            cv2.addWeighted(fear, 0.3, frame_fear, 0, 0, fear)
        frame_fear[np.where(mask_fear)] = 0
        frame_fear += fear

        # Sad emotion
        sad = cv2.resize(cv2.imread(f'{image_dir}/sad.png'), (size, size))
        _, mask_sad = cv2.threshold(cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_sad = frame[sad_position[0] + 10:sad_position[0] + size + 10,
                    sad_position[1] + 10:sad_position[1] + size + 10]
        if inside != "sad":
            cv2.addWeighted(sad, 0.3, frame_sad, 0, 0, sad)
        frame_sad[np.where(mask_sad)] = 0
        frame_sad += sad

        # Disgust emotion
        disgust = cv2.resize(cv2.imread(f'{image_dir}/disgust.png'), (size, size))
        _, mask_disgust = cv2.threshold(cv2.cvtColor(disgust, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_disgust = frame[disgust_position[0] + 10:disgust_position[0] + size + 10,
                        disgust_position[1] + 10:disgust_position[1] + size + 10]
        if inside != "disgust":
            cv2.addWeighted(disgust, 0.3, frame_disgust, 0, 0, disgust)
        frame_disgust[np.where(mask_disgust)] = 0
        frame_disgust += disgust

    else:

        xPlus = 0
        if bingo == "true":
            cv2.putText(frame, "SCORE! Actual:", miss_score_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 0, 0), 2)
            xPlus = 280
        else:
            cv2.putText(frame, "MISS! Actual:", miss_score_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (0, 0, 0), 2)
            xPlus = 240

        if actualEmotion == "happy":
            happy_actual = cv2.resize(cv2.imread(f'{image_dir}/happy.png'), (30, 30))
            ret_happy_actual, mask_happy_actual = cv2.threshold(cv2.cvtColor(happy_actual, cv2.COLOR_BGR2GRAY), 1,
                                                                255, cv2.THRESH_BINARY)
            frame_happy_actual = frame[actual_prediction_position[0] + 10:actual_prediction_position[0] + 30 + 10,
                                 actual_prediction_position[1] + 10 + xPlus:actual_prediction_position[1] + 30 + 10 + xPlus]
            frame_happy_actual[np.where(mask_happy_actual)] = 0
            frame_happy_actual += happy_actual

        if actualEmotion == "neutral":
            netrual = cv2.resize(cv2.imread(f'{image_dir}/neutral.png'), (30, 30))
            _, mask_neutral = cv2.threshold(cv2.cvtColor(netrual, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
            frame_netrual = frame[actual_prediction_position[0] + 10:actual_prediction_position[0] + 30 + 10,
                            actual_prediction_position[1] + 10 + xPlus:actual_prediction_position[1] + 30 + 10 + xPlus]
            frame_netrual[np.where(mask_neutral)] = 0
            frame_netrual += netrual

        if actualEmotion == "surprise":
            surprise = cv2.resize(cv2.imread(f'{image_dir}/surprise.png'), (30, 30))
            _, mask_surprise = cv2.threshold(cv2.cvtColor(surprise, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
            frame_surprise = frame[actual_prediction_position[0] + 10:actual_prediction_position[0] + 30 + 10,
                             actual_prediction_position[1] + 10 + xPlus:actual_prediction_position[1] + 30 + 10 + xPlus]
            frame_surprise[np.where(mask_surprise)] = 0
            frame_surprise += surprise

        if actualEmotion == "angry":
            angry = cv2.resize(cv2.imread(f'{image_dir}/angry.png'), (30, 30))
            _, mask_angry = cv2.threshold(cv2.cvtColor(angry, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
            frame_angry = frame[actual_prediction_position[0] + 10:actual_prediction_position[0] + 30 + 10,
                          actual_prediction_position[1] + 10 + xPlus:actual_prediction_position[1] + 30 + 10 + xPlus]
            frame_angry[np.where(mask_angry)] = 0
            frame_angry += angry

        if actualEmotion == "fear":
            fear = cv2.resize(cv2.imread(f'{image_dir}/fear.png'), (30, 30))
            _, mask_fear = cv2.threshold(cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
            frame_fear = frame[actual_prediction_position[0] + 10:actual_prediction_position[0] + 30 + 10,
                         actual_prediction_position[1] + 10 + xPlus:actual_prediction_position[1] + 30 + 10 + xPlus]
            frame_fear[np.where(mask_fear)] = 0
            frame_fear += fear

        if actualEmotion == "sad":
            sad = cv2.resize(cv2.imread(f'{image_dir}/sad.png'), (30, 30))
            _, mask_sad = cv2.threshold(cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
            frame_sad = frame[actual_prediction_position[0] + 10:actual_prediction_position[0] + 30 + 10,
                        actual_prediction_position[1] + 10 + xPlus:actual_prediction_position[1] + 30 + 10 + xPlus]
            frame_sad[np.where(mask_sad)] = 0
            frame_sad += sad

        if actualEmotion == "disgust":
            disgust = cv2.resize(cv2.imread(f'{image_dir}/disgust.png'), (30, 30))
            _, mask_disgust = cv2.threshold(cv2.cvtColor(disgust, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
            frame_disgust = frame[actual_prediction_position[0] + 10:actual_prediction_position[0] + 30 + 10,
                            actual_prediction_position[1] + 10 + xPlus:actual_prediction_position[1] + 30 + 10 + xPlus]
            frame_disgust[np.where(mask_disgust)] = 0
            frame_disgust += disgust

        # Happy emotion
        happy = cv2.resize(cv2.imread(f'{image_dir}/happy.png'), (size, size))
        ret_happy, mask_happy = cv2.threshold(cv2.cvtColor(happy, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_happy = frame[happy_position[0] + 10:happy_position[0] + size + 10,
                      happy_position[1] + 10:happy_position[1] + size + 10]
        if guessedEmotion != "happy":
            cv2.addWeighted(happy, 0.3, frame_happy, 0, 0, happy)
        frame_happy[np.where(mask_happy)] = 0
        frame_happy += happy

        # Neutral emotion
        netrual = cv2.resize(cv2.imread(f'{image_dir}/neutral.png'), (size, size))
        _, mask_neutral = cv2.threshold(cv2.cvtColor(netrual, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_netrual = frame[neutral_position[0] + 10:neutral_position[0] + size + 10,
                        neutral_position[1] + 10:neutral_position[1] + size + 10]
        if guessedEmotion != "neutral":
            cv2.addWeighted(netrual, 0.3, frame_netrual, 0, 0, netrual)
        frame_netrual[np.where(mask_neutral)] = 0
        frame_netrual += netrual

        # Surprise emotion
        surprise = cv2.resize(cv2.imread(f'{image_dir}/surprise.png'), (size, size))
        _, mask_surprise = cv2.threshold(cv2.cvtColor(surprise, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_surprise = frame[surprise_position[0] + 10:surprise_position[0] + size + 10,
                         surprise_position[1] + 10:surprise_position[1] + size + 10]
        if guessedEmotion != "surprise":
            cv2.addWeighted(surprise, 0.3, frame_surprise, 0, 0, surprise)
        frame_surprise[np.where(mask_surprise)] = 0
        frame_surprise += surprise

        # Angry emotion
        angry = cv2.resize(cv2.imread(f'{image_dir}/angry.png'), (size, size))
        _, mask_angry = cv2.threshold(cv2.cvtColor(angry, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_angry = frame[angry_position[0] + 10:angry_position[0] + size + 10,
                      angry_position[1] + 10:angry_position[1] + size + 10]
        if guessedEmotion != "angry":
            cv2.addWeighted(angry, 0.3, frame_angry, 0, 0, angry)
        frame_angry[np.where(mask_angry)] = 0
        frame_angry += angry

        # Fear emotion
        fear = cv2.resize(cv2.imread(f'{image_dir}/fear.png'), (size, size))
        _, mask_fear = cv2.threshold(cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_fear = frame[fear_position[0] + 10:fear_position[0] + size + 10,
                     fear_position[1] + 10:fear_position[1] + size + 10]
        if guessedEmotion != "fear":
            cv2.addWeighted(fear, 0.3, frame_fear, 0, 0, fear)
        frame_fear[np.where(mask_fear)] = 0
        frame_fear += fear

        # Sad emotion
        sad = cv2.resize(cv2.imread(f'{image_dir}/sad.png'), (size, size))
        _, mask_sad = cv2.threshold(cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_sad = frame[sad_position[0] + 10:sad_position[0] + size + 10,
                    sad_position[1] + 10:sad_position[1] + size + 10]
        if guessedEmotion != "sad":
            cv2.addWeighted(sad, 0.3, frame_sad, 0, 0, sad)
        frame_sad[np.where(mask_sad)] = 0
        frame_sad += sad

        # Disgust emotion
        disgust = cv2.resize(cv2.imread(f'{image_dir}/disgust.png'), (size, size))
        _, mask_disgust = cv2.threshold(cv2.cvtColor(disgust, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        frame_disgust = frame[disgust_position[0] + 10:disgust_position[0] + size + 10,
                        disgust_position[1] + 10:disgust_position[1] + size + 10]
        if guessedEmotion != "disgust":
            cv2.addWeighted(disgust, 0.3, frame_disgust, 0, 0, disgust)
        frame_disgust[np.where(mask_disgust)] = 0
        frame_disgust += disgust


quotes = {
    "default": [
        "Preizkusite", "Preizkusite", "Preizkusite", "Preizkusite", "Preizkusite"
    ],
    "happy_correct": [
        "Prvi vesel", "Drugi vesel", "Tretji vesel", "Cetrti vesel", "Peti vesel"
    ],
    "happy_incorrect": [
        "Prvi vesel", "Drugi vesel", "Tretji vesel", "Cetrti vesel", "Peti vesel"
    ],
    "angry_correct": [
        "Prvi jezen", "Drugi jezen", "Tretji jezen", "Cetrti jezen", "Peti jezen"
    ],
    "angry_incorrect": [
        "Prvi jezen", "Drugi jezen", "Tretji jezen", "Cetrti jezen", "Peti jezen"
    ],
    "sad_correct": [
        "Prvi zalosten", "Drugi zalosten", "Tretji zalosten", "Cetrti zalosten", "Peti zalosten"
    ],
    "sad_incorrect": [
        "Prvi zalosten", "Drugi zalosten", "Tretji zalosten", "Cetrti zalosten", "Peti zalosten"
    ],
    "neutral_correct": [
        "Prvi nevtralen", "Drugi nevtralen", "Tretji nevtralen", "Cetrti nevtralen", "Peti nevtralen"
    ],
    "neutral_incorrect": [
        "Prvi nevtralen", "Drugi nevtralen", "Tretji nevtralen", "Cetrti nevtralen", "Peti nevtralen"
    ],
    "disgust_correct": [
        "Prvi gnus", "Drugi gnus", "Tretji gnus", "Cetrti gnus", "Peti gnus"
    ],
    "disgust_incorrect": [
        "Prvi gnus", "Drugi gnus", "Tretji gnus", "Cetrti gnus", "Peti gnus"
    ],
    "fear_correct": [
        "Prvi strah", "Drugi strah", "Tretji strah", "Cetrti strah", "Peti strah"
    ],
    "fear_incorrect": [
        "Prvi strah", "Drugi strah", "Tretji strah", "Cetrti strah", "Peti strah"
    ],
    "surprise_correct": [
        "Prvi presenecen", "Drugi presenecen", "Tretji presenecen", "Cetrti presenecen", "Peti presenecen"
    ],
    "surprise_incorrect": [
        "Prvi presenecen", "Drugi presenecen", "Tretji presenecen", "Cetrti presenecen", "Peti presenecen"
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
        {'x': 120, 'y': 160},
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
    read = False
    bingo = ""
    predictEmotion = False
    guessing = True
    predictedEmotion = ""
    predictedEmotionCorrect = 0
    predictedEmotionTotal = 0

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
        if not read:
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
                    prediction = "angry"
                elif prediction_result == 1:
                    prediction = "disgust"
                elif prediction_result == 2:
                    prediction = "fear"
                elif prediction_result == 3:
                    prediction = "happy"
                elif prediction_result == 4:
                    prediction = "sad"
                elif prediction_result == 5:
                    prediction = "surprise"
                else:
                    prediction = "neutral"

                read = True

        # Predict self emotion
        if predictEmotion and guessing:
            if predictedEmotion != "":
                if prediction == predictedEmotion:
                    predictedEmotionCorrect += 1
                    bingo = "true"
                else:
                    bingo = "false"
                predictedEmotionTotal += 1
                guessing = False
            else:
                print("Select valid emotion!")
            predictEmotion = False

        # Fill top bar with emojis and info about scores
        topBarInfo(frame, mouse.get_position(), predictedEmotionCorrect, predictedEmotionTotal, bingo, guessing, predictedEmotion, prediction)

        # Using Hand to open or move the boxes
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
                if newY < 130:
                    newY = 130
                if newY >= full_y - overlayImageSize:
                    newY = int(full_y - overlayImageSize)

                overlayImagePositions.insert(0, {'x': newX, 'y': newY})

            if isMovingImage == False and totalFingers > 0:
                quote, idx = getQuoteForImage(imagePosition, overlayImagePositions, prediction, bingo)

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

        if mouse.is_pressed() and guessing:
            read = False
            predictEmotion = True

            mousePosition = mouse.get_position()
            inside = pointerInside(mousePosition[0], mousePosition[1], 60)
            predictedEmotion = inside

        if keyboard.is_pressed('r'):
            bingo = ""
            guessing = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    work()
