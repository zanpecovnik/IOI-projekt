import cv2
import os
import HandDetectionModule as hdm
import ctypes

def getAveragePositionFromLandmarks(lmList):
    if len(lmList) != 0:
        x = int(sum(map(lambda subList: subList[1], lmList)) / len(lmList))
        y = int(sum(map(lambda subList: subList[2], lmList)) / len(lmList))
        return {'x': x, 'y': y}
    return {'x': -999, 'y': -999}

def findImageInFrontOfHand(avgPosition):
    for imagePosition in overlayImagePositions:
        if avgPosition['x'] >= imagePosition['x'] and avgPosition['x'] <= imagePosition['x'] + overlayImageSize and avgPosition['y'] >= imagePosition['y'] and avgPosition['y'] <= imagePosition['y'] + overlayImageSize:
            return imagePosition
    return None

def getQuoteForImage(position, emotion="Vesel"):
    idx = overlayImagePositions.index(position)
    return quotes[emotion][idx], idx

quotes = {
    "Vesel": [
        "Prvi vesel", "drugi vesel", "tretji vesel", "cetrti vesel", "peti vesel"
    ],
    "Jezen": [
        "Prvi jezen", "drugi jezen", "tretji jezen", "cetrti jezen", "peti jezen"
    ],
    "Zalosten": [
        "Prvi zalosten", "drugi zalosten", "tretji zalosten", "cetrti zalosten", "peti zalosten"
    ],
    "Nevtralen": [
        "Prvi nevtralen", "drugi nevtralen", "tretji nevtralen", "cetrti nevtralen", "peti nevtralen"
    ],
    "Presenecen": [
        "Prvi presenecen", "drugi presenecen", "tretji presenecen", "cetrti presenecen", "peti presenecen"
    ]
}

imageDir = "images"

overlayImageSize = 180
overlayImage = cv2.imread(f'{imageDir}/square.jpg')
overlayImagePositions = [
    {'x': 120, 'y': 60},
    {'x': 300, 'y': 440},
    {'x': 520, 'y': 270},
    {'x': 760, 'y': 150},
    {'x': 890, 'y': 540}
]

WINDOW_NAME = 'Full'

cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

detector = hdm.handDetector(detectionCon=0.75)
fingerTipIds = [4, 8, 12, 16, 20]
isMovingImage = False
prevTotalFingers = -1

while True:

    user32 = ctypes.windll.user32
    screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    success, img = cap.read()
    frame_height, frame_width, _ = img.shape

    scale_width = float(screen_width) / float(frame_width)
    scale_height = float(screen_height) / float(frame_height)

    if scale_height > scale_width:
        imgScale = scale_width

    else:
        imgScale = scale_height

    full_x, full_y = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(full_x), int(full_y)))

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    totalFingers = -1
    quote = None
    idx = -1

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
    imagePosition = findImageInFrontOfHand(averagePosition)

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
            quote, idx = getQuoteForImage(imagePosition)

    for i, position in enumerate(overlayImagePositions):
        if idx > -1 and i == idx:
            cv2.putText(img, quote, (position['x'], int(position['y'] + overlayImageSize / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        else:
            img[position['y'] : position['y'] + overlayImageSize, position['x'] : position['x'] + overlayImageSize] = overlayImage

    prevTotalFingers = totalFingers
    cv2.imshow(WINDOW_NAME, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()