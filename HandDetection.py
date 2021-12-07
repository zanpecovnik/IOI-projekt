import cv2
import os
import HandDetectionModule as hdm

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

def getQuoteForImage(position, overlayImagePositions, emotion="Vesel"):
    idx = overlayImagePositions.index(position)
    return quotes[emotion][idx], idx

def do_work():
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
        ],
        "Prestrasen": [
            "Prvi prestrasen", "drugi prestrasen", "tretji prestrasen", "cetrti prestrasen", "peti prestrasen"
        ],
        "Ogaben": [
            "Prvi ogaben", "drugi ogaben", "tretji ogaben", "cetrti ogaben", "peti ogaben"
        ]
    }

    camWidth, camHeight = 1280, 720
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

    cap = cv2.VideoCapture(0)
    cap.set(3, camWidth)
    cap.set(4, camHeight)

    detector = hdm.handDetector(detectionCon=0.75)
    fingerTipIds = [4, 8, 12, 16, 20]
    isMovingImage = False
    prevTotalFingers = -1

    while True:
        success, img = cap.read()
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
                if newX >= camWidth - overlayImageSize:
                    newX = camWidth - overlayImageSize

                newY = int(averagePosition['y'] - overlayImageSize / 2)
                if newY < 0:
                    newY = 0
                if newY >= camHeight - overlayImageSize:
                    newY = camHeight - overlayImageSize

                overlayImagePositions.insert(0, {'x': newX, 'y': newY})

            if isMovingImage == False and totalFingers > 0:
                quote, idx = getQuoteForImage(imagePosition, overlayImagePositions)

        for i, position in enumerate(overlayImagePositions):
            if idx > -1 and i == idx:
                cv2.putText(img, quote, (position['x'], int(position['y'] + overlayImageSize / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            else:
                img[position['y'] : position['y'] + overlayImageSize, position['x'] : position['x'] + overlayImageSize] = overlayImage

        prevTotalFingers = totalFingers
        cv2.imshow("HandDetection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    do_work()