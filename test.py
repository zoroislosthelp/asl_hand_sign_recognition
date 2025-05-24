# for model1.h5

import cv2
from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from tensorflow.keras.models import load_model # type: ignore

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
# classifier = Classifier("Model/model1.h5", "Model/label.txt")

model = load_model("Model/model1.h5")
with open("Model/label.txt", "r") as f:
    labels = f.read().splitlines()
    
offset = 20
imgSize = 300
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize
        except:
            continue

        imgInput = imgWhite / 255.0
        imgInput = imgInput.reshape(1, 300, 300, 3)
        prediction = model.predict(imgInput)
        index = np.argmax(prediction)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break


# for model2.h5
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import math
# from tensorflow.keras.models import load_model

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=2)

# model = load_model("Model/model2.h5")
# with open("Model/label.txt", "r") as f:
#     labels = f.read().splitlines()

# offset = 20
# target_height = 224
# target_width = 244

# while True:
#     success, img = cap.read()
#     if not success:
#         print("Failed to capture image")
#         continue

#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)

#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
#         x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
#         imgCrop = img[y1:y2, x1:x2]

#         if imgCrop.size == 0:
#             continue

#         try:
#             imgResize = cv2.resize(imgCrop, (target_width, target_height))
#         except:
#             continue

#         imgInput = imgResize / 255.0
#         imgInput = imgInput.reshape(1, target_height, target_width, 3)

#         prediction = model.predict(imgInput)
#         index = np.argmax(prediction)

#         cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
#                       (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x, y - 26),
#                     cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#         cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ResizedInput", imgResize)

#     cv2.imshow("Image", imgOutput)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
