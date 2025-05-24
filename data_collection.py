import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import string

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
maxImagesPerFolder = 100
alphabet = list(string.ascii_uppercase)

# Function to find the next available folder
def get_next_folder(base_path="hand_sign_detection/Data"):
    os.makedirs(base_path, exist_ok=True)
    for letter in alphabet:
        folder_path = os.path.join(base_path, letter)
        if not os.path.exists(folder_path) or len(os.listdir(folder_path)) < maxImagesPerFolder:
            os.makedirs(folder_path, exist_ok=True)
            return folder_path
    raise Exception("All folders from A-Z are full!")

folder = get_next_folder()
counter = len(os.listdir(folder))

while True:
    success, img = cap.read()
    if not success:
        print("Camera not detected or failed to read frame")
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        hImg, wImg, _ = img.shape

        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(wImg, x + w + offset), min(hImg, y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Empty crop, skipping frame")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        if counter >= maxImagesPerFolder:
            folder = get_next_folder()
            counter = 0
            print(f"Switched to new folder: {folder}")
            time.sleep(2)
        filename = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, imgWhite)
        counter += 1
        print(f'Saved {filename}')
    
    if key == ord("q"):
        break
