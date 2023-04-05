import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy  # Move mouse and click

##########################
wCam, hCam = 640, 480   # Hegiht and width of camera
frameR = 100 # Frame Reduction
smoothening = 7
##########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Opening Camera
cap = cv2.VideoCapture(1)
cap.set(3, wCam)  # Width of the camera
cap.set(4, hCam)  # Height of the camera
detector = htm.handDetector(maxHands=1) #
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img) # Find hand using HandTrackingModule
    lmList, bbox = detector.findPosition(img) # Find position of hand using HandTrackingModule

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:] # lmList has list of landmarks with x and y coordinates, we use it to get coordinates of index and middle finger
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
    
    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
    (255, 0, 255), 2)

    # 4. Only Index Finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:

        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
    
        # 7. Move Mouse
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
        
    # 8. Both Index and middle fingers are up : Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)
        print(length)

        # 10. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
            15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
    
    # 11.Printing the Frame Rate
        # every time when the loop reaches the end some time is taken to complete all the operations 
        # so we will calculate the time taken to complete the loop (cTime - pTime) and calculate the frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime) # Frame per second
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, # font and size
    (255, 0, 0), 3)  # Color of the text

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)